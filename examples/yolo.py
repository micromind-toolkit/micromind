from micromind import MicroMind
from micromind import Metric, Stage
from micromind.utils.yolo_helpers import postprocess, calculate_iou, average_precision, mean_average_precision, draw_bounding_boxes_and_save

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from micromind.utils.parse import parse_arguments

from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh, scale_boxes
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from micromind.networks.modules import YOLOv8


class Loss(v8DetectionLoss):
    def __init__(self, h, m, device):  # model must be de-paralleled
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with
        the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates
        from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box,
        cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class YOLO(MicroMind):
    def __init__(self, m_cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modules["yolo"] = YOLOv8(1, 1, 1, 80)

        self.m_cfg = m_cfg

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        preprocessed_batch = {}
        preprocessed_batch["img"] = (
            batch["img"].to(self.device, non_blocking=True).float() / 255
        )
        for k in batch:
            if isinstance(batch[k], torch.Tensor) and k != "img":
                preprocessed_batch[k] = batch[k].to(self.device)

        return preprocessed_batch

    def forward(self, batch):
        self.modules.eval()
        preprocessed_batch = self.preprocess_batch(batch)
        pred = self.modules["yolo"](preprocessed_batch["img"].to(self.device))

        return pred

    def compute_loss(self, pred, batch):
        self.criterion = Loss(self.m_cfg, self.modules["yolo"].head, self.device)
        preprocessed_batch = self.preprocess_batch(batch)

        lossi_sum, lossi = self.criterion(
            pred[1],
            preprocessed_batch,
        )

        return lossi_sum

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.modules.parameters(), lr=0.)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            "min",
            factor=0.2,
            patience=50,
            threshold=10,
            min_lr=0,
            verbose=True,
        )
        return opt, sched

    def mAP(self, pred, batch):

        preprocessed_batch = self.preprocess_batch(batch)
        post_predictions = postprocess(
            preds=pred[0].detach().cpu(), img=preprocessed_batch, orig_imgs=batch
        )

        # nel batch sono xywh con le dimensioni in 640x640
        batch_bboxes_xyxy = xywh2xyxy(batch["bboxes"])
        batch_bboxes_xyxy[:, :4] *= batch["resized_shape"][0][0] # *672

        #breakpoint()
        # TODO: NON TUTTE LE IMMAGINI HANNO UNA DIMENSIONE A 640 E VENGNON RIDIMENSIONATE AD ESEMPIO A (672, 512)
        # for i in range(len(batch["batch_idx"])):
        #     tmp = torch.tensor(batch["ori_shape"][int(batch["batch_idx"][i].item())])
        #     if torch.argmin(tmp)==0:
                
        #         batch_bboxes_xyxy[i][1] -= (640-tmp.min())/2
        #         batch_bboxes_xyxy[i][3] -= (640-tmp.min())/2
        #     else:
        #         batch_bboxes_xyxy[i][0] -= (640-tmp.min())/2
        #         batch_bboxes_xyxy[i][2] -= (640-tmp.min())/2
        # breakpoint()
        post_boxes = []
        for i in range(len(batch["batch_idx"])):
            for b in range(len(batch_bboxes_xyxy[batch["batch_idx"]==i, :])):
                post_boxes.append(scale_boxes(batch["resized_shape"][i], batch_bboxes_xyxy[batch["batch_idx"]==i, :][b], batch["ori_shape"][i]))
        post_boxes = torch.stack(post_boxes)
        #breakpoint()

        postpred = post_predictions.copy()
        print("RICOMINCIATOOOOOOOOO")
        breakpoint()
        mmAP = []
        print("BATCH_SIZE: ", batch_size)
        for batch_el in range(batch_size): # for every element in the batch
            print("BATCH_l: ", batch_el)
            ap_sum=0
            for class_id in range(80): # for every class in the dataset, compute the average precision for that class
                num_cls = torch.sum(batch["batch_idx"] == batch_el).item() # number of objs in the current batch element
                bbox_cls = post_boxes[batch["batch_idx"] == batch_el] # bboxes for those objs in the current batch element
                cls_cls = batch["cls"][batch["batch_idx"] == batch_el] # classes for those objs in the current batch element
                gt = torch.cat((bbox_cls, torch.ones((num_cls, 1)), cls_cls), dim=1) # ground_truth
                #breakpoint()
                ap = average_precision(post_predictions[batch_el], gt, class_id) # compute ap for a class for that batch element
                ap_sum += ap


            # NUMERO DI DIVERSE CLASSI PRESENTI NEL BATCH
            print("AP_SUM:  ", ap_sum, torch.unique(gt[:, -1]).size(0))
            
            mAP = ap_sum / torch.unique(gt[:, -1]).size(0)
            #mAP = ap_sum / torch.unique(postpred[batch_el][:, -1]).size(0)
            print(f"mAP_img{batch_el}", mAP)
            breakpoint()

            mmAP.append(mAP) 
        mmAP = sum(mmAP)/len(mmAP)
        #print("mmAP", mmAP)
        
        #ultra_metric.update(results=post_predictions)
        #m_test = ultra_metric.map50()
        return torch.Tensor([mmAP])


if __name__ == "__main__":
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.cfg import get_cfg

    m_cfg = get_cfg("yolo_cfg/default.yaml")
    data_cfg = check_det_dataset("yolo_cfg/coco.yaml")
    batch_size = 16

    mode = "train"
    coco8_dataset = build_yolo_dataset(
        m_cfg, "datasets/coco", batch_size, data_cfg, mode=mode, rect=mode == "val"
    )

    train_loader = DataLoader(
        coco8_dataset,
        batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=getattr(coco8_dataset, "collate_fn", None),
    )

    mode = "val"
    coco8_dataset = build_yolo_dataset(
        m_cfg, "datasets/coco", batch_size, data_cfg, mode=mode, rect=mode == "val"
    )

    val_loader = DataLoader(
        coco8_dataset,
        batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=getattr(coco8_dataset, "collate_fn", None),
    )

    hparams = parse_arguments()
    m = YOLO(m_cfg, hparams=hparams)

    m.train(
        epochs=50,
        datasets={"train": train_loader, "val": val_loader},
        # metrics = [map],
        debug=False,
    )

    # m.test(
    # datasets={"test": testloader},
    # )

