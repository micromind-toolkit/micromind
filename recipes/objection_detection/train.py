import torch
from prepare_data import create_loaders
from torchinfo import summary
from ultralytics.utils.ops import scale_boxes, xywh2xyxy
from yolo_loss import Loss

import micromind as mm
from micromind.networks import PhiNet
from micromind.networks.yolov8 import SPPF, DetectionHead, Yolov8Neck
from micromind.utils.parse import parse_arguments
from micromind.utils.yolo_helpers import (
    load_config,
    mean_average_precision,
    postprocess,
)


class YOLO(mm.MicroMind):
    def __init__(self, m_cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modules["phinet"] = PhiNet(
            input_shape=(3, 672, 672),
            alpha=3,
            num_layers=7,
            beta=1,
            t_zero=6,
            include_top=False,
            compatibility=False,
            divisor=8,
            downsampling_layers=[4, 5, 7],  # check consistency with return_layers
            return_layers=[5, 6, 7],
        )
        self.modules["sppf"] = SPPF(576, 576)
        self.modules["neck"] = Yolov8Neck([144, 288, 576])
        self.modules["head"] = DetectionHead(filters=(144, 288, 576))

        tot_params = 0
        for m in self.modules.values():
            temp = summary(m, verbose=0)
            tot_params += temp.total_params

        print(f"Total parameters of model: {tot_params*1e-6:.2f} M")

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
        preprocessed_batch = self.preprocess_batch(batch)
        # pred = self.modules["yolo"](preprocessed_batch["img"].to(self.device))
        backbone = self.modules["phinet"](preprocessed_batch["img"].to(self.device))[1]
        backbone[-1] = self.modules["sppf"](backbone[-1])
        neck = self.modules["neck"](*backbone)
        head = self.modules["head"](neck)

        return head

    def compute_loss(self, pred, batch):
        self.criterion = Loss(self.m_cfg, self.modules["head"], self.device)
        preprocessed_batch = self.preprocess_batch(batch)

        lossi_sum, lossi = self.criterion(
            pred[1],
            preprocessed_batch,
        )

        return lossi_sum

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.modules.parameters(), lr=1e-3)
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

    @torch.no_grad()
    def mAP(self, pred, batch):
        preprocessed_batch = self.preprocess_batch(batch)
        post_predictions = postprocess(
            preds=pred[0], img=preprocessed_batch, orig_imgs=batch
        )

        batch_bboxes_xyxy = xywh2xyxy(batch["bboxes"])
        dim = batch["resized_shape"][0][0]
        batch_bboxes_xyxy[:, :4] *= dim

        batch_bboxes = []
        for i in range(len(batch["batch_idx"])):
            for b in range(len(batch_bboxes_xyxy[batch["batch_idx"] == i, :])):
                batch_bboxes.append(
                    scale_boxes(
                        batch["resized_shape"][i],
                        batch_bboxes_xyxy[batch["batch_idx"] == i, :][b],
                        batch["ori_shape"][i],
                    )
                )
        batch_bboxes = torch.stack(batch_bboxes)
        mmAP = mean_average_precision(post_predictions, batch, batch_bboxes)

        return torch.Tensor([mmAP])


if __name__ == "__main__":
    batch_size = 8
    hparams = parse_arguments()

    m_cfg, data_cfg = load_config("cfg/coco8.yaml")
    train_loader, val_loader = create_loaders(m_cfg, data_cfg, batch_size)

    exp_folder = mm.utils.checkpointer.create_experiment_folder(
        hparams.output_folder, hparams.experiment_name
    )

    checkpointer = mm.utils.checkpointer.Checkpointer(exp_folder, key="loss")

    yolo_mind = YOLO(m_cfg, hparams=hparams)

    mAP = mm.Metric("mAP", yolo_mind.mAP, eval_only=True, eval_period=1)

    yolo_mind.train(
        epochs=20,
        datasets={"train": train_loader, "val": val_loader},
        metrics=[mAP],
        checkpointer=checkpointer,
        debug=hparams.debug,
    )
