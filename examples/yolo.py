from micromind import MicroMind

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss, TaskAlignedAssigner

import sys; sys.path.append("/home/franz/dev/micromind/yolo_teo")
from modules import YOLOv8

class Loss(v8DetectionLoss):
    def __init__(self, h, m, device):    # model must be de-paralleled
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

class YOLO(MicroMind):
    def __init__(self, m_cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modules["yolo"] = YOLOv8(
            1, 1, 1, num_classes=80
        )

        self.modules["yolo"].load_state_dict(
            torch.load("/home/franz/dev/micromind/yolo_teo/yolov8l.pt"
        ))

        self.m_cfg = m_cfg

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        for k in batch:
            if not k in ["im_file", "ori_shape", "resized_shape"]:
                batch[k] = batch[k].to(self.device)
        return batch

    def forward(self, batch):
        batch = self.preprocess_batch(batch)

        return self.modules["yolo"](batch["img"].to(self.device))

    def compute_loss(self, pred, batch):
        self.criterion = Loss(self.m_cfg, self.modules["yolo"].head, self.device)
        batch = self.preprocess_batch(batch)

        lossi_sum, lossi =  self.criterion(
            pred[1],    # pass elements at the beginning of the backward graph
            batch
        )

        return lossi_sum

    def configure_optimizers(self):
        return torch.optim.Adam(self.modules.parameters(), lr=0.001), None


if __name__ == "__main__":
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.cfg  import get_cfg

    m_cfg = get_cfg("examples/yolo_cfg/default.yaml")
    data_cfg = check_det_dataset("examples/yolo_cfg/coco8.yaml")
    batch_size = 32

    coco8_dataset = build_yolo_dataset(
        m_cfg, "/mnt/data/coco8", batch_size, data_cfg
    )

    loader = DataLoader(
        coco8_dataset, batch_size, collate_fn=getattr(coco8_dataset, 'collate_fn', None)
    )

    m = YOLO(
        m_cfg
    )

    m.train(
        epochs=3,
        datasets={"train": loader, "val": loader},
    )

    # m.test(
        # datasets={"test": testloader},
    # )

