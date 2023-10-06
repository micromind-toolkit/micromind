from micromind import MicroMind
from micromind import PhiNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class ImageClassification(MicroMind):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modules["classifier"] = PhiNet(
                (3, 32, 32),
                include_top=True,
                num_classes=10
            )

    def forward(self, batch):
        return self.modules["classifier"](batch[0])

    def compute_loss(self, pred, batch):
        return nn.CrossEntropyLoss()(pred, batch[1])


if __name__ == "__main__":
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.cfg  import get_cfg

    m_cfg = get_cfg("examples/yolo_cfg/default.yaml")
    data_cfg = check_det_dataset("examples/yolo_cfg/coco8.yaml")
    batch_size = 32

    build_yolo_dataset(
        m_cfg, "/mnt/data/coco8", batch_size, data_cfg
    )

    # m = ImageClassification()
# 
    # m.train(
        # epochs=10,
        # datasets={"train": trainloader, "val": testloader, "test": testloader},
        # debug=False
    # )
# 
    # m.test(
        # datasets={"test": testloader},
    # )

