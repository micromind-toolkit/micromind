"""
YOLO training.

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023

This code allows you to train an object detection model with the YOLOv8 neck and loss.

To run this script, you can start it with:
    python train.py
"""

import torch
import torch.nn as nn
from prepare_data import create_loaders
from torchinfo import summary
from timm.loss import (
    BinaryCrossEntropy,
    JsdCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)

import micromind as mm
from micromind.networks import PhiNet
from micromind.utils import parse_configuration
import sys


class ImageClassification(mm.MicroMind):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modules["classifier"] = PhiNet(
            input_shape=hparams.input_shape,
            alpha=hparams.alpha,
            num_layers=hparams.num_layers,
            beta=hparams.beta,
            t_zero=hparams.t_zero,
            compatibility=False,
            divisor=hparams.divisor,
            downsampling_layers=hparams.downsampling_layers,
            return_layers=hparams.return_layers,
            # classification-specific
            include_top=True,
            num_classes=hparams.num_classes
        )

        tot_params = 0
        for m in self.modules.values():
            temp = summary(m, verbose=0)
            tot_params += temp.total_params

        print(f"Total parameters of model: {tot_params * 1e-6:.2f} M")

    def setup_criterion(self):
        """ Setup of the loss function based on augmentation strategy. """
        # setup loss function
        if self.hparams.jsd_loss:
            assert num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(
                num_splits=num_aug_splits, smoothing=self.hparams.smoothing
            )
        elif self.hparams.mixup > 0 or self.hparams.cutmix > 0.0 or self.hparams.cutmix_minmax is not None:
            # smoothing is handled with mixup target transform which outputs sparse,
            # soft targets
            if self.hparams.bce_loss:
                train_loss_fn = BinaryCrossEntropy(target_threshold=self.hparams.bce_target_thresh)
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif self.hparams.smoothing:
            if self.hparams.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    smoothing=self.hparams.smoothing, target_threshold=self.hparams.bce_target_thresh
                )
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=self.hparams.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()

        return train_loss_fn

    def forward(self, batch):
        if not self.hparams.prefetcher:
            img, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                img, target = mixup_fn(img, target)
        else:
            img, target = batch

        return (self.modules["classifier"](img), target)

    def compute_loss(self, pred, batch):
        self.criterion = self.setup_criterion()

        # taking it from pred because it might be augmented
        return self.criterion(pred[0], pred[1])

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.modules.parameters(), lr=1e-2, weight_decay=0.0005)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=5000, eta_min=1e-7
        )
        return opt, sched


def top_k_accuracy( k=1):
    """
    Computes the top-K accuracy.

    Arguments
    ---------
    k : int
       Number of top elements to consider for accuracy.

    Returns
    -------
        accuracy : Callable
            Top-K accuracy.
    """
    def acc(pred, batch):
        _, indices = torch.topk(pred[0], k, dim=1)
        correct = torch.sum(indices == pred[1].view(-1, 1))
        accuracy = correct.item() / pred[1].size(0)
        
        return torch.Tensor([accuracy]).to(pred[0].device)

    return acc


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please pass the configuration file to the script."
    hparams = parse_configuration(sys.argv[1])

    train_loader, val_loader = create_loaders(hparams)

    exp_folder = mm.utils.checkpointer.create_experiment_folder(
        hparams.output_folder, hparams.experiment_name
    )

    checkpointer = mm.utils.checkpointer.Checkpointer(exp_folder, key="loss")

    yolo_mind = ImageClassification(hparams=hparams)

    mAP = mm.Metric("top1_acc", top_k_accuracy(k=1))
    mAP = mm.Metric("top5_acc", top_k_accuracy(k=5))

    yolo_mind.train(
        epochs=10,
        datasets={"train": train_loader, "val": val_loader},
        metrics=[mAP],
        checkpointer=checkpointer,
        debug=hparams.debug,
    )
