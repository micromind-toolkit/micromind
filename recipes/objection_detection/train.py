"""
YOLO training.

This code allows you to train an object detection model with the YOLOv8 neck and loss.

To run this script, you can start it with:
    python train.py cfg/yolo_phinet.py

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023
"""

import torch
from prepare_data import create_loaders
from torchinfo import summary
from ultralytics.utils.ops import scale_boxes, xywh2xyxy
from yolo_loss import Loss

import micromind as mm
from micromind.networks import PhiNet
from micromind.networks.yolo import SPPF, DetectionHead, Yolov8Neck
from micromind.utils import parse_configuration
from micromind.utils.yolo import (
    load_config,
    mean_average_precision,
    postprocess,
)
import sys
import os


class YOLO(mm.MicroMind):
    def __init__(self, m_cfg, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modules["phinet"] = PhiNet(
            input_shape=hparams.input_shape,
            alpha=hparams.alpha,
            num_layers=hparams.num_layers,
            beta=hparams.beta,
            t_zero=hparams.t_zero,
            include_top=False,
            compatibility=False,
            divisor=hparams.divisor,
            downsampling_layers=hparams.downsampling_layers,
            return_layers=hparams.return_layers,
        )

        sppf_ch, neck_filters, up, head_filters = self.get_parameters()

        self.modules["sppf"] = SPPF(*sppf_ch)
        self.modules["neck"] = Yolov8Neck(filters=neck_filters, up=up)
        self.modules["head"] = DetectionHead(filters=head_filters)

        tot_params = 0
        for m in self.modules.values():
            temp = summary(m, verbose=0)
            tot_params += temp.total_params

        print(f"Total parameters of model: {tot_params * 1e-6:.2f} M")

        self.m_cfg = m_cfg

    def get_parameters(self):
        """
        Gets the parameters with which to initialize the network detection part
        (SPPF block, Yolov8Neck, DetectionHead).
        """
        in_shape = self.modules["phinet"].input_shape
        x = torch.randn(1, *in_shape)
        y = self.modules["phinet"](x)

        c1 = c2 = y[1][2].shape[1]
        sppf = SPPF(c1, c2)
        out_sppf = sppf(y[1][2])

        neck_filters = [y[1][0].shape[1], y[1][1].shape[1], out_sppf.shape[1]]
        up = [2, 2]
        up[0] = y[1][1].shape[2] / out_sppf.shape[2]
        up[1] = y[1][0].shape[2] / (up[0] * out_sppf.shape[2])
        temp = """The layers you selected are not valid. \
            Please choose only layers between which the spatial resolution \
            doubles every time. Eventually, you can achieve this by \
            changing the downsampling layers."""

        assert up == [2, 2], " ".join(temp.split())

        neck = Yolov8Neck(filters=neck_filters, up=up)
        out_neck = neck(y[1][0], y[1][1], out_sppf)

        head_filters = (
            out_neck[0].shape[1],
            out_neck[1].shape[1],
            out_neck[2].shape[1],
        )
        # head = DetectionHead(filters=head_filters)

        return (c1, c2), neck_filters, up, head_filters

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
        opt = torch.optim.SGD(self.modules.parameters(), lr=1e-2, weight_decay=0.0005)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=14000, eta_min=1e-3
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


def replace_datafolder(hparams, data_cfg):
    """Replaces the data root folder, if told to do so from the configuration."""
    data_cfg["path"] = str(data_cfg["path"])
    data_cfg["path"] = (
        data_cfg["path"][:-1] if data_cfg["path"][-1] == "/" else data_cfg["path"]
    )
    for key in ["train", "val"]:
        if hasattr(hparams, "data_dir"):
            if hparams.data_dir != data_cfg["path"]:
                data_cfg[key] = str(data_cfg[key]).replace(data_cfg["path"], "")
                data_cfg[key] = (
                    data_cfg[key][1:] if data_cfg[key][0] == "/" else data_cfg[key]
                )
                data_cfg[key] = os.path.join(hparams.data_dir, data_cfg[key])

    data_cfg["path"] = hparams.data_dir

    return data_cfg


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please pass the configuration file to the script."
    hparams = parse_configuration(sys.argv[1])

    m_cfg, data_cfg = load_config(hparams.data_cfg)

    # check if specified path for images is different, correct it in case
    data_cfg = replace_datafolder(hparams, data_cfg)

    train_loader, val_loader = create_loaders(m_cfg, data_cfg, hparams.batch_size)

    exp_folder = mm.utils.checkpointer.create_experiment_folder(
        hparams.output_folder, hparams.experiment_name
    )

    checkpointer = mm.utils.checkpointer.Checkpointer(
        exp_folder, hparams=hparams, key="loss"
    )

    yolo_mind = YOLO(m_cfg, hparams=hparams)

    mAP = mm.Metric("mAP", yolo_mind.mAP, eval_only=True, eval_period=1)

    yolo_mind.train(
        epochs=200,
        datasets={"train": train_loader, "val": val_loader},
        metrics=[mAP],
        checkpointer=checkpointer,
        debug=hparams.debug,
    )
