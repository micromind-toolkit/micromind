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
        """Initializes the YOLO model."""
        super().__init__(*args, **kwargs)
        self.m_cfg = m_cfg

        self.modules["backbone"] = PhiNet(
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

        heads = hparams.heads
        sppf_ch, neck_filters, up, head_filters = self.get_parameters(heads=heads)

        self.modules["sppf"] = SPPF(*sppf_ch)
        self.modules["neck"] = Yolov8Neck(filters=neck_filters, up=up, heads=heads)
        self.modules["head"] = DetectionHead(filters=head_filters, heads=heads)

        self.criterion = Loss(self.m_cfg, self.modules["head"], self.device)

        print("Number of parameters for each module:")
        print(self.compute_params())

    def get_parameters(self, heads=[True, True, True]):
        """
        Gets the parameters with which to initialize the network detection part
        (SPPF block, Yolov8Neck, DetectionHead).
        """
        in_shape = self.modules["backbone"].input_shape
        x = torch.randn(1, *in_shape)
        y = self.modules["backbone"](x)

        c1 = c2 = y[0].shape[1]
        sppf = SPPF(c1, c2)
        out_sppf = sppf(y[0])

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
        # keep only the heads we want
        head_filters = [head for heads, head in zip(heads, head_filters) if heads]

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
        """Runs the forward method by calling every module."""
        preprocessed_batch = self.preprocess_batch(batch)
        backbone = self.modules["backbone"](preprocessed_batch["img"].to(self.device))
        neck_input = backbone[1]
        neck_input.append(self.modules["sppf"](backbone[0]))
        neck = self.modules["neck"](*neck_input)
        head = self.modules["head"](neck)

        return head

    def compute_loss(self, pred, batch):
        """Computes the loss."""
        preprocessed_batch = self.preprocess_batch(batch)

        lossi_sum, lossi = self.criterion(
            pred,
            preprocessed_batch,
        )

        return lossi_sum

    def configure_optimizers(self):
        """Configures the optimizer and the scheduler."""
        opt = torch.optim.SGD(self.modules.parameters(), lr=1e-2, weight_decay=0.0005)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=14000, eta_min=1e-3
        )
        return opt, sched

    @torch.no_grad()
    def mAP(self, pred, batch):
        """Compute the mean average precision (mAP) for a batch of predictions.

        Arguments
        ---------
        pred : torch.Tensor
            Model predictions for the batch.
        batch : dict
            A dictionary containing batch information, including bounding boxes,
            classes and shapes.

        Returns
        -------
        torch.Tensor
            A tensor containing the computed mean average precision (mAP) for the batch.
        """
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

        batch_bboxes = torch.stack(batch_bboxes).to(self.device)
        mmAP = mean_average_precision(
            post_predictions, batch, batch_bboxes, data_cfg["nc"]
        )

        return torch.Tensor([mmAP])


def replace_datafolder(hparams, data_cfg):
    """Replaces the data root folder, if told to do so from the configuration."""
    data_cfg["path"] = str(data_cfg["path"])
    data_cfg["path"] = (
        data_cfg["path"][:-1] if data_cfg["path"][-1] == "/" else data_cfg["path"]
    )
    for key in ["train", "val"]:
        if not isinstance(data_cfg[key], list):
            data_cfg[key] = [data_cfg[key]]
        new_list = []
        for tmp in data_cfg[key]:
            if hasattr(hparams, "data_dir"):
                if hparams.data_dir != data_cfg["path"]:
                    tmp = str(tmp).replace(data_cfg["path"], "")
                    tmp = tmp[1:] if tmp[0] == "/" else tmp
                    tmp = os.path.join(hparams.data_dir, tmp)
                    new_list.append(tmp)
        data_cfg[key] = new_list

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
