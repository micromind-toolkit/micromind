"""
YOLOv8 inference.

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023

This code allows you to launch an object detection inference using the yolov8 model.
To launch this script, pass as arguments the path of the image on which to perform
the inference and the configuration of the model to use
(available options are "n", "s", "m", "l", "x").
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import time
import torchvision

from micromind.networks.yolov8 import SPPF, Yolov8Neck, DetectionHead
from micromind.networks.phinet import PhiNet
from micromind.core import MicroMind
from micromind.utils.yolo_helpers import (
    preprocess,
    postprocess,
    draw_bounding_boxes_and_save,
)


class Detect(nn.Module):
    def __init__(self, back, sppf, neck, head):
        super().__init__()
        self.backbone = back
        self.sppf = sppf
        self.neck = neck
        self.head = head

    def forward(self, x):
        backbone = self.backbone(x)[1]
        backbone[-1] = self.sppf(backbone[-1])
        neck = self.neck(*backbone)
        head = self.head(neck)
        return head


class ObjectDetectionInference(MicroMind):
    """Implements an object detectin network through MicroMind"""

    def __init__(self, backbone, sppf, neck, head):
        """Defines the structure of the network.

        Arguments
        ---------
        backbone : nn.Module
            Backbone of the network.
        sppf : nn.Module
            SPPF block of the network.
        neck : nn.Module
            Neck of the network.
        head : nn.Module
            Detection head of the network.
        """
        super().__init__()
        self.modules["backbone"] = backbone
        self.modules["sppf"] = sppf
        self.modules["neck"] = neck
        self.modules["head"] = head

    @torch.no_grad()
    def forward(self, x):
        """Executes the detection network.

        Arguments
        ---------
        x : torch.Tensor
            Input to the detection network.

        Returns
        -------
            Output of the detection network : torch.Tensor
        """
        backbone = self.modules["backbone"](x)[1]
        backbone[-1] = self.modules["sppf"](backbone[-1])
        neck = self.modules["neck"](*backbone)
        head = self.modules["head"](neck)
        return head

    def compute_loss(self):
        """Since we only want to get the output of the network, we are not interested in calculating the loss."""
        pass


if __name__ == "__main__":
    assert len(sys.argv) > 1
    if len(sys.argv) <= 2:
        print("Falling back on yolov8l model. Configuration was not passed.")
        print(
            "If you want to change configuration, "
            + "give as argument one valid option between {n, s, m, l, x}."
        )
        print(
            "If you want to get the model weights file, "
            + "launch load_params.py with the desired configuration."
        )
        conf = "l"
    else:
        conf = str(sys.argv[2])

    # weights_file = f"yolov8{conf}.pt"
    weights_file = "./results/exp/save/epoch_16_val_loss_46.9538.ckpt"
    output_folder_path = Path("./outputs_yolov8")
    output_folder_path.mkdir(parents=True, exist_ok=True)
    img_paths = [sys.argv[1]]
    for img_path in img_paths:
        image = torchvision.io.read_image(img_path)
        out_paths = [
            (
                output_folder_path
                / f"{Path(img_path).stem}_output{Path(img_path).suffix}"
            ).as_posix()
        ]
        if not isinstance(image, torch.Tensor):
            print("Error in image loading. Check your image file.")
            sys.exit(1)

        pre_processed_image = preprocess(image)
        dict_ = torch.load(weights_file)

        phinet = PhiNet(
            input_shape=(3, 672, 672),
            alpha=3,
            num_layers=7,
            beta=1,
            t_zero=6,
            include_top=False,
            compatibility=False,
            divisor=8,
            downsampling_layers=[4, 5, 7],
            return_layers=[5, 6, 7],
        )
        sppf = SPPF(576, 576)
        neck = Yolov8Neck([144, 288, 576])
        head = DetectionHead(80, filters=(144, 288, 576))

        phinet.load_state_dict(dict_["phinet"], strict=True)
        sppf.load_state_dict(dict_["sppf"], strict=True)
        neck.load_state_dict(dict_["neck"], strict=True)
        head.load_state_dict(dict_["head"], strict=True)
        print(f"Imported checkpoint {weights_file}")

        model = ObjectDetectionInference(phinet, sppf, neck, head)
        # model.load_modules(weights_file)
        # model.modules.eval()
        # model.modules.float()
        model.eval()

        st = time.time()

        with torch.no_grad():
            predictions = model(pre_processed_image)  # .cpu()
            print(f"Did inference in {int(round(((time.time() - st) * 1000)))}ms")
            post_predictions = postprocess(
                preds=predictions[0], img=pre_processed_image, orig_imgs=image
            )

        class_labels = [s.strip() for s in open("cfg/coco.names", "r").readlines()]
        draw_bounding_boxes_and_save(
            orig_img_paths=img_paths,
            output_img_paths=out_paths,
            all_predictions=post_predictions,
            class_labels=class_labels,
        )
