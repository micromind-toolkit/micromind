"""
YOLOv8 inference.

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023

This code allows you to launch an object detection inference using a YOLO MicroMind.

To run this script, you should pass the checkpoint with the weights and the path to
an image, following this example:
    python inference.py model.ckpt cat.png
"""

import sys
import time
from pathlib import Path

import torch
import torchvision

from micromind.utils.yolo import (
    draw_bounding_boxes_and_save,
    postprocess,
    preprocess,
)
from train import YOLO


class Inference(YOLO):
    def __init__(self):
        super().__init__(m_cfg={})

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
        backbone = self.modules["phinet"](x)[1]
        backbone[-1] = self.modules["sppf"](backbone[-1])
        neck = self.modules["neck"](*backbone)
        head = self.modules["head"](neck)
        return head


if __name__ == "__main__":
    assert len(sys.argv) > 1

    weights_file = sys.argv[1]
    output_folder_path = Path("./outputs_yolov8")
    output_folder_path.mkdir(parents=True, exist_ok=True)
    img_paths = [sys.argv[2]]
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

        model = Inference()
        model.eval()
        model.load_modules(weights_file)

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
