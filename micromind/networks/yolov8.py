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
import cv2
import numpy as np
import torch
import time

from modules import *


if __name__ == "__main__":

    assert len(sys.argv) > 1
    if len(sys.argv) <= 2:
        print("Falling back on yolov8l model. Configuration was not passed.")
        print(
            "If you want to change configuration, give as argument one valid option between {n, s, m, l, x}."
        )
        print(
            "If you want to get the model weights file, launch load_params.py with the desired configuration"
        )
        conf = "l"
    else:
        conf = str(sys.argv[2])

    weights_file = f"yolov8{conf}.pt"
    output_folder_path = Path("./outputs_yolov8")
    output_folder_path.mkdir(parents=True, exist_ok=True)
    img_paths = [sys.argv[1]]
    for img_path in img_paths:
        image = [cv2.imread(img_path)]
        out_paths = [
            (
                output_folder_path
                / f"{Path(img_path).stem}_output{Path(img_path).suffix}"
            ).as_posix()
        ]
        if not isinstance(image[0], np.ndarray):
            print("Error in image loading. Check your image file.")
            sys.exit(1)

        pre_processed_image = preprocess(image)
        model = YOLOv8(*get_variant_multiples(conf), num_classes=80).half()
        model.load_state_dict(torch.load(weights_file), strict=True)
        print(f"Imported checkpoint {weights_file}")
        model = model.float()
        model.eval()

        st = time.time()

        with torch.no_grad():
            predictions = model(pre_processed_image)#.cpu()
            print(f"Did inference in {int(round(((time.time() - st) * 1000)))}ms")
            post_predictions = postprocess(
                preds=predictions, img=pre_processed_image, orig_imgs=image
            )

        class_labels = [s.strip() for s in open("../../examples/yolo_cfg/coco.names", "r").readlines()]
        draw_bounding_boxes_and_save(
            orig_img_paths=img_paths,
            output_img_paths=out_paths,
            all_predictions=post_predictions,
            class_labels=class_labels,
        )
