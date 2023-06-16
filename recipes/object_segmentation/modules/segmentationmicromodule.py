# Ultralytics YOLO custom model DetectionHeadModule.py

import contextlib
from pathlib import Path

import torch
import torch.nn as nn

from micromind import PhiNet

from object_detection.modules.detectionmicromodule import DetectionMicroModel

from ultralytics.yolo.utils.loss import v8SegmentationLoss

from ultralytics.nn.modules import (
    Detect,
    Pose,
    Segment,
)
from ultralytics.yolo.utils import (
    LOGGER,
    yaml_load,
)
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.torch_utils import initialize_weights

from micromind.networks.microyolohead import Microhead

class SegmentationMicroModel(DetectionMicroModel):
    """YOLOv8 custom detection model for micromind backbone."""

    def __init__(self, cfg='yolov8n-seg.yaml', ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8SegmentationLoss(self)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)
