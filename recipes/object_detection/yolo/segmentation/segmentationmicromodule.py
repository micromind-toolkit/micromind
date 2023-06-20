# Ultralytics YOLO custom model DetectionHeadModule.py

from ..detection.detectionmicromodule import DetectionMicroModel

from ultralytics.yolo.utils.loss import v8SegmentationLoss

from ultralytics.yolo.utils import LOGGER


class SegmentationMicroModel(DetectionMicroModel):
    """YOLOv8 custom detection model for micromind backbone."""

    def __init__(self, backbone=None, head=None, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, backbone=backbone, head=head, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8SegmentationLoss(self)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} has not supported augment inference"
            f" yet! Now using single-scale inference instead."
        )
        return self._predict_once(x)
