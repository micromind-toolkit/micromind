# Ultralytics YOLO 🚀, AGPL-3.0 license, adaptation to micromind

from ultralytics.yolo.utils import RANK

from ultralytics.yolo.v8.detect import DetectionTrainer
from ..detection.detectionmicromodule import DetectionMicroModel


class DetectionMicroTrainer(DetectionTrainer):
    def get_model(self, backbone=None, head=None, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model especially adapted to work with the phinet
        backbone.
        """
        model = DetectionMicroModel(
            backbone=backbone,
            head=head,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model
