# Ultralytics YOLO 🚀, AGPL-3.0 license, adaptation to micromind

from ultralytics.yolo.utils import RANK

from ultralytics.yolo.v8.detect import DetectionTrainer
from modules.detectionheadmodule import DetectionHeadModel

class DetectionHeadTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionHeadModel(nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model