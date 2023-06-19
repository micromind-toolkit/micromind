# Ultralytics YOLO 🚀, AGPL-3.0 license, adaptation to micromind

from ultralytics.yolo.utils import RANK

from ultralytics.yolo.v8.segment import SegmentationTrainer
from .segmentationmicromodule import SegmentationMicroModel


class SegmentationMicroTrainer(SegmentationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model especially adapted to work with the phinet
        backbone.
        """
        model = SegmentationMicroModel(
            nc=self.data["nc"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model
