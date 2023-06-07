# Ultralytics YOLO 🚀, AGPL-3.0 license, adaptation to micromind

from ultralytics.yolo.utils import DEFAULT_CFG, RANK

from ultralytics.yolo.v8.detect import DetectionTrainer
from modules.detectionheadmodule import DetectionHeadModel


# BaseTrainer python usage
class DetectionMicroTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionHeadModel(nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model or "yolov8n.pt"
    data = cfg.data or "coco128.yaml"  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ""

    # parse arguments here from sh command
    args = dict(model=model, data=data, device=device, epochs=300, imgsz=320)
    trainer = DetectionMicroTrainer(overrides=args)
    trainer.train()


if __name__ == "__main__":
    train()
