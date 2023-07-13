from ultralytics import YOLO
from torchinfo import summary


def train_nn():

    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    summary(model.model, input_size=(1, 3, 320, 320), depth=5, verbose=1)

    # Train the model
    """
    model.train(
        data="coco128.yaml",
        epochs=1,
        imgsz=320,
        device="cpu",
        task="detect",
    )
    """


if __name__ == "__main__":
    train_nn()
