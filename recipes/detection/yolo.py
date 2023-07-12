from ultralytics import YOLO

def train_nn():

    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Train the model
    model.train(
        data="coco128.yaml",
        epochs=1,
        imgsz=320,
        device="cpu",
        task="detect",
    )

if __name__ == "__main__":
    train_nn()
