import os

from micromind import PhiNet

from yolo.model import microYOLO
from yolo.microyolohead import Microhead


def train_nn():

    # define backbone
    backbone = PhiNet(
        input_shape=(3, 320, 320),
        alpha=0.67,
        num_layers=6,
        beta=1,
        t_zero=4,
        include_top=False,
        num_classes=80,
        compatibility=False,
    )
    # define head
    head = Microhead(task="segment")

    # load a model
    model = microYOLO(
        backbone=backbone, head=head, task="segment", nc=80
    )  # build a new model from scratch DEFAULT_CFG

    # Train the model
    model.train(
        data=get_dataset("coco-seg.yaml"), epochs=50, imgsz=320, device="cpu", batch=4
    )
    model.export()


def get_dataset(dataset_name):
    cwd = os.getcwd()
    dataset_path = os.path.join(cwd, "datasets")
    if not os.path.exists(dataset_path):
        raise Exception("Dataset not found")
    else:
        return os.path.join(dataset_path, dataset_name)


if __name__ == "__main__":
    train_nn()
