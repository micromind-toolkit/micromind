from micromind import PhiNet
from micromind import Microhead

from micromind.yolo.model import microYOLO

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
    # model = microYOLO('weights.pt') # load a pretrained model (recommended for training)
    
    # Train the model
    model.train(data="coco128-seg.yaml", epochs=5, imgsz=320, device="cpu")
    model.export()

if __name__ == "__main__":
    train_nn()
