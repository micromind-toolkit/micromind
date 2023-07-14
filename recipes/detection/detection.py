from micromind import PhiNet

from micromind import microYOLO
from micromind import Microhead


def train_nn():

    _alpha = 0.67
    _deeper_head = True

    _feature_sizes = [
        int(16 * _alpha / 0.67),
        int(32 * _alpha / 0.67),
        int(64 * _alpha / 0.67),
    ]

    if _deeper_head:
        _feature_sizes = [
            int(32 * _alpha / 0.67),
            int(64 * _alpha / 0.67),
            int(128 * _alpha / 0.67),
        ]

    # define backbone
    backbone = PhiNet(
        input_shape=(3, 320, 320),
        alpha=_alpha,
        num_layers=6 + (1 if _deeper_head else 0),
        beta=1,
        t_zero=4,
        include_top=False,
        num_classes=80,
        compatibility=True,
        downsampling_layers=[5, 7] + ([8] if _deeper_head else []),  # S2
        squeeze_excite=False,
    )

    # define head
    head = Microhead(
        feature_sizes=_feature_sizes,
        concat_layers=[6, 4, 12, 9],
        head_concat_layers=[18],
        number_heads=2,  # 2 stands as the second head is the detection head
        deeper_head=_deeper_head,
        no_SPPF=False,
    )

    # load a model
    model = microYOLO(
        backbone=backbone, head=head, task="detect", nc=80
    )  # build a new model from scratch DEFAULT_CFG

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
