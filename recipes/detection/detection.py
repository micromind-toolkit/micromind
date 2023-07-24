from micromind import PhiNet

from micromind import microYOLO
from micromind import Microhead

import math


def train_nn():

    # note for 0.33 -> write 0.335 for the round up error
    # if you get an error about shape incompatibility, try to slightly change the alpha
    #  value
    _alpha = 0.67
    _deeper_head = True

    _feature_sizes = [
        math.ceil(16 * _alpha / 0.67),
        math.ceil(32 * _alpha / 0.67),
        math.ceil(64 * _alpha / 0.67),
    ]

    if _deeper_head:
        _feature_sizes = [
            math.ceil(32 * _alpha / 0.67),
            math.ceil(64 * _alpha / 0.67),
            math.ceil(128 * _alpha / 0.67),
        ]

    # the backbone is PhiNet
    # it changes the number of layers according to the alpha value
    # also the number of downsampling layers changes according to the deeper_layer flag
    #  (from 6 to 7)
    # the squeeze and excite block is not used for latency speedup
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

    # the head is Microhead
    # you can change the number of feature sizes according to the number of downsampling
    # layers
    # usually the number of detection heads is 3, but it can be changed according to the
    # task
    # the fewer the detection heads, the more the speedup
    # to choose which head to use, you need to change the number of heads in the
    # number_heads parameter
    # also you need to change the head_concat_layers parameter to match the number of
    # heads
    # e.g. number_heads=2, head_concat_layers=[15, 18]
    #      number_heads=1, head_concat_layers=[15]
    head = Microhead(
        feature_sizes=_feature_sizes,
        concat_layers=[6, 4, 12, 9],
        head_concat_layers=[15, 18, 21],
        heads_used=[1, 1, 1],
        deeper_head=_deeper_head,
        no_SPPF=False,
    )

    # load a model
    model = microYOLO(
        backbone=backbone, head=head, task="detect", nc=80
    )  # build a new model from scratch DEFAULT_CFG

    # Train the model
    # Change the parameters also here
    # These parameters are the most lightweight possible in terms of training
    #
    model.train(
        data="coco128.yaml",
        epochs=1,
        imgsz=320,
        device="cpu",
        task="detect",
    )


if __name__ == "__main__":
    train_nn()
