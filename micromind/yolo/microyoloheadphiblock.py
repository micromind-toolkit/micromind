import torch
import torch.nn as nn

from micromind.networks.phinet import PhiNetConvBlock

from ultralytics.nn.modules import SPPF, Concat, Conv, Detect, Segment


class Microhead(nn.Module):
    def __init__(
        self,
        nc=80,
        number_heads=3,
        feature_sizes=[16, 32, 64],
        concat_layers=[6, 4, 12, 9],  # following the row in the yolov8 architecture
        head_concat_layers=[15, 18, 21],
        task="detect",
        deeper_head=False,
        no_SPPF=False,
    ) -> None:
        """This class represents the implementation of a head.

        In particular it is an adaptation of the yoloV8 head for the micromind toolkit.
        This head is going to be used with the phinet backbone in a
        particular configuration so to match the input size, and the
        various intermediate steps of the network.

        The head is responsible for processing the intermediate features
        of the neural network and generating detections. It takes input from
        multiple feature levels and performs operations such as
        upsampling, concatenation, and convolution.

        Args:
            nc (int, required): Number of classes to be detected. Defaults to 80.
            number_heads (int, required): Number of heads to be used. Defaults to 3.
                (between 1 and 3)
            feature_sizes (list, required): List of the intermediary feature.
                Defaults to [64, 128, 256].
                Note: the number of features will be added to the network from
                the last one to the first. For example if the list has only one
                element, only the last concatenation in sequence of the head will
                be made. If the list has two elements, the last two concatenations
                will be made, and so on.
            concat_layers (list, required): List of the layers to be concatenated,
                in the head layers (one for every scale). Defaults to [6, 4, 12].
            head_concat_layers (list, required): List of the layers where the heads
                have to be connected. Defaults to [15, 18, 21].

        Returns:
            None

        """

        head_concat_layers = [
            x - 1 if no_SPPF else x + 1 if deeper_head else x
            for x in head_concat_layers
        ]
        scale_deep = 0.5 if deeper_head else 1

        # some errors checks
        if number_heads not in [1, 2, 3]:
            raise ValueError("Number of heads must be between 1 and 3")
        if nc < 1:
            raise ValueError("Number of classes must be greater than 0")
        if any([x < 1 and x > 1024 for x in feature_sizes]):
            raise ValueError("Feature sizes must be greater than 0")
        if task not in ["detect", "segment"]:
            raise ValueError("The task specified is not supported")

        super().__init__()
        self._layers = torch.nn.ModuleList()
        self._save = []

        if not no_SPPF:
            layer9 = SPPF(feature_sizes[-1], feature_sizes[-1], 5)
            layer9.i, layer9.f, layer9.type, layer9.n = (
                9 + (1 if deeper_head else 0),
                -1,
                "ultralytics.nn.modules.block.SPPF",
                1,
            )
            self._layers.append(layer9)
            self._save.extend(
                x % layer9.i
                for x in ([layer9.f] if isinstance(layer9.f, int) else layer9.f)
                if x != -1
            )

        layer10 = nn.Upsample(scale_factor=2, mode="nearest")
        layer10.i, layer10.f, layer10.type, layer10.n = (
            10 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
            -1,
            "torch.nn.modules.upsampling.Upsample",
            1,
        )
        self._layers.append(layer10)
        self._save.extend(
            x % layer10.i
            for x in ([layer10.f] if isinstance(layer10.f, int) else layer10.f)
            if x != -1
        )

        if number_heads >= 1:
            layer11 = Concat(dimension=1)
            layer11.i, layer11.f, layer11.type, layer11.n = (
                11 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                [-1, concat_layers[0] + (2 if deeper_head else 0)],
                "ultralytics.nn.modules.conv.Concat",
                1,
            )
            self._layers.append(layer11)
            self._save.extend(
                x % layer11.i
                for x in ([layer11.f] if isinstance(layer11.f, int) else layer11.f)
                if x != -1
            )

            layer12 = PhiNetConvBlock(
                in_shape=(
                    feature_sizes[1] + feature_sizes[2],
                    20 * scale_deep,
                    20 * scale_deep,
                ),
                stride=1,
                filters=feature_sizes[1],
                expansion=0.5,
                has_se=False,
                block_id=12,
            )
            layer12.i, layer12.f, layer12.type, layer12.n = (
                12 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                -1,
                "micromind.networks.PhiNetConvBlock",
                3,
            )
            self._layers.append(layer12)
            self._save.extend(
                x % layer12.i
                for x in ([layer12.f] if isinstance(layer12.f, int) else layer12.f)
                if x != -1
            )

            layer13 = nn.Upsample(scale_factor=2, mode="nearest")
            layer13.i, layer13.f, layer13.type, layer13.n = (
                13 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                -1,
                "torch.nn.modules.upsampling.Upsample",
                1,
            )
            self._layers.append(layer13)
            self._save.extend(
                x % layer13.i
                for x in ([layer13.f] if isinstance(layer13.f, int) else layer13.f)
                if x != -1
            )

            layer14 = Concat(1)
            layer14.i, layer14.f, layer14.type, layer14.n = (
                14 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                [-1, concat_layers[1] + (2 if deeper_head else 0)],
                "ultralytics.nn.modules.conv.Concat",
                1,
            )
            self._layers.append(layer14)
            self._save.extend(
                x % layer14.i
                for x in ([layer14.f] if isinstance(layer14.f, int) else layer14.f)
                if x != -1
            )

            layer15 = PhiNetConvBlock(
                in_shape=(
                    feature_sizes[0] + feature_sizes[1],
                    40 * scale_deep,
                    40 * scale_deep,
                ),
                stride=1,
                filters=feature_sizes[0],
                expansion=0.5,
                has_se=False,
                block_id=12,
            )
            layer15.i, layer15.f, layer15.type, layer15.n = (
                15 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                -1,
                "micromind.networks.PhiNetConvBlock",
                3,
            )
            self._layers.append(layer15)
            self._save.extend(
                x % layer15.i
                for x in ([layer15.f] if isinstance(layer15.f, int) else layer15.f)
                if x != -1
            )

            if number_heads >= 2:
                layer16 = Conv(feature_sizes[0], feature_sizes[0], 3, 2)
                layer16.i, layer16.f, layer16.type, layer16.n = (
                    16 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                    -1,
                    "ultralytics.nn.modules.conv.Conv",
                    1,
                )
                self._layers.append(layer16)
                self._save.extend(
                    x % layer16.i
                    for x in ([layer16.f] if isinstance(layer16.f, int) else layer16.f)
                    if x != -1
                )

                layer17 = Concat(1)
                layer17.i, layer17.f, layer17.type, layer17.n = (
                    17 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                    [
                        -1,
                        concat_layers[2]
                        + (1 if deeper_head else 0)
                        + (-1 if no_SPPF else 0),
                    ],
                    "ultralytics.nn.modules.conv.Concat",
                    1,
                )
                self._layers.append(layer17)
                self._save.extend(
                    x % layer17.i
                    for x in ([layer17.f] if isinstance(layer17.f, int) else layer17.f)
                    if x != -1
                )

                layer18 = PhiNetConvBlock(
                    in_shape=(
                        feature_sizes[0] + feature_sizes[1],
                        40 * scale_deep,
                        40 * scale_deep,
                    ),
                    stride=1,
                    filters=feature_sizes[1],
                    expansion=0.5,
                    has_se=False,
                    block_id=12,
                )
                layer18.i, layer18.f, layer18.type, layer18.n = (
                    18 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                    -1,
                    "micromind.networks.PhiNetConvBlock",
                    3,
                )
                self._layers.append(layer18)
                self._save.extend(
                    x % layer18.i
                    for x in ([layer18.f] if isinstance(layer18.f, int) else layer18.f)
                    if x != -1
                )

            if number_heads >= 3:

                layer19 = Conv(feature_sizes[1], feature_sizes[1], 3, 2)
                layer19.i, layer19.f, layer19.type, layer19.n = (
                    19 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                    -1,
                    "ultralytics.nn.modules.conv.Conv",
                    1,
                )
                self._layers.append(layer19)
                self._save.extend(
                    x % layer19.i
                    for x in ([layer19.f] if isinstance(layer19.f, int) else layer19.f)
                    if x != -1
                )

                layer20 = Concat(1)
                layer20.i, layer20.f, layer20.type, layer20.n = (
                    20 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                    [
                        -1,
                        (
                            concat_layers[3]
                            + (1 if deeper_head else 0)
                            + (-1 if no_SPPF else 0)
                        ),
                    ],
                    "ultralytics.nn.modules.conv.Concat",
                    1,
                )
                self._layers.append(layer20)
                self._save.extend(
                    x % layer20.i
                    for x in ([layer20.f] if isinstance(layer20.f, int) else layer20.f)
                    if x != -1
                )

                layer21 = PhiNetConvBlock(
                    in_shape=(
                        feature_sizes[1] + feature_sizes[2],
                        20 * scale_deep,
                        20 * scale_deep,
                    ),
                    stride=1,
                    filters=feature_sizes[2],
                    expansion=0.5,
                    has_se=False,
                    block_id=21,
                )
                layer21.i, layer21.f, layer21.type, layer21.n = (
                    21 + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                    -1,
                    "micromind.networks.PhiNetConvBlock",
                    3,
                )
                self._layers.append(layer21)
                self._save.extend(
                    x % layer21.i
                    for x in ([layer21.f] if isinstance(layer21.f, int) else layer21.f)
                    if x != -1
                )

        if task == "detect":

            layer_index = 19 if number_heads == 2 else 16 if number_heads == 1 else 22

            head = Detect(nc, ch=feature_sizes)
            head.i, head.f, head.type, head.n = (
                layer_index + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                head_concat_layers,
                "ultralytics.nn.modules.conv.Detect",
                1,
            )
            self._save.extend(
                x % head.i
                for x in ([head.f] if isinstance(head.f, int) else head.f)
                if x != -1
            )

            self._layers.append(head)

        elif task == "segment":
            head = Segment(nc=nc, nm=32, npr=64, ch=feature_sizes)
            head.i, head.f, head.type, head.n = (
                layer_index + (1 if deeper_head else 0) + (-1 if no_SPPF else 0),
                head_concat_layers,
                "ultralytics.nn.modules.conv.Segment",
                1,
            )
            self._save.extend(
                x % head.i
                for x in ([head.f] if isinstance(head.f, int) else head.f)
                if x != -1
            )
            self._layers.append(head)
