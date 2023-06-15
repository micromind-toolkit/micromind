import torch
import torch.nn as nn

from ultralytics.nn.modules import SPPF, C2f, Concat, Conv, Detect

try:
    import thop
except ImportError:
    thop = None


class Microhead(nn.Module):
    def __init__(
        self,
        nc=80,
        number_heads=3,
        feature_sizes=[16, 32, 64],
        concat_layers=[6, 4, 12],
        head_concat_layers=[15, 18, 21],
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
            concat_layers (list, required): List of the layers to be concatenated.
            head_concat_layers (list, required): List of the layers where the heads
                have to be connected.

        Returns:
            None

        """

        # some errors checks
        if number_heads not in [1, 2, 3]:
            raise ValueError("Number of heads must be between 1 and 3")
        if nc < 1:
            raise ValueError("Number of classes must be greater than 0")
        if any([x < 1 and x > 1024 for x in feature_sizes]):
            raise ValueError("Feature sizes must be greater than 0")

        # some helper variables
        number_feature_maps = len(feature_sizes)
        skipped_concat_layer = 0

        # if we reached this point we are good to go!
        super().__init__()
        self._layers = torch.nn.ModuleList()
        self._save = []

        layer9 = SPPF(feature_sizes[2], feature_sizes[2], 5)  # idk for the 5
        layer9.i, layer9.f, layer9.type, layer9.n = (
            9,
            -1,
            "ultralytics.nn.modules.block.SPPF",
            1,
        )
        self._layers.append(layer9)
        self._save.extend(
            x % layer9.i
            for x in ([layer9.f] if isinstance(layer9.f, int) else layer9.f)
            if x != -1
        )  # append to savelist

        layer10 = nn.Upsample(scale_factor=2, mode="nearest")
        layer10.i, layer10.f, layer10.type, layer10.n = (
            10,
            -1,
            "torch.nn.modules.upsampling.Upsample",
            1,
        )
        self._layers.append(layer10)
        self._save.extend(
            x % layer10.i
            for x in ([layer10.f] if isinstance(layer10.f, int) else layer10.f)
            if x != -1
        )  # append to savelist

        if number_feature_maps == 3:
            layer11 = Concat(dimension=1)
            layer11.i, layer11.f, layer11.type, layer11.n = (
                11,
                # previous layer concatenated to the first of the list
                [-1, concat_layers[0]],
                "ultralytics.nn.modules.conv.Concat",
                1,
            )
            self._layers.append(layer11)
            self._save.extend(
                x % layer11.i
                for x in ([layer11.f] if isinstance(layer11.f, int) else layer11.f)
                if x != -1
            )  # append to savelist
        else:
            skipped_concat_layer += 1

        layer12 = C2f(feature_sizes[1] + feature_sizes[2], feature_sizes[1], 1)
        layer12.i, layer12.f, layer12.type, layer12.n = (
            12,
            -1,
            "ultralytics.nn.modules.block.C2f",
            3,
        )
        self._layers.append(layer12)
        self._save.extend(
            x % layer12.i
            for x in ([layer12.f] if isinstance(layer12.f, int) else layer12.f)
            if x != -1
        )  # append to savelist

        layer13 = nn.Upsample(scale_factor=2, mode="nearest")
        layer13.i, layer13.f, layer13.type, layer13.n = (
            13,
            -1,
            "torch.nn.modules.upsampling.Upsample",
            1,
        )
        self._layers.append(layer13)
        self._save.extend(
            x % layer13.i
            for x in ([layer13.f] if isinstance(layer13.f, int) else layer13.f)
            if x != -1
        )  # append to savelist

        if number_feature_maps >= 2:
            layer14 = Concat(1)
            layer14.i, layer14.f, layer14.type, layer14.n = (
                14,
                # previous layer concatenated to the second of the list
                [-1, concat_layers[1]],
                "ultralytics.nn.modules.conv.Concat",
                1,
            )
            self._layers.append(layer14)
            self._save.extend(
                x % layer14.i
                for x in ([layer14.f] if isinstance(layer14.f, int) else layer14.f)
                if x != -1
            )  # append to savelist
        else:
            skipped_concat_layer += 1

        layer15 = C2f(feature_sizes[0] + feature_sizes[1], feature_sizes[0], 1)
        layer15.i, layer15.f, layer15.type, layer15.n = (
            15,
            -1,
            "ultralytics.nn.modules.block.C2f",
            3,
        )
        self._layers.append(layer15)
        self._save.extend(
            x % layer15.i
            for x in ([layer15.f] if isinstance(layer15.f, int) else layer15.f)
            if x != -1
        )  # append to savelist

        layer16 = Conv(feature_sizes[0], feature_sizes[0], 3, 2)
        layer16.i, layer16.f, layer16.type, layer16.n = (
            16,
            -1,
            "ultralytics.nn.modules.conv.Conv",
            1,
        )
        self._layers.append(layer16)
        self._save.extend(
            x % layer16.i
            for x in ([layer16.f] if isinstance(layer16.f, int) else layer16.f)
            if x != -1
        )  # append to savelist

        layer17 = Concat(1)
        layer17.i, layer17.f, layer17.type, layer17.n = (
            17,
            # previous layer concatenated to the second of the list minus the skipped
            # layer
            [
                -1,
                concat_layers[2] - skipped_concat_layer,
            ],  # the skipped connection is added because
            # there might be some skipped layer and the nn has to take that into account
            "ultralytics.nn.modules.conv.Concat",
            1,
        )
        self._layers.append(layer17)
        self._save.extend(
            x % layer17.i
            for x in ([layer17.f] if isinstance(layer17.f, int) else layer17.f)
            if x != -1
        )  # append to savelist

        layer18 = C2f(feature_sizes[0] + feature_sizes[1], feature_sizes[1], 1)
        layer18.i, layer18.f, layer18.type, layer18.n = (
            18,
            -1,
            "ultralytics.nn.modules.block.C2f",
            3,
        )
        self._layers.append(layer18)
        self._save.extend(
            x % layer18.i
            for x in ([layer18.f] if isinstance(layer18.f, int) else layer18.f)
            if x != -1
        )  # append to savelist

        layer19 = Conv(feature_sizes[1], feature_sizes[1], 3, 2)
        layer19.i, layer19.f, layer19.type, layer19.n = (
            19,
            -1,
            "ultralytics.nn.modules.conv.Conv",
            1,
        )
        self._layers.append(layer19)
        self._save.extend(
            x % layer19.i
            for x in ([layer19.f] if isinstance(layer19.f, int) else layer19.f)
            if x != -1
        )  # append to savelist

        layer20 = Concat(1)
        layer20.i, layer20.f, layer20.type, layer20.n = (
            20,
            [-1, 9],
            "ultralytics.nn.modules.conv.Concat",
            1,
        )
        self._layers.append(layer20)
        self._save.extend(
            x % layer20.i
            for x in ([layer20.f] if isinstance(layer20.f, int) else layer20.f)
            if x != -1
        )  # append to savelist

        layer21 = C2f(feature_sizes[1] + feature_sizes[2], feature_sizes[2], 1)
        layer21.i, layer21.f, layer21.type, layer21.n = (
            21,
            -1,
            "ultralytics.nn.modules.block.C2f",
            3,
        )
        self._layers.append(layer21)
        self._save.extend(
            x % layer21.i
            for x in ([layer21.f] if isinstance(layer21.f, int) else layer21.f)
            if x != -1
        )  # append to savelist

        # based on the number of detections scales create the detections layers
        base_connections = [x - skipped_concat_layer for x in head_concat_layers]

        # the skipped connection is added because there might be some skipped
        # layers and the head also has to take that into account

        head_connections = get_connections_based_on_number_of_heads_arg(
            base_connections, number_heads
        )
        head = Detect(nc, [16, 32, 64])
        head.i, head.f, head.type, head.n = (
            22,
            head_connections,
            "ultralytics.nn.modules.conv.Detect",
            1,
        )
        self._save.extend(
            x % head.i
            for x in ([head.f] if isinstance(head.f, int) else head.f)
            if x != -1
        )  # append to savelist
        self._layers.append(head)

        # END HARDCODED HEAD ---------------------------------------------


def get_connections_based_on_number_of_heads_arg(head_connections, number_of_heads):
    start = 3 - number_of_heads
    return head_connections[start:]
