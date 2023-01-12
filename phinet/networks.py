
from .model_utils import (
    SeparableConv2d,
    ReLUMax,
    HSwish,
    correct_pad,
    get_xpansion_factor,
)
from .blocks import PhiNetConvBlock

import torch.nn as nn
import torch


class PhiNet(nn.Module):
    def __init__(
        self,
        res=96, # this goes as input_shape
        in_channels=3,  # this goes as input_shape
        B0=7,   # num_layers
        alpha=0.2,
        beta=1.0,
        t_zero=6,
        h_swish=False,  # S1
        squeeze_excite=False,   # S1
        downsampling_layers=[5, 7], # S2
        conv5_percent=0,    # S2
        first_conv_stride=2,    # S2
        first_conv_filters=48,  # hard-coded
        b1_filters=24,  # hard-coded
        b2_filters=48,  # hard-coded
        include_top=False,
        pooling=None,   # remove - not used
        num_classes=10, 
        residuals=True, # S2
        input_tensor=None,  # remove
        conv2d_input=False, # S2
        pool=False, # S2
    ):
        """Generates PhiNets architecture

        Args:
            res (int, optional): [base network input resolution]. Defaults to 96.
            B0 (int, optional): [base network number of blocks]. Defaults to 7.
            alpha (float, optional): [base network width multiplier]. Defaults to 0.35.
            beta (float, optional): [shape factor]. Defaults to 1.0.
            t_zero (int, optional): [initial expansion factor]. Defaults to 6.
            h_swish (bool, optional): [True to use h_swish]defaults to False.
            squeeze_excite (bool, optional): [True to use SE blocks]. Defaults to False.
            downsampling_layers (list, optional): []. Defaults to [5,7].
            conv5_percent (int, optional): [description]. Defaults to 0.
            first_conv_stride (int, optional): [Downsampling at the network input \
                    - first conv stride]. Defaults to 2.
            first_conv_filters (int, optional): [description]. Defaults to 48.
            b1_filters (int, optional): [description]. Defaults to 24.
            b2_filters (int, optional): [description]. Defaults to 48.
            include_top (bool, optional): [description]. Defaults to True.
            pooling ([type], optional): [description]. Defaults to None.
            classes (int, optional): [description]. Defaults to 10.
            residuals (bool, optional): [disable residual connections to lower ram \
                    usage - residuals]. Defaults to True.
            input_tensor ([type], optional): [description]. Defaults to None.
        """
        super(PhiNet, self).__init__()
        self.classify = include_top

        num_blocks = round(B0)
        input_shape = (round(res), round(res), in_channels)

        self._layers = torch.nn.ModuleList()

        # Define self.activation function
        if h_swish:
            activation = HSwish()
        else:
            activation = ReLUMax(6)

        mp = nn.MaxPool2d((2, 2))

        if not conv2d_input:
            pad = nn.ZeroPad2d(
                padding=correct_pad(input_shape, 3),
            )

            self._layers.append(pad)

            sep1 = SeparableConv2d(
                in_channels,
                int(first_conv_filters * alpha),
                kernel_size=3,
                stride=(first_conv_stride, first_conv_stride),
                padding="valid",
                bias=False,
                activation=activation,
            )

            self._layers.append(sep1)
            # self._layers.append(activation)

            block1 = PhiNetConvBlock(
                in_shape=(
                    int(first_conv_filters * alpha),
                    res / first_conv_stride,
                    res / first_conv_stride,
                ),
                filters=int(b1_filters * alpha),
                stride=1,
                expansion=1,
                has_se=False,
                res=residuals,
                h_swish=h_swish,
            )

            self._layers.append(block1)
        else:

            c1 = nn.Conv2d(
                in_channels, int(b1_filters * alpha), kernel_size=(3, 3), bias=False
            )

            bn_c1 = nn.BatchNorm2d(int(b1_filters * alpha))

            self._layers.append(c1)
            self._layers.append(activation)
            self._layers.append(bn_c1)

        block2 = PhiNetConvBlock(
            (int(b1_filters * alpha), res / first_conv_stride, res / first_conv_stride),
            filters=int(b1_filters * alpha),
            stride=2 if (not pool) else 1,
            expansion=get_xpansion_factor(t_zero, beta, 1, num_blocks),
            block_id=1,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
        )

        block3 = PhiNetConvBlock(
            (
                int(b1_filters * alpha),
                res / first_conv_stride / 2,
                res / first_conv_stride / 2,
            ),
            filters=int(b1_filters * alpha),
            stride=1,
            expansion=get_xpansion_factor(t_zero, beta, 2, num_blocks),
            block_id=2,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
        )

        block4 = PhiNetConvBlock(
            (
                int(b1_filters * alpha),
                res / first_conv_stride / 2,
                res / first_conv_stride / 2,
            ),
            filters=int(b2_filters * alpha),
            stride=2 if (not pool) else 1,
            expansion=get_xpansion_factor(t_zero, beta, 3, num_blocks),
            block_id=3,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
        )

        self._layers.append(block2)
        if pool:
            self._layers.append(mp)
        self._layers.append(block3)
        self._layers.append(block4)
        if pool:
            self._layers.append(mp)

        block_id = 4
        block_filters = b2_filters
        spatial_res = res / first_conv_stride / 4
        in_channels_next = int(b2_filters * alpha)
        while num_blocks >= block_id:
            if block_id in downsampling_layers:
                block_filters *= 2
                if pool:
                    self._layers.append(mp)

            pn_block = PhiNetConvBlock(
                (in_channels_next, spatial_res, spatial_res),
                filters=int(block_filters * alpha),
                stride=(2 if (block_id in downsampling_layers) and (not pool) else 1),
                expansion=get_xpansion_factor(t_zero, beta, block_id, num_blocks),
                block_id=block_id,
                has_se=squeeze_excite,
                res=residuals,
                h_swish=h_swish,
                k_size=(5 if (block_id / num_blocks) > (1 - conv5_percent) else 3),
            )

            self._layers.append(pn_block)
            in_channels_next = int(block_filters * alpha)
            spatial_res = (
                spatial_res / 2 if block_id in downsampling_layers else spatial_res
            )
            block_id += 1

        if include_top:
            # Includes classification head if required
            self.glob_pooling = lambda x: nn.functional.avg_pool2d(x, x.size()[2:])
            self.class_conv2d = nn.Conv2d(
                int(block_filters * alpha), int(1280 * alpha), kernel_size=1, bias=True
            )
            self.final_conv = nn.Conv2d(
                int(1280 * alpha), num_classes, kernel_size=1, bias=True
            )

            # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # self.classifier = nn.Linear(int(block_filters * alpha), num_classes)
            # self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        """Executes PhiNet network

        Args:
            x ([Tensor]): [input batch]
        """
        for layers in self._layers:
            x = layers(x)

        if self.classify:
            x = self.glob_pooling(x)
            x = self.final_conv(self.class_conv2d(x))
            x = x.view(-1, x.shape[1])

        return x
