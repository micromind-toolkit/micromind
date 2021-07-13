from model.model_utils import DepthwiseConv2d, SeparableConv2d, ReLUMax, HSwish, correct_pad, get_xpansion_factor
from model.phinet_convblock import PhiNetConvBlock

import torch.nn as nn
import torch


class PhiNet(nn.Module):
    def __init__(self, res=96, in_channels=3, B0=7, alpha=0.2, beta=1.0, t_zero=6, h_swish=False, squeeze_excite=False,
                 downsampling_layers=[5, 7], conv5_percent=0, first_conv_stride=2, first_conv_filters=48, b1_filters=24,
                 b2_filters=48, include_top=True, pooling=None, classes=10, residuals=True, input_tensor=None):
        """Generates PhiNets architecture

        Args:
            res (int, optional): [base network input resolution]. Defaults to 96.
            B0 (int, optional): [base network number of blocks]. Defaults to 7.
            alpha (float, optional): [base network width multiplier]. Defaults to 0.35.
            beta (float, optional): [shape factor]. Defaults to 1.0.
            t_zero (int, optional): [initial expansion factor]. Defaults to 6.
            h_swish (bool, optional): [Approximate Hswish activation - Enable for performance, disable for compatibility (gets replaced by relu6)]. Defaults to False.
            squeeze_excite (bool, optional): [SE blocks - Enable for performance, disable for compatibility]. Defaults to False.
            downsampling_layers (list, optional): [Indices of downsampling blocks (between 5 and B0)]. Defaults to [5,7].
            conv5_percent (int, optional): [description]. Defaults to 0.
            first_conv_stride (int, optional): [Downsampling at the network input - first conv stride]. Defaults to 2.
            first_conv_filters (int, optional): [description]. Defaults to 48.
            b1_filters (int, optional): [description]. Defaults to 24.
            b2_filters (int, optional): [description]. Defaults to 48.
            include_top (bool, optional): [description]. Defaults to True.
            pooling ([type], optional): [description]. Defaults to None.
            classes (int, optional): [description]. Defaults to 10.
            residuals (bool, optional): [disable residual connections to lower ram usage - residuals]. Defaults to True.
            input_tensor ([type], optional): [description]. Defaults to None.
        """
        super(PhiNet, self).__init__()
        num_blocks = round(B0)
        input_shape = (round(res), round(res), in_channels)

        self._layers = torch.nn.ModuleList()

        # Define self.activation function
        if h_swish:
            self.activation = HSwish()
        else:
            self.activation = ReLUMax(6)

        self.sep1 = SeparableConv2d(
            in_channels,
            int(first_conv_filters * alpha),
            kernel_size=3,
            stride=(first_conv_stride, first_conv_stride),
            padding="valid",
            bias=False,
        )

        # sep_bn = nn.BatchNorm2d(
        #     int(first_conv_filters * alpha),
        #     eps=1e-3,
        #     momentum=0.999,
        # )

        self._layers.append(self.sep1)
        self._layers.append(self.activation)

        self.block1 = PhiNetConvBlock(
            in_shape=(int(first_conv_filters * alpha), res / first_conv_stride, res / first_conv_stride),
            filters=int(b1_filters * alpha),
            stride=1,
            expansion=1,
            has_se=False,
            res=residuals,
            h_swish=h_swish
        )

        self.block2 = PhiNetConvBlock(
            (int(b1_filters * alpha), res / first_conv_stride, res / first_conv_stride),
            filters=int(b1_filters * alpha),
            stride=2,
            expansion=get_xpansion_factor(t_zero, beta, 1, num_blocks),
            block_id=1,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish
        )
        
        self.block3 = PhiNetConvBlock(
            (int(b1_filters * alpha), res / first_conv_stride / 2, res / first_conv_stride / 2),
            filters=int(b1_filters * alpha),
            stride=1,
            expansion=get_xpansion_factor(t_zero, beta, 2, num_blocks),
            block_id=2,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish
        )

        self.block4 = PhiNetConvBlock(
            (int(b1_filters * alpha), res / first_conv_stride / 2, res / first_conv_stride / 2),
            filters=int(b2_filters * alpha),
            stride=2,
            expansion=get_xpansion_factor(t_zero, beta, 3, num_blocks),
            block_id=3,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish
        )

        self._layers.append(self.block1)
        self._layers.append(self.block2)
        self._layers.append(self.block3)
        self._layers.append(self.block4)

        block_id = 4
        block_filters = b2_filters
        spatial_res = res / first_conv_stride / 4
        in_channels_next = int(b2_filters * alpha)
        while num_blocks >= block_id:
            if block_id in downsampling_layers:
                block_filters *= 2

            self.pn_block = PhiNetConvBlock(
                    (in_channels_next, spatial_res, spatial_res),
                    filters=int(block_filters * alpha),
                    stride=(2 if block_id in downsampling_layers else 1),
                    expansion=get_xpansion_factor(t_zero, beta, block_id, num_blocks),
                    block_id=block_id,
                    has_se=squeeze_excite,
                    res=residuals,
                    h_swish=h_swish,
                    k_size=(5 if (block_id / num_blocks) > (1 - conv5_percent) else 3)
                )

            self._layers.append(self.pn_block)
            in_channels_next = int(block_filters * alpha)
            spatial_res = spatial_res / 2 if block_id in downsampling_layers else spatial_res
            block_id += 1
    
    def forward(self, x):
        """Executes PhiNet network

        Args:
            x ([Tensor]): [input batch]
        """
        # i = 0
        for l in self._layers:
            # print("Layer ", i, l)
            x = l(x)
            # print("Output of layer ", i, x.shape)
            # i += 1

        return x
