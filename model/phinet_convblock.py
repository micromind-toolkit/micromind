from model.model_utils import DepthwiseConv2d, SEBlock, ReLUMax, HSwish, correct_pad

import torch.nn as nn
import torch


class PhiNetConvBlock(nn.Module):
    """Implements PhiNet's convolutional block"""

    def __init__(self, in_shape, expansion, stride, filters, has_se, block_id=None, res=True, h_swish=True, k_size=3):
        """Defines the structure of the PhiNet conv block

        Args:
            in_shape ([Tuple]): [Input shape, as returned by Tensor.shape]
            expansion ([Int]): [Expansion coefficient]
            stride ([Int]): [Stride for conv block]
            filters ([Int]): [description]
            block_id ([Int]): [description]
            has_se (bool): [description]
            res (bool, optional): [description]. Defaults to True.
            h_swish (bool, optional): [description]. Defaults to True.
            k_size (int, optional): [description]. Defaults to 3.
        """
        super(PhiNetConvBlock, self).__init__()
        self.skip_conn = False

        self._layers = torch.nn.ModuleList()
        in_channels = in_shape[0]
        # Define activation function
        if h_swish:
            self.activation = HSwish()
        else:
            self.activation = ReLUMax(6)

        if block_id:
            # Expand
            self.conv1 = nn.Conv2d(
                in_channels, int(expansion * in_channels),
                kernel_size=1,
                padding="same",
                bias=False,
            )

            self.bn1 = nn.BatchNorm2d(
                int(expansion * in_channels),
                eps=1e-3,
                momentum=0.999,
            )

            self._layers.append(self.conv1)
            self._layers.append(self.bn1)
            self._layers.append(self.activation)

        if stride == 2:
            self.pad = nn.ZeroPad2d(
                padding=correct_pad(in_shape, 3),
            )

            self._layers.append(self.pad)

        d_mul = 1
        in_channels_dw = int(expansion * in_channels) if block_id else in_channels
        out_channels_dw = in_channels_dw * d_mul
        self.dw1 = DepthwiseConv2d(
            in_channels=in_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            stride=stride,
            bias=False,
            padding="same" if stride == 1 else "valid",
            # name=prefix + 'depthwise'
        )

        self.bn_dw1 = nn.BatchNorm2d(
            out_channels_dw,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(self.dw1)
        self._layers.append(self.bn_dw1)
        self._layers.append(self.activation)

        if has_se:
            num_reduced_filters = max(1, int(in_channels * 0.25))
            self.se_block = SEBlock(int(expansion * in_channels), num_reduced_filters, h_swish=h_swish)
            self._layers.append(self.se_block)

        self.conv2 = nn.Conv2d(
            in_channels=int(expansion * in_channels),
            out_channels=filters,
            kernel_size=1,
            padding="same",
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            filters,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(self.conv2)
        self._layers.append(self.bn2)

        if res and in_channels == filters and stride == 1:
            self.skip_conn = True

    def forward(self, x):
        """Executes PhiNet's convolutional block

        Args:
            x ([Tensor]): [Conv block input]

        Returns:
            [Tensor]: [Output of convolutional block]
        """
        if self.skip_conn:
            inp = x

        for l in self._layers:
            # print(l, l(x).shape)
            x = l(x)

        if self.skip_conn:
            return x + inp

        return x
