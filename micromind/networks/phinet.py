"""
Code for PhiNets (https://doi.org/10.1145/3510832).

Authors:
    - Francesco Paissan, 2023
    - Alberto Ancilotto, 2023
    - Matteo Beltrami, 2023
    - Matteo Tremonti, 2023
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torch.ao.nn.quantized as nnq


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    It ensures that all layers have a channel number that is divisible by divisor.

    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(input_shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_shape[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_shape[0] % 2, 1 - input_shape[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return (
        int(correct[1] - adjust[1]),
        int(correct[1]),
        int(correct[0] - adjust[0]),
        int(correct[0]),
    )


def preprocess_input(x, **kwargs):
    """Normalise channels between [-1, 1]"""

    return (x / 128.0) - 1


def get_xpansion_factor(t_zero, beta, block_id, num_blocks):
    """Compute expansion factor based on the formula from the paper"""
    return (t_zero * beta) * block_id / num_blocks + t_zero * (
        num_blocks - block_id
    ) / num_blocks


class ReLUMax(torch.nn.Module):
    def __init__(self, max):
        super(ReLUMax, self).__init__()
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=0, max=self.max)


class SEBlock(torch.nn.Module):
    """Implements squeeze-and-excitation block"""

    def __init__(self, in_channels, out_channels, h_swish=True):
        """Constructor of SEBlock

        Args:
            in_channels ([int]): [Input number of channels]
            out_channels ([int]): [Output number of channels]
            h_swish (bool, optional): [Whether to use the h_swish]. Defaults to True.
        """
        super(SEBlock, self).__init__()

        self.se_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.se_conv2 = nn.Conv2d(
            out_channels, in_channels, kernel_size=1, bias=False, padding=0
        )

        if h_swish:
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = ReLUMax(6)

        # It serves for the quantization.
        # The behavior remains equivalent for the unquantized models.
        self.mult = nnq.FloatFunctional()

    def forward(self, x):
        """Executes SE Block

        Args:
            x ([Tensor]): [input tensor]

        Returns:
            [Tensor]: [output of squeeze-and-excitation block]
        """

        inp = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.se_conv(x)
        x = self.activation(x)
        x = self.se_conv2(x)
        x = torch.sigmoid(x)

        return self.mult.mul(inp, x)  # Equivalent to ``torch.mul(a, b)``


class DepthwiseConv2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


class SeparableConv2d(torch.nn.Module):
    """Implements SeparableConv2d"""

    def __init__(
        self,
        in_channels,
        out_channels,
        activation=torch.nn.functional.relu,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        depth_multiplier=1,
    ):
        """Constructor of SeparableConv2d

        Args:
            in_channels ([int]): [Input number of channels]
            out_channels ([int]): [Output number of channels]
            kernel_size (int, optional): [Kernel size]. Defaults to 3.
            stride (int, optional): [Stride for conv]. Defaults to 1.
            padding (int, optional): [Padding for conv]. Defaults to 0.
            dilation (int, optional): []. Defaults to 1.
            bias (bool, optional): []. Defaults to True.
            padding_mode (str, optional): []. Defaults to 'zeros'.
            depth_multiplier (int, optional): [Depth multiplier]. Defaults to 1.
        """
        super().__init__()

        self._layers = torch.nn.ModuleList()

        depthwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=0,
            dilation=1,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        spatialConv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            # groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        bn = torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999)

        self._layers.append(depthwise)
        self._layers.append(spatialConv)
        self._layers.append(bn)
        self._layers.append(activation)

    def forward(self, x):
        """Executes SeparableConv2d block

        Args:
            x ([Tensor]): [Input tensor]

        Returns:
            [Tensor]: [Output of convolution]
        """
        for layer in self._layers:
            x = layer(x)

        return x


class PhiNetConvBlock(nn.Module):
    """Implements PhiNet's convolutional block"""

    def __init__(
        self,
        in_shape,
        expansion,
        stride,
        filters,
        has_se,
        block_id=None,
        res=True,
        h_swish=True,
        k_size=3,
        dp_rate=0.05,
        divisor=1,
    ):
        """Defines the structure of a PhiNet convolutional block.

        Arguments
        -------
        in_shape : tuple
            Input shape of the conv block.
        expansion : float
            Expansion coefficient for this convolutional block.
        stride: int
            Stride for the conv block.
        filters : int
            Output channels of the convolutional block.
        block_id : int
            ID of the convolutional block.
        has_se : bool
            Whether to include use Squeeze and Excite or not.
        res : bool
            Whether to use the residual connection or not.
        h_swish : bool
            Whether to use HSwish or not.
        k_size : int
            Kernel size for the depthwise convolution.

        """
        super(PhiNetConvBlock, self).__init__()

        self.param_count = 0

        self.skip_conn = False

        self._layers = torch.nn.ModuleList()
        in_channels = in_shape[0]

        # Define activation function
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        if block_id:
            # Expand
            conv1 = nn.Conv2d(
                in_channels,
                _make_divisible(int(expansion * in_channels), divisor=divisor),
                kernel_size=1,
                padding=0,
                bias=False,
            )

            bn1 = nn.BatchNorm2d(
                _make_divisible(int(expansion * in_channels), divisor=divisor),
                eps=1e-3,
                momentum=0.999,
            )

            self._layers.append(conv1)
            self._layers.append(bn1)
            self._layers.append(activation)

        if stride == 2:
            padding = correct_pad([res, res], 3)

        self._layers.append(nn.Dropout2d(dp_rate))

        d_mul = 1
        in_channels_dw = (
            _make_divisible(int(expansion * in_channels), divisor=divisor)
            if block_id
            else in_channels
        )
        out_channels_dw = in_channels_dw * d_mul
        dw1 = DepthwiseConv2d(
            in_channels=in_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            stride=stride,
            bias=False,
            padding=k_size // 2 if stride == 1 else (padding[1], padding[3]),
        )

        bn_dw1 = nn.BatchNorm2d(
            out_channels_dw,
            eps=1e-3,
            momentum=0.999,
        )

        # It is necessary to reinitialize the activation
        # for functions using Module.children() to work properly.
        # Module.children() does not return repeated layers.
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        self._layers.append(dw1)
        self._layers.append(bn_dw1)
        self._layers.append(activation)

        if has_se:
            num_reduced_filters = _make_divisible(
                max(1, int(out_channels_dw / 6)), divisor=divisor
            )
            se_block = SEBlock(out_channels_dw, num_reduced_filters, h_swish=h_swish)
            self._layers.append(se_block)

        conv2 = nn.Conv2d(
            in_channels=out_channels_dw,
            out_channels=filters,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        bn2 = nn.BatchNorm2d(
            filters,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(conv2)
        self._layers.append(bn2)

        if res and in_channels == filters and stride == 1:
            self.skip_conn = True
            # It serves for the quantization.
            # The behavior remains equivalent for the unquantized models.
            self.op = nnq.FloatFunctional()

    def forward(self, x):
        """Executes PhiNet convolutional block

        Arguments:
            x : torch.Tensor
                Input to the convolutional block.

        Returns:
            Ouput of the convolutional block : torch.Tensor
        """

        if self.skip_conn:
            inp = x

        for layer in self._layers:
            x = layer(x)

        if self.skip_conn:
            return self.op.add(x, inp)  # Equivalent to ``torch.add(a, b)``

        return x


class PhiNet(nn.Module):
    def get_complexity(self):
        """Returns MAC and number of parameters of initialized architecture.

        Returns
        -------
            Dictionary with complexity characterization of the network. : dict

        Example
        -------
        .. doctest::

            >>> from micromind.networks import PhiNet
            >>> model = PhiNet((3, 224, 224))
            >>> model.get_complexity()
            {'MAC': 9817670, 'params': 30917}
        """
        temp = summary(
            self, input_data=torch.zeros([1] + list(self.input_shape)), verbose=0
        )

        return {"MAC": temp.total_mult_adds, "params": temp.total_params}

    def get_MAC(self):
        """Returns number of MACs for this architecture.

        Returns
        -------
            Number of MAC for this network. : int

        Example
        -------
        .. doctest::

            >>> from micromind.networks import PhiNet
            >>> model = PhiNet((3, 224, 224))
            >>> model.get_MAC()
            9817670
        """
        return self.get_complexity()["MAC"]

    def get_params(self):
        """Returns number of params for this architecture.

        Returns
        -------
            Number of parameters for this network. : int

        Example
        -------
        .. doctest::

            >>> from micromind.networks import PhiNet
            >>> model = PhiNet((3, 224, 224))
            >>> model.get_params()
            30917
        """
        return self.get_complexity()["params"]

    def __init__(
        self,
        input_shape: List[int],
        num_layers: int = 7,  # num_layers
        alpha: float = 0.2,
        beta: float = 1.0,
        t_zero: float = 6,
        include_top: bool = False,
        num_classes: int = 10,
        compatibility: bool = False,
        downsampling_layers: List[int] = [5, 7],  # S2
        conv5_percent: float = 0.0,  # S2
        first_conv_stride: int = 2,  # S2
        residuals: bool = True,  # S2
        conv2d_input: bool = False,  # S2
        pool: bool = False,  # S2
        h_swish: bool = True,  # S1
        squeeze_excite: bool = True,  # S1
        divisor: int = 1,
    ) -> None:
        """This class implements the PhiNet architecture.

        Arguments
        -------
        input_shape : tuple
            Input resolution as (C, H, W).
        num_layers : int
            Number of convolutional blocks.
        alpha: float
            Width multiplier for PhiNet architecture.
        beta : float
            Shape factor of PhiNet.
        t_zero : float
            Base expansion factor for PhiNet.
        include_top : bool
            Whether to include classification head or not.
        num_classes : int
            Number of classes for the classification head.
        compatibility : bool
            `True` to maximise compatibility among embedded platforms (changes network).

        """
        super(PhiNet, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.t_zero = t_zero
        self.num_layers = num_layers
        self.num_classes = num_classes

        if compatibility:  # disables operations hard for some platforms
            h_swish = False
            squeeze_excite = False

        # this hyperparameters are hard-coded. Defined here as variables just so
        # you can play with them.
        first_conv_filters = 48
        b1_filters = 24
        b2_filters = 48

        if not isinstance(num_layers, int):
            num_layers = round(num_layers)

        assert len(input_shape) == 3, "Expected 3 elements list as input_shape."
        in_channels = input_shape[0]
        res = max(input_shape[1], input_shape[2])  # assumes squared input
        self.input_shape = input_shape

        self.classify = include_top
        self._layers = torch.nn.ModuleList()

        # Define self.activation function
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        mp = nn.MaxPool2d((2, 2))

        if not conv2d_input:
            pad = nn.ZeroPad2d(
                padding=correct_pad([res, res], 3),
            )

            self._layers.append(pad)

            sep1 = SeparableConv2d(
                in_channels,
                _make_divisible(int(first_conv_filters * alpha), divisor=divisor),
                kernel_size=3,
                stride=(first_conv_stride, first_conv_stride),
                padding=0,
                bias=False,
                activation=activation,
            )

            self._layers.append(sep1)
            # self._layers.append(activation)

            block1 = PhiNetConvBlock(
                in_shape=(
                    _make_divisible(int(first_conv_filters * alpha), divisor=divisor),
                    res / first_conv_stride,
                    res / first_conv_stride,
                ),
                filters=_make_divisible(int(b1_filters * alpha), divisor=divisor),
                stride=1,
                expansion=1,
                has_se=False,
                res=residuals,
                h_swish=h_swish,
                divisor=divisor,
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
            (
                _make_divisible(int(b1_filters * alpha), divisor=divisor),
                res / first_conv_stride,
                res / first_conv_stride,
            ),
            filters=_make_divisible(int(b1_filters * alpha), divisor=divisor),
            stride=2 if (not pool) else 1,
            expansion=get_xpansion_factor(t_zero, beta, 1, num_layers),
            block_id=1,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
            divisor=divisor,
        )

        block3 = PhiNetConvBlock(
            (
                _make_divisible(int(b1_filters * alpha), divisor=divisor),
                res / first_conv_stride / 2,
                res / first_conv_stride / 2,
            ),
            filters=_make_divisible(int(b1_filters * alpha), divisor=divisor),
            stride=1,
            expansion=get_xpansion_factor(t_zero, beta, 2, num_layers),
            block_id=2,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
            divisor=divisor,
        )

        block4 = PhiNetConvBlock(
            (
                _make_divisible(int(b1_filters * alpha), divisor=divisor),
                res / first_conv_stride / 2,
                res / first_conv_stride / 2,
            ),
            filters=_make_divisible(int(b2_filters * alpha), divisor=divisor),
            stride=2 if (not pool) else 1,
            expansion=get_xpansion_factor(t_zero, beta, 3, num_layers),
            block_id=3,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
            divisor=divisor,
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
        in_channels_next = _make_divisible(int(b2_filters * alpha), divisor=divisor)
        while num_layers >= block_id:
            if block_id in downsampling_layers:
                block_filters *= 2
                if pool:
                    self._layers.append(mp)

            pn_block = PhiNetConvBlock(
                (in_channels_next, spatial_res, spatial_res),
                filters=_make_divisible(int(block_filters * alpha), divisor=divisor),
                stride=(2 if (block_id in downsampling_layers) and (not pool) else 1),
                expansion=get_xpansion_factor(t_zero, beta, block_id, num_layers),
                block_id=block_id,
                has_se=squeeze_excite,
                res=residuals,
                h_swish=h_swish,
                k_size=(5 if (block_id / num_layers) > (1 - conv5_percent) else 3),
                divisor=divisor,
            )

            self._layers.append(pn_block)
            in_channels_next = _make_divisible(
                int(block_filters * alpha), divisor=divisor
            )
            spatial_res = (
                spatial_res / 2 if block_id in downsampling_layers else spatial_res
            )
            block_id += 1

        if include_top:
            # Includes classification head if required
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(
                    _make_divisible(int(block_filters * alpha), divisor=divisor),
                    num_classes,
                    bias=True,
                ),
            )

    def forward(self, x):
        """Executes PhiNet network

        Arguments
        -------
            x : torch.Tensor
                Network input.

        Returns
        ------
            Logits if `include_top=True`, otherwise embeddings : torch.Tensor
        """
        for layers in self._layers:
            x = layers(x)

        if self.classify:
            x = self.classifier(x)

        return x
