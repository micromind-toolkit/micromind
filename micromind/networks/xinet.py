"""
Code for XiNet (https://shorturl.at/mtHT0)

Authors:
    - Francesco Paissan, 2023
    - Alberto Ancilotto, 2023
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple, Optional, List


def autopad(k: int, p: Optional[int] = None):
    """Implements padding to mimic "same" behaviour.
    Arguments
    ---------
    k : int
        Kernel size for the convolution.
    p : Optional[int]
        Padding value to be applied.

    """
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class XiConv(nn.Module):
    """Implements XiNet's convolutional block as presented in the original paper.

    Arguments
    ---------
    c_in: int
        Number of input channels.
    c_out: int
        Number of output channels.
    kernel_size: Union[int, Tuple] = 3
        Kernel size for the main convolution.
    stride: Union[int, Tuple] = 1
        Stride for the main convolution.
    padding: Optional[Union[int, Tuple]] = None
        Padding that is applied in the main convolution.
    groups: Optional[int] = 1
        Number of groups for the main convolution.
    act: Optional[bool] = True
        When True, uses SiLU activation function after
        the main convolution.
    gamma: Optional[float] = 4
        Compression factor for the convolutional block.
    attention: Optional[bool] = True
        When True, uses attention.
    skip_tensor_in: Optional[bool] =True
        When True, defines broadcasting skip connection block.
    skip_channels: Optional[int] = 1
        Number of channels for the input block.
    pool: Optional[bool] = None
        When True, applies pooling after the main convolution.
    attention_k: Optional[int] = 3
        Kernel for the attention module.
    attention_lite: Optional[bool] = True
        When True, uses efficient attention implementation.
    batchnorm: Optional[bool] = True
        When True, uses batch normalization inside the ConvBlock.
    dropout_rate: Optional[int] = 0
        Dropout probability.
    skip_k: Optional[int] = 1
        Kernel for the broadcast skip connection.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: Union[int, Tuple] = 3,
        stride: Union[int, Tuple] = 1,
        padding: Optional[Union[int, Tuple]] = None,
        groups: Optional[int] = 1,
        act: Optional[bool] = True,
        gamma: Optional[float] = 4,
        attention: Optional[bool] = True,
        skip_tensor_in: Optional[bool] = True,
        skip_channels: Optional[int] = 1,
        pool: Optional[bool] = None,
        attention_k: Optional[int] = 3,
        attention_lite: Optional[bool] = True,
        batchnorm: Optional[bool] = True,
        dropout_rate: Optional[int] = 0,
        skip_k: Optional[int] = 1,
    ):
        super().__init__()
        self.compression = int(gamma)
        self.attention = attention
        self.attention_lite = attention_lite
        self.attention_lite_ch_in = c_out // self.compression // 2
        self.pool = pool
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate

        if self.compression > 1:
            self.compression_conv = nn.Conv2d(
                c_in, c_out // self.compression, 1, 1, groups=groups, bias=False
            )
        self.main_conv = nn.Conv2d(
            c_out // self.compression if self.compression > 1 else c_in,
            c_out,
            kernel_size,
            stride,
            groups=groups,
            padding=autopad(kernel_size, padding),
            bias=False,
        )
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

        if attention:
            if attention_lite:
                self.att_pw_conv = nn.Conv2d(
                    c_out, self.attention_lite_ch_in, 1, 1, groups=groups, bias=False
                )
            self.att_conv = nn.Conv2d(
                c_out if not attention_lite else self.attention_lite_ch_in,
                c_out,
                attention_k,
                1,
                groups=groups,
                padding=autopad(attention_k, None),
                bias=False,
            )
            self.att_act = nn.Sigmoid()

        if pool:
            self.mp = nn.MaxPool2d(pool)
        if skip_tensor_in:
            self.skip_conv = nn.Conv2d(
                skip_channels,
                c_out // self.compression,
                skip_k,
                1,
                groups=groups,
                padding=autopad(skip_k, None),
                bias=False,
            )
        if batchnorm:
            self.bn = nn.BatchNorm2d(c_out)
        if dropout_rate > 0:
            self.do = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """Computes the forward step of the XiNet's convolutional block.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        ConvBlock output. : torch.Tensor
        """
        s = None
        # skip connection
        if isinstance(x, list):
            s = F.adaptive_avg_pool2d(x[1], output_size=x[0].shape[2:])
            s = self.skip_conv(s)
            x = x[0]

        # compression convolution
        if self.compression > 1:
            x = self.compression_conv(x)

        if s is not None:
            x = x + s

        if self.pool:
            x = self.mp(x)

        # main conv and activation
        x = self.main_conv(x)
        if self.batchnorm:
            x = self.bn(x)
        x = self.act(x)

        # attention conv
        if self.attention:
            if self.attention_lite:
                att_in = self.att_pw_conv(x)
            else:
                att_in = x
            y = self.att_act(self.att_conv(att_in))
            x = x * y

        if self.dropout_rate > 0:
            x = self.do(x)

        return x


class XiNet(nn.Module):
    """Defines a XiNet.

    Arguments
    ---------
    alpha: float
        Width multiplier.
    gamma : float
        Compression factor.
    num_layers : int = 5
        Number of convolutional blocks.
    in_channels : int = 3
        Number of input channels
    num_classes : int
        Number of classes. It is used only when include_top is True.
    include_top : Optional[bool]
        When True, defines an MLP for classification.
    base_filters : int
        Number of base filters for the ConvBlock.
    return_layers : Optional[List]
        Ids of the layers to be returned after processing the foward
        step.

    Example
    -------
    .. doctest::
        >>> from micromind.networks import XiNet
        >>> model = XiNet()
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 4.0,
        num_layers: int = 5,
        in_channels: int = 3,
        num_classes=1000,
        include_top=False,
        base_filters: int = 16,
        return_layers: Optional[List] = None,
    ):
        super().__init__()

        self._layers = nn.ModuleList([])
        self.include_top = include_top
        self.return_layers = return_layers

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                int(base_filters * alpha),
                7,
                padding=7 // 2,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(int(base_filters * alpha)),
            nn.SiLU(),
        )

        num_filters = [
            int(2 ** (base_filters**0.5 + i)) for i in range(0, num_layers)
        ]
        skip_channels_num = int(base_filters * 2 * alpha)

        for i in range(
            len(num_filters) - 2
        ):  # Account for the last two layers separately
            self._layers.append(
                XiConv(
                    int(num_filters[i] * alpha),
                    int(num_filters[i + 1] * alpha),
                    kernel_size=3,
                    stride=1,
                    pool=2,
                    skip_tensor_in=(i != 0),
                    skip_channels=skip_channels_num,
                    gamma=gamma,
                )
            )
            self._layers.append(
                XiConv(
                    int(num_filters[i + 1] * alpha),
                    int(num_filters[i + 1] * alpha),
                    kernel_size=3,
                    stride=1,
                    skip_tensor_in=True,
                    skip_channels=skip_channels_num,
                    gamma=gamma,
                )
            )

        # Adding the last two layers with attention=False
        self._layers.append(
            XiConv(
                int(num_filters[-2] * alpha),
                int(num_filters[-1] * alpha),
                kernel_size=2,
                stride=1,
                skip_tensor_in=True,
                skip_channels=skip_channels_num,
                attention=False,
            )
        )
        self._layers.append(
            XiConv(
                int(num_filters[-1] * alpha),
                int(num_filters[-1] * alpha),
                kernel_size=2,
                stride=1,
                skip_tensor_in=True,
                skip_channels=skip_channels_num,
                attention=False,
            )
        )

        if self.return_layers is not None:
            print(f"XiNet configured to return layers {self.return_layers}:")
            for i in self.return_layers:
                print(f"Layer {i} - {self._layers[i].__class__}")

        if self.include_top:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(int(num_filters[-1] * alpha), num_classes),
            )

    def forward(self, x):
        """Computes the forward step of the XiNet.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Output of the network, as defined from
            the init. : Union[torch.Tensor, Tuple]
        """
        x = self.conv1(x)
        skip = None
        ret = []
        for layer_id, layer in enumerate(self._layers):
            if layer_id == 0:
                x = layer(x)
                skip = x
            else:
                x = layer([x, skip])

            if layer_id in self.return_layers:
                ret.append(x)

        if self.include_top:
            x = self.classifier(x)

        if self.return_layers is not None:
            return x, ret

        return x
