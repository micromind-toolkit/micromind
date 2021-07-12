import torch.nn as nn
import torch


def correct_pad(input_shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling

    Args:
        input_shape ([tuple/int]): [Input size]
        kernel_size ([tuple/int]): [Kernel size]

    Returns:
        [tuple]: [Padding coeffs]
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_shape[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_shape[0] % 2, 1 - input_shape[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return (correct[1] - adjust[1], correct[1], correct[0], correct[0] - adjust[0])

def preprocess_input(x, **kwargs):
    """Normalise channels between [-1, 1]

    Args:
        x ([Tensor]): [Contains the image, number of channels is arbitrary]

    Returns:
        [Tensor]: [Channel-wise normalised tensor]
    """

    return (x/128.)-1


def get_xpansion_factor(t_zero, beta, block_id, num_blocks):
    return (t_zero*beta)*block_id/num_blocks + t_zero*(num_blocks-block_id)/num_blocks


class DepthwiseConv2d(torch.nn.Conv2d):
    """Depthwise 2D conv

    Args:
        torch ([Tensor]): [Input tensor for convolution]
    """
    def __init__(self,
                 in_channels,
                 depth_multiplier=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros'
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
            padding_mode=padding_mode
        )