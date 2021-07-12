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

    return (correct[1] - adjust[1], correct[1], correct[0] - adjust[0], correct[0])

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

class SEBlock(torch.nn.Module):
    def __init__(self, batch_size, in_channels, out_channels, h_swish=True):
        super(SEBlock, self).__init__()
        
        self.glob_pooling = lambda x: nn.functional.avg_pool2d(x, x.size()[2:])

        self.se_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding="same",
            bias=True,
        )

        self.se_conv2 = nn.Conv2d(
            out_channels,
            in_channels,
            kernel_size=1,
            bias=False,
            padding="same"
        )

        if h_swish:
            self.activation = lambda x: x * nn.ReLU6()(x + 3) / 6
        else:
            self.activation = lambda x: torch.min(nn.functional.ReLU(x), 6)


    def forward(self, x):
        inp = x
        x = self.glob_pooling(x)
        x = self.se_conv(x)
        x = self.activation(x)
        x = self.se_conv2(x)
        x = torch.sigmoid(x)
        x = x.expand_as(inp) * inp
        
        return x