from model.model_utils import DepthwiseConv2d, correct_pad

import pytorch_lightning as pl
import torch.nn as nn
import torch


class PhiNetConvBlock(nn.Module):
    """Implements PhiNet's convolutional block"""
    def __init__(self, in_shape, expansion, stride, filters, block_id, has_se, res=True, h_swish=True, k_size=3):
        self._layers = list()

        in_channels = in_shape[1]
        # Define activation function
        if h_swish:
            activation = lambda x: x * nn.functional.ReLU6(x + 3) / 6
        else:
            activation = lambda x: torch.min(nn.functional.ReLU(x), 6)
        
        if block_id:
            # Expand
            conv1 = nn.Conv2D(
                in_channels, int(expansion * in_channels),
                kernel_size=1,
                padding="same",
                bias=False,
                )

            bn1 = nn.BatchNorm2d(
                int(expansion * in_channels),
                epsilon=1e-3,
                momentum=0.999,
                )
            
            self._layers += [conv1, bn1, activation]

        else:
            prefix = 'expanded_conv_'

        
        if stride == 2:
            pad = nn.ZeroPad2D(
                padding=correct_pad(in_shape, 3),
                )

            self._layers += [pad]
            
        d_mul = 1
        in_channels_dw = int(expansion * in_channels) if block_id else in_channels
        out_channels_dw = in_channels_dw * d_mul
        dw1 = DepthwiseConv2D(
            in_channels=in_channels_dw,
            out_channels=out_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            strides=stride,
            bias=False,
            padding="same" if stride == 1 else "valid",
            # name=prefix + 'depthwise'
            )

        bn_dw1 = nn.BatchNorm2d(
            out_channels_dw,
            epsilon=1e-3,
            momentum=0.999,
            )

        self._layers += [dw1, bn_dw1, activation]

        if has_se:
            def global_average_pool(x):
                """Performs global average pooling

                Args:
                    x ([Tensor]): [Tensor on which to perform global average pooling]

                Returns:
                    [Tensor]: [Pooled tensor]
                """
                self.temp = x

                return nn.avg_pool2d(x, x.size()[2:])

            num_reduced_filters = max(1, int(in_channels * 0.25))   

            target_shape = (in_shape[0], int(expansion * in_channels), 1, 1)
            reshape = lambda x: torch.reshape(x, target_shape)

            se_conv = nn.Conv2d(
                int(expansion * in_channels),
                int(expansion * in_channels),
                kernel_size=1,
                padding="same",
                bias=True,
            )

            sigmoid = lambda x: nn.functional.sigmoid(x)
            mul = lambda x: x * self.temp

            self._layers += [global_average_pool, reshape, se_conv, sigmoid, activation, mul]

        
        conv2 = nn.Conv2D(
            in_channels = self.temp.shape[1],
            out_channels = filters,
            kernel_size=1,
            padding="same",
            bias=False,
            )

        bn2 = nn.BatchNorm2d(
            filters,
            epsilon=1e-3,
            momentum=0.999,
            )

        self._layers += [conv2, bn2]

        if res and in_channels == filters and stride == 1:
            self.skip_conn = True

        return self._layers

    def forward(x):
        """Executes PhiNet's convolutional block

        Args:
            x ([Tensor]): [Conv block input]

        Returns:
            [Tensor]: [Output of convolutional block]
        """
        if self.skip_conn:
            inp = x
        
        for l in self._layers:
            x = l(x)

        if self.skip_conn:
            return x + inp
        
        return x