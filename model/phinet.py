from model.model_utils import DepthwiseConv2d, correct_pad
from model.phinet_convblock import PhiNetConvBlock

import pytorch_lightning as pl
import torch.nn as nn
import torch


class PhiNet(nn.Module):
    def __init__(
        res=96,
        in_channels=3, 
        B0=7, 
        alpha=0.35,
        beta=1.0,
        t_zero=6,
        h_swish=False,
        squeeze_excite=False, 
        downsampling_layers=[5,7], 
        conv5_percent=0,
        first_conv_stride=2, 
        first_conv_filters=48, 
        b1_filters=24, 
        b2_filters=48,
        include_top=True,
        pooling=None,
        classes=10,
        residuals=True,
        input_tensor=None,
        **kwargs):
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
        num_blocks = round(B0)
        input_shape = (round(res), round(res), in_channels)

        self._layers = list()

        # Define activation function
        if h_swish:
            activation = lambda x: x * nn.ReLU6()(x + 3) / 6
        else:
            activation = lambda x: torch.min(nn.functional.ReLU(x), 6)
        
        if block_id:
            # Expand
            conv1 = nn.Conv2d(
                in_channels, int(first_conv_filters * alpha),
                kernel_size=1,
                padding="same",
                bias=False,
                )

            bn1 = nn.BatchNorm2d(
                int(expansion * in_channels),
                eps=1e-3,
                momentum=0.999,
                )
            
            self._layers += [conv1, bn1, activation]

        pad = nn.ZeroPad2d(
            padding=correct_pad(in_shape[1:], 3),
            )

        self._layers += [pad]
            
        d_mul = 1
        in_channels_dw = int(expansion * in_channels) if block_id else in_channels
        out_channels_dw = in_channels_dw * d_mul
        dw1 = DepthwiseConv2d(
            in_channels=in_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            stride=first_conv_stride,
            bias=False,
            padding="valid",
            )

        bn_dw1 = nn.BatchNorm2d(
            out_channels_dw,
            eps=1e-3,
            momentum=0.999,
            )

        self._layers += [dw1, bn_dw1, activation]

        block1 = PhiNetConvBlock(
            in_shape=(in_channels, res/first_conv_stride, res/first_conv_stride),
            filters=int(b1_filters*alpha),
            stride=1,
            expansion=1, 
            block_id=0,
            has_se=False, 
            res=residuals, 
            h_swish=h_swish
            )

        #NETWORK BODY
        block2 = PhiNetConvBlock(
                (int(b1_filters*alpha), res/first_conv_stride, res/first_conv_stride),
                filters=int(b1_filters*alpha),
                stride=2,
                expansion=get_xpansion_factor(t_zero, beta, 1, num_blocks),
                block_id=1,
                has_se=squeeze_excite,
                res=residuals,
                h_swish=h_swish
                )
        block3 = PhiNetConvBlock(
                (int(b1_filters*alpha), res/first_conv_stride/2, res/first_conv_stride/2),
                filters=int(b1_filters*alpha),
                stride=1,
                expansion=get_xpansion_factor(t_zero, beta, 2, num_blocks),
                block_id=2,
                has_se=squeeze_excite,
                res=residuals,
                h_swish=h_swish
                )

        block4 = PhiNetConvBlock(
                (int(b1_filters*alpha), res/first_conv_stride/2, res/first_conv_stride/2),
                filters=int(b2_filters*alpha),
                stride=2,
                expansion=get_xpansion_factor(t_zero, beta, 3, num_blocks),
                block_id=3,
                has_se=squeeze_excite,
                res=residuals,
                h_swish=h_swish
                )

        self._layers += [block1, block2, block3, block4]

        block_id=4
        block_filters=b2_filters
        downsampled = 0
        while (num_blocks>=block_id):
            if block_id in downsampling_layers:
                block_filters*=2
            self._layers += [PhiNetConvBlock(
                            (int(b2_filters*alpha), res/first_conv_stride/4/(2*downsampled), res/first_conv_stride/4/(2*downsampled)),
                            filters=int(block_filters*alpha), 
                            stride=(2 if block_id in downsampling_layers else 1),
                            expansion=get_xpansion_factor(t_zero, beta, block_id, num_blocks), 
                            block_id=block_id, 
                            has_se=squeeze_excite, 
                            res=residuals, 
                            h_swish=h_swish, 
                            k_size=(5 if (block_id/num_blocks)>(1-conv5_percent) else 3))]
            block_id+=1

            if block_id in downsampling_layers:
                downsampled += 1
        

        # if include_top:
        #     x = layers.GlobalAveragePooling2D()(x)
        #     pooled_shape = (1, 1, x.shape[-1])
        #     x = layers.Reshape(pooled_shape)(x)
        #     last_block_filters = int(1280 * alpha)
        #     x = layers.Conv2D(last_block_filters,
        #                     kernel_size=1,
        #                     use_bias=True,
        #                     name='ConvFinal')(x)
        #     #x = layers.Dense(classes, activation='softmax',
        #     #                use_bias=True, name='Logits')(x)

        #     x = layers.Conv2D(classes, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
        #     x = layers.Flatten()(x)
        #     x = layers.Softmax()(x)
        # else:
        #     if pooling == 'avg':
        #         x = layers.GlobalAveragePooling2D()(x)
        #     elif pooling == 'max':
        #         x = layers.GlobalMaxPooling2D()(x)



        # # Create model.
        # model = models.Model(inputs, x, name='phinet_r%s_a%0.2f_B%s_tz%s_b%0.2f' % (res0,alpha, num_blocks,t_zero, beta))

        # return model

    def forward(self, x):
        """Executes PhiNet network

        Args:
            x ([Tensor]): [input batch]
        """
        for l in self._layers:
            print(l, l(x).shape)
            x = l(x)
            
        return x
