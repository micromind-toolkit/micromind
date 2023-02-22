"""
Code for PhiNets (https://doi.org/10.1145/3510832).

Authors:
    - Francesco Paissan, 2023
    - Alberto Ancilotto, 2023
    - Matteo Beltrami, 2023
"""
from .model_utils import (
    SeparableConv2d,
    ReLUMax,
    HSwish,
    correct_pad,
    get_xpansion_factor,
)
import micromind

from pathlib import Path
from torchinfo import summary
import torch.nn as nn
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from types import SimpleNamespace
import logging

from .model_utils import DepthwiseConv2d, SEBlock, ReLUMax, HSwish, correct_pad

import torch.nn as nn
import torch


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
            activation = HSwish()
        else:
            activation = ReLUMax(6)

        if block_id:
            # Expand
            conv1 = nn.Conv2d(
                in_channels,
                int(expansion * in_channels),
                kernel_size=1,
                padding=0,
                bias=False,
            )

            bn1 = nn.BatchNorm2d(
                int(expansion * in_channels),
                eps=1e-3,
                momentum=0.999,
            )

            self._layers.append(conv1)
            self._layers.append(bn1)
            self._layers.append(activation)

        if stride == 2:
            pad = nn.ZeroPad2d(
                padding=correct_pad(in_shape, 3),
            )

            self._layers.append(pad)

        self._layers.append(nn.Dropout2d(dp_rate))

        d_mul = 1
        in_channels_dw = int(expansion * in_channels) if block_id else in_channels
        out_channels_dw = in_channels_dw * d_mul
        dw1 = DepthwiseConv2d(
            in_channels=in_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            stride=stride,
            bias=False,
            padding=k_size // 2 if stride == 1 else 0,
        )

        bn_dw1 = nn.BatchNorm2d(
            out_channels_dw,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(dw1)
        self._layers.append(bn_dw1)
        self._layers.append(activation)

        if has_se:
            num_reduced_filters = max(1, int(expansion * in_channels / 6))
            se_block = SEBlock(
                int(expansion * in_channels), num_reduced_filters, h_swish=h_swish
            )
            self._layers.append(se_block)

        conv2 = nn.Conv2d(
            in_channels=int(expansion * in_channels),
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
            return x + inp

        return x


class PhiNet(nn.Module):
    @classmethod
    def from_pretrained(
        cls,
        dataset,
        alpha,
        beta,
        t_zero,
        num_layers,
        resolution,
        num_classes=None,
        classifier=True,
        device=None,
    ):
        """Loads parameters from checkpoint through Hugging Face Hub.
        This function constructs two strings, "repo_dir" to find the model on Hugging
        Face Hub and "file_to_choose" to select the correct file inside the repo, and
        use them to download the pretrained model and initialize the PhiNet.

        Arguments
        ---------
        dataset : string
            The dataset on which the model has been trained with.
        alpha : float
            The alpha hyperparameter.
        beta : float
            The beta hyperparameter.
        t_zero : float
            The t_zero hyperparameter.
        num_layers : int
            The number of layers.
        resolution : int
            The resolution of the images used during training.
        num_classes : int
            The number of classes that the model has been trained for.
            If None, it gets the specific value determined by the dataset used.
        classifier : bool
            If True, the model returend includes the classifier.
        device : string
            The device that loads all the tensors.
            If None, it's set to "cuda" if it's available, it's set to "cpu" otherwise.

        Returns
        -------
            PhiNet: nn.Module

        Example
        -------
        >>> from micromind import PhiNet
        >>> model = PhiNet.from_pretrained("CIFAR-10", 3.0, 0.75, 6.0, 7, 160)
        """
        if num_classes is None:
            num_classes = micromind.datasets_info[dataset]["Nclasses"]

        repo_dir = f"micromind/{dataset}"
        file_to_choose = f"\
                phinet_a{float(alpha)}_b{float(beta)}_tzero{float(t_zero)}_Nlayers{num_layers}\
                _res{resolution}{micromind.datasets_info[dataset]['ext']}\
            ".replace(
            " ", ""
        )

        assert (
            num_classes == micromind.datasets_info[dataset]["Nclasses"]
        ), "Can't load model because num_classes does not match with dataset."

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        try:
            downloaded_file_path = hf_hub_download(
                repo_id=repo_dir, filename=file_to_choose
            )
            state_dict = torch.load(str(downloaded_file_path), map_location=device)
            model_found = True

        except EntryNotFoundError:
            state_dict = {
                "args": SimpleNamespace(
                    alpha=alpha,
                    beta=beta,
                    t_zero=t_zero,
                    num_layers=num_layers,
                    num_classes=num_classes,
                )
            }
            model_found = False
            logging.warning("Model initialized without loading checkpoint.")

        # model initialized
        model = cls(
            (micromind.datasets_info[dataset]["NChannels"], resolution, resolution),
            alpha=state_dict["args"].alpha,
            beta=state_dict["args"].beta,
            t_zero=state_dict["args"].t_zero,
            num_layers=state_dict["args"].num_layers,
            num_classes=state_dict["args"].num_classes,
            include_top=classifier,
            compatibility=False,
        )

        # model initialized with downloaded parameters
        if model_found:
            model.load_state_dict(state_dict["state_dict"], strict=False)
            print("Checkpoint loaded successfully.")

        return model

    def save_params(self, save_path: Path):
        """Saves state_dict of model into a given path.

        Arguments
        ---------
        save_path : string or Path
            Path where you want to store the state dict.

        Returns
        -------
            None

        Example
        -------
        >>> from micromind import PhiNet
        >>> model = PhiNet((3, 224, 224))
        >>> model.save_params("checkpoint.pt")
        """
        torch.save(self.state_dict(), save_path)

    def from_checkpoint(self, load_path: Path):
        """Loads state_dict of model into current instance of the PhiNet class.

        Arguments
        ---------
        load_path : string or Path
            Path where you want to store the state dict.

        Returns
        -------
            None

        Example
        -------
        >>> from micromind import PhiNet
        >>> model = PhiNet((3, 224, 224))
        >>> model.from_checkpoint("checkpoint.pt")
        """
        self.load_state_dict(torch.load(load_path))

    def get_complexity(self):
        """Returns MAC and number of parameters of initialized architecture.

        Returns
        -------
            Dictionary containing MAC count and number of parameters for the network. : dict

        Example
        -------
        >>> from micromind import PhiNet
        >>> model = PhiNet((3, 224, 224))
        >>> model.get_complexity()
        """
        temp = summary(self, input_data=torch.zeros([1] + list(self.input_shape)))

        return {"MAC": temp.total_mult_adds, "params": temp.total_params}

    def get_MAC(self):
        """Returns number of MACs for this architecture.

        Returns
        -------
            Number of MAC for this network. : int

        Example
        -------
        >>> from micromind import PhiNet
        >>> model = PhiNet((3, 224, 224))
        >>> model.get_MAC()
        """
        return self.get_complexity()["MAC"]

    def get_params(self):
        """Returns number of params for this architecture.

        Returns
        -------
            Number of parameters for this network. : int

        Example
        -------
        >>> from micromind import PhiNet
        >>> model = PhiNet((3, 224, 224))
        >>> model.get_params()
        """
        return self.get_complexity()["params"]

    def __init__(
        self,
        input_shape: list[int],
        num_layers: int = 7,  # num_layers
        alpha: float = 0.2,
        beta: float = 1.0,
        t_zero: float = 6,
        include_top: bool = False,
        num_classes: int = 10,
        compatibility: bool = False,
        downsampling_layers: list[int] = [5, 7],  # S2
        conv5_percent: float = 0.0,  # S2
        first_conv_stride: int = 2,  # S2
        residuals: bool = True,  # S2
        conv2d_input: bool = False,  # S2
        pool: bool = False,  # S2
        h_swish: bool = True,  # S1
        squeeze_excite: bool = True,  # S1
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
            True to maximise compatibility among embedded platforms. Compromises performance a bit.

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
        res = max(input_shape[0], input_shape[1])
        self.input_shape = input_shape

        self.classify = include_top
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
                padding=0,
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
            expansion=get_xpansion_factor(t_zero, beta, 1, num_layers),
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
            expansion=get_xpansion_factor(t_zero, beta, 2, num_layers),
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
            expansion=get_xpansion_factor(t_zero, beta, 3, num_layers),
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
        while num_layers >= block_id:
            if block_id in downsampling_layers:
                block_filters *= 2
                if pool:
                    self._layers.append(mp)

            pn_block = PhiNetConvBlock(
                (in_channels_next, spatial_res, spatial_res),
                filters=int(block_filters * alpha),
                stride=(2 if (block_id in downsampling_layers) and (not pool) else 1),
                expansion=get_xpansion_factor(t_zero, beta, block_id, num_layers),
                block_id=block_id,
                has_se=squeeze_excite,
                res=residuals,
                h_swish=h_swish,
                k_size=(5 if (block_id / num_layers) > (1 - conv5_percent) else 3),
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

            self.new_convolution = nn.Conv2d(
                int(block_filters * alpha), num_classes, kernel_size=1, bias=True
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
            x = self.glob_pooling(x)
            x = self.new_convolution(x)
            x = x.view(-1, x.shape[1])

        return x
