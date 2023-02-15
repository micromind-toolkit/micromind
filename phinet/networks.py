from .model_utils import (
    SeparableConv2d,
    ReLUMax,
    HSwish,
    correct_pad,
    get_xpansion_factor,
)
from .blocks import PhiNetConvBlock
import phinet

from pathlib import Path
from torchinfo import summary
import torch.nn as nn
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from types import SimpleNamespace
import logging


class PhiNet(nn.Module):
    @classmethod
    def from_pretrained(
        cls,
        dataset,
        alpha,
        beta,
        t_zero,
        num_layers,
        num_classes,
        resolution,
        classifier=True,
        device=None,
    ):
        """Loads parameters from checkpoint through Hugging Face Hub.

        Args:
            dataset ([dictionary/string]): [Dataset chosen]
            alpha ([tuple/float]): [alpha value]
            beta ([tuple/float]): [beta value]
            t_zero ([tuple/float]): [t_zero value]
            num_layers ([tuple/int]): [number of layers]
            num_classes ([tuple/int]): [number of classes]
            resolution ([tuple/int]): [resolution]
            classifier ([bool]): [include classifier or not]
            device ([string]): [cpu or gpu]


        Returns:
            [PhiNet]: [Model]
        """
        repo_dir = f"mbeltrami/{dataset}"
        file_to_choose = f"\
                phinet_a{float(alpha)}_b{float(beta)}_tzero{float(t_zero)}_Nlayers{num_layers}\
                _res{resolution}{phinet.datasets_info[dataset]['ext']}\
            ".replace(
            " ", ""
        )

        assert (
            num_classes == phinet.datasets_info[dataset]["Nclasses"]
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
            (phinet.datasets_info[dataset]["NChannels"], resolution, resolution),
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
        """Saves model state_dict in `save_path`."""
        torch.save(self.state_dict(), save_path)

    def from_checkpoint(self, load_path: Path):
        """Loads parameters from checkpoint."""
        self.load_state_dict(torch.load(load_path))

    def get_complexity(self):
        """Returns MAC and number of parameters of initialized architecture."""
        temp = summary(self, input_data=torch.zeros([1] + list(self.input_shape)))

        return {"MAC": temp.total_mult_adds, "params": temp.total_params}

    def get_MAC(self):
        """Returns number of MACs for this architecture."""
        return self.get_complexity()["MAC"]

    def get_params(self):
        """Returns number of params for this architecture."""
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
            self.class_conv2d = nn.Conv2d(
                int(block_filters * alpha), int(1280 * alpha), kernel_size=1, bias=True
            )
            self.final_conv = nn.Conv2d(
                int(1280 * alpha), num_classes, kernel_size=1, bias=True
            )

            self.new_convolution = nn.Conv2d(
                int(block_filters * alpha), num_classes, kernel_size=1, bias=True
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
            # x = self.final_conv(self.class_conv2d(x))
            x = self.new_convolution(x)
            x = x.view(-1, x.shape[1])

        return x
