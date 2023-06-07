import torch
import torch.nn as nn

from ultralytics.nn.modules import SPPF, C2f, Concat, Conv, Detect
from ultralytics.yolo.utils import (
    LOGGER,
)

try:
    import thop
except ImportError:
    thop = None


class Microhead(nn.Module):
    def __init__(self) -> None:
        """This class is an implementation of the head"""
        super().__init__()
        self._layers = torch.nn.ModuleList()
        self._save = []

        layer9 = SPPF(256, 256, 5)
        layer9.i, layer9.f, layer9.type, layer9.n = (
            9,
            -1,
            "ultralytics.nn.modules.block.SPPF",
            1,
        )
        self._layers.append(layer9)
        self._save.extend(
            x % layer9.i
            for x in ([layer9.f] if isinstance(layer9.f, int) else layer9.f)
            if x != -1
        )  # append to savelist

        layer10 = nn.Upsample(scale_factor=2, mode="nearest")
        layer10.i, layer10.f, layer10.type, layer10.n = (
            10,
            -1,
            "torch.nn.modules.upsampling.Upsample",
            1,
        )
        self._layers.append(layer10)
        self._save.extend(
            x % layer10.i
            for x in ([layer10.f] if isinstance(layer10.f, int) else layer10.f)
            if x != -1
        )  # append to savelist

        layer11 = Concat(1)
        layer11.i, layer11.f, layer11.type, layer11.n = (
            11,
            [-1, 6],
            "ultralytics.nn.modules.conv.Concat",
            1,
        )
        # c2 = sum(ch[x] for x in layer11.f)
        self._layers.append(layer11)
        self._save.extend(
            x % layer11.i
            for x in ([layer11.f] if isinstance(layer11.f, int) else layer11.f)
            if x != -1
        )  # append to savelist

        layer12 = C2f(384, 128, 1)
        layer12.i, layer12.f, layer12.type, layer12.n = (
            12,
            -1,
            "ultralytics.nn.modules.block.C2f",
            3,
        )
        self._layers.append(layer12)
        self._save.extend(
            x % layer12.i
            for x in ([layer12.f] if isinstance(layer12.f, int) else layer12.f)
            if x != -1
        )  # append to savelist

        layer13 = nn.Upsample(scale_factor=2, mode="nearest")
        layer13.i, layer13.f, layer13.type, layer13.n = (
            13,
            -1,
            "torch.nn.modules.upsampling.Upsample",
            1,
        )
        self._layers.append(layer13)
        self._save.extend(
            x % layer13.i
            for x in ([layer13.f] if isinstance(layer13.f, int) else layer13.f)
            if x != -1
        )  # append to savelist

        layer14 = Concat(1)
        layer14.i, layer14.f, layer14.type, layer14.n = (
            14,
            [-1, 4],
            "ultralytics.nn.modules.conv.Concat",
            1,
        )
        self._layers.append(layer14)
        self._save.extend(
            x % layer14.i
            for x in ([layer14.f] if isinstance(layer14.f, int) else layer14.f)
            if x != -1
        )  # append to savelist

        layer15 = C2f(192, 64, 1)
        layer15.i, layer15.f, layer15.type, layer15.n = (
            15,
            -1,
            "ultralytics.nn.modules.block.C2f",
            3,
        )
        self._layers.append(layer15)
        self._save.extend(
            x % layer15.i
            for x in ([layer15.f] if isinstance(layer15.f, int) else layer15.f)
            if x != -1
        )  # append to savelist

        layer16 = Conv(64, 64, 3, 2)
        layer16.i, layer16.f, layer16.type, layer16.n = (
            16,
            -1,
            "ultralytics.nn.modules.conv.Conv",
            1,
        )
        self._layers.append(layer16)
        self._save.extend(
            x % layer16.i
            for x in ([layer16.f] if isinstance(layer16.f, int) else layer16.f)
            if x != -1
        )  # append to savelist

        layer17 = Concat(1)
        layer17.i, layer17.f, layer17.type, layer17.n = (
            17,
            [-1, 12],
            "ultralytics.nn.modules.conv.Concat",
            1,
        )
        self._layers.append(layer17)
        self._save.extend(
            x % layer17.i
            for x in ([layer17.f] if isinstance(layer17.f, int) else layer17.f)
            if x != -1
        )  # append to savelist

        layer18 = C2f(192, 128, 1)
        layer18.i, layer18.f, layer18.type, layer18.n = (
            18,
            -1,
            "ultralytics.nn.modules.block.C2f",
            3,
        )
        self._layers.append(layer18)
        self._save.extend(
            x % layer18.i
            for x in ([layer18.f] if isinstance(layer18.f, int) else layer18.f)
            if x != -1
        )  # append to savelist

        layer19 = Conv(128, 128, 3, 2)
        layer19.i, layer19.f, layer19.type, layer19.n = (
            19,
            -1,
            "ultralytics.nn.modules.conv.Conv",
            1,
        )
        self._layers.append(layer19)
        self._save.extend(
            x % layer19.i
            for x in ([layer19.f] if isinstance(layer19.f, int) else layer19.f)
            if x != -1
        )  # append to savelist

        layer20 = Concat(1)
        layer20.i, layer20.f, layer20.type, layer20.n = (
            20,
            [-1, 8],
            "ultralytics.nn.modules.conv.Concat",
            1,
        )
        self._layers.append(layer20)
        self._save.extend(
            x % layer20.i
            for x in ([layer20.f] if isinstance(layer20.f, int) else layer20.f)
            if x != -1
        )  # append to savelist

        layer21 = C2f(384, 256, 1)
        layer21.i, layer21.f, layer21.type, layer21.n = (
            21,
            -1,
            "ultralytics.nn.modules.block.C2f",
            3,
        )
        self._layers.append(layer21)
        self._save.extend(
            x % layer21.i
            for x in ([layer21.f] if isinstance(layer21.f, int) else layer21.f)
            if x != -1
        )  # append to savelist

        head = Detect(80, [64, 128, 256])
        head.i, head.f, head.type, head.n = (
            22,
            [15, 18, 21],
            "ultralytics.nn.modules.conv.Detect",
            1,
        )
        self._save.extend(
            x % head.i
            for x in ([head.f] if isinstance(head.f, int) else head.f)
            if x != -1
        )  # append to savelist
        self._layers.append(head)

        for i, layer in enumerate(self._layers):
            i, f, t, n = layer.i, layer.f, layer.type, layer.n
            layer.np = sum(x.numel() for x in layer.parameters())  # number params
            n_ = max(round(n * 0.33), 1) if n > 1 else n  # depth gain
            args = []
            LOGGER.info(
                f"{i:>3}{str(f):>20}{n_:>3}{layer.np:10.0f}" f"{t:<45}{str(args):<30}"
            )  # print

        # END HARDCODED HEAD -----------------------------------------------------------
