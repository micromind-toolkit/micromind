"""
YOLOv8 building blocks.

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023

This file contains the definition of the building blocks of the yolov8 network.
Model architecture has been taken from
https://github.com/ultralytics/ultralytics/issues/189
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from micromind.utils.yolo_helpers import autopad, make_anchors, dist2bbox


class Upsample:
    def __init__(self, scale_factor, mode="nearest"):
        assert mode == "nearest"
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x: torch.Tensor):
        assert (
            len(x.shape) > 2 and len(x.shape) <= 5
        ), "Input tensor must have 3 to 5 dimensions"
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return upsampled


class Conv(nn.Module):
    """Implements YOLOv8's convolutional block"""

    def __init__(
        self, c1, c2, kernel_size=1, stride=1, padding=None, dilation=1, groups=1
    ):
        """Defines the structure of a YOLOv8 convolutional block.

        Arguments
        ---------
        c1 : int
            Input channels of the convolutional block.
        c2 : int
            Output channels of the convolutional block.
        kernel_size : int
            Kernel size for the convolutional block.
        stride : int
            Stride for the convolutional block.
        padding : int
            Padding for the convolutional block.
        dilation : int
            Dilation for the convolutional block.
        groups : int
            Groups for the convolutional block.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size, padding, dilation),
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.silu = nn.SiLU()

    def forward(self, x):
        """Executes YOLOv8 convolutional block.

        Arguments
        ---------
        x : torch.Tensor
            Input to the convolutional block.

        Returns
        -------
            Ouput of the convolutional block : torch.Tensor
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x


class Bottleneck(nn.Module):
    """Implements YOLOv8's bottleneck block"""

    def __init__(
        self,
        c1,
        c2,
        shortcut: bool,
        groups=1,
        kernels: list = (3, 3),
        channel_factor=0.5,
    ):
        """Defines the structure of a YOLOv8 bottleneck block.

        Arguments
        ---------
        c1 : int
            Input channels of the bottleneck block.
        c2 : int
            Output channels of the bottleneck block.
        shortcut : bool
            Decides whether to perform a shortcut in the bottleneck block.
        groups : int
            Groups for the bottleneck block.
        kernels : list
            Kernel size for the bottleneck block.
        channel_factor : float
            Decides the number of channels of the intermediate result
            between the two convolutional blocks.
        """
        super().__init__()
        c_ = int(c2 * channel_factor)
        self.cv1 = Conv(c1, c_, kernel_size=kernels[0], stride=1, padding=None)
        self.cv2 = Conv(
            c_, c2, kernel_size=kernels[1], stride=1, padding=None, groups=groups
        )
        self.residual = c1 == c2 and shortcut

    def forward(self, x):
        """Executes YOLOv8 bottleneck block.

        Arguments
        ---------
        x : torch.Tensor
            Input to the bottleneck block.

        Returns
        -------
            Ouput of the bottleneck block : torch.Tensor
        """
        if self.residual:
            return x + self.cv2(self.cv1(x))
        else:
            return self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Implements YOLOv8's C2f block"""

    def __init__(self, c1, c2, n=1, shortcut=False, groups=1, e=0.5):
        """Defines the structure of a YOLOv8 bottleneck block.

        Arguments
        ---------
        c1 : int
            Input channels of the C2f block.
        c2 : int
            Output channels of the C2f block.
        n : int
            Number of bottleck blocks executed in the C2f block.
        shortcut : bool
            Decides whether to perform a shortcut in the bottleneck blocks.
        groups : int
            Groups for the C2f block.
        e : float
            Factor for cancatenating intermeidate results.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(
            c1,
            2 * self.c,
            1,
        )
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.bottleneck = nn.ModuleList(
            [
                Bottleneck(
                    self.c,
                    self.c,
                    shortcut,
                    groups,
                    kernels=[(3, 3), (3, 3)],
                    channel_factor=1.0,
                )
                for _ in range(n)
            ]
        )

    def forward(self, x):
        """Executes YOLOv8 C2f block.

        Arguments
        ---------
        x : torch.Tensor
            Input to the C2f block.

        Returns
        -------
            Ouput of the C2f block : torch.Tensor
        """
        x = self.cv1(x)
        y = list(torch.chunk(x, chunks=2, dim=1))
        y.extend(m(y[-1]) for m in self.bottleneck)
        z = y[0]
        for i in y[1:]:
            z = torch.cat((z, i), dim=1)
        return self.cv2(z)


class SPPF(nn.Module):
    """Implements YOLOv8's SPPF block"""

    def __init__(self, c1, c2, k=5):
        """Defines the structure of a YOLOv8 SPPF block.

        Arguments
        ---------
        c1 : int
            Input channels of the SPPF block.
        c2 : int
            Output channels of the SPPF block.
        k : int
            Kernel size for the SPPF block Maxpooling operations
        """
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, padding=None)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, padding=None)
        self.maxpool = nn.MaxPool2d(
            kernel_size=k, stride=1, padding=2, dilation=1, ceil_mode=False
        )

    def forward(self, x):
        """Executes YOLOv8 SPPF block.

        Arguments
        ---------
        x : torch.Tensor
            Input to the SPPF block.

        Returns
        -------
            Ouput of the SPPF block : torch.Tensor
        """
        x = self.cv1(x)
        x2 = self.maxpool(x)
        x3 = self.maxpool(x2)
        x4 = self.maxpool(x3)

        y = torch.cat((x, x2, x3, x4), dim=1)
        return self.cv2(y)


class DFL(nn.Module):
    """Implements YOLOv8's DFL block"""

    def __init__(self, c1=16):
        """Defines the structure of a YOLOv8 DFL block.

        Arguments
        ---------
        c1 : int
            Input channels of the DFL block.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False)
        weight = torch.arange(c1).reshape(1, c1, 1, 1).float()
        self.conv.weight.requires_grad = False
        self.conv.weight.copy_(weight)
        self.c1 = c1

    @torch.no_grad()  # TODO: check when training
    def forward(self, x):
        """Executes YOLOv8 DFL block.

        Arguments
        ---------
        x : torch.Tensor
            Input to the DFL block.

        Returns
        -------
            Ouput of the DFL block : torch.Tensor
        """
        b, _, a = x.shape
        y = x.reshape(b, 4, self.c1, a).transpose(2, 1)
        y = F.softmax(y, dim=1)
        y = self.conv(y)
        y = y.reshape(b, 4, a)
        return y


class Darknet(nn.Module):
    """Implements YOLOv8's convolutional backbone"""

    def __init__(self, w, r, d):
        """Defines the structure of a YOLOv8 convolutional backbone.

        Arguments
        ---------
        w : float
            Width multiple of the Darknet.
        r : float
            Ratio multiple of the Darknet.
        d : float
            Depth multiple of the Darknet.
        """
        super().__init__()
        self.b1 = nn.Sequential(
            Conv(c1=3, c2=int(64 * w), kernel_size=3, stride=2, padding=1),
            Conv(int(64 * w), int(128 * w), kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            C2f(c1=int(128 * w), c2=int(128 * w), n=round(3 * d), shortcut=True),
            Conv(int(128 * w), int(256 * w), 3, 2, 1),
            C2f(int(256 * w), int(256 * w), round(6 * d), True),
        )
        self.b3 = nn.Sequential(
            Conv(int(256 * w), int(512 * w), kernel_size=3, stride=2, padding=1),
            C2f(int(512 * w), int(512 * w), round(6 * d), True),
        )
        self.b4 = nn.Sequential(
            Conv(int(512 * w), int(512 * w * r), kernel_size=3, stride=2, padding=1),
            C2f(int(512 * w * r), int(512 * w * r), round(3 * d), True),
        )

        self.b5 = SPPF(int(512 * w * r), int(512 * w * r), 5)

    def forward(self, x):
        """Executes YOLOv8 convolutional backbone.

        Arguments
        ---------
        x : torch.Tensor
            Input to the Darknet.

        Returns
        -------
            Three intermediate representations with different resolutions : tuple
        """
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        x5 = self.b5(x4)
        return (x2, x3, x5)


class Yolov8Neck(nn.Module):
    """Implements YOLOv8's neck"""

    def __init__(self, filters=[256, 512, 768], d=1):
        """Defines the structure of a YOLOv8 neck.

        Arguments
        ---------
        w : float
            Width multiple of the Darknet.
        r : float
            Ratio multiple of the Darknet.
        d : float
            Depth multiple of the Darknet.
        """
        super().__init__()
        self.up = Upsample(2, mode="nearest")
        self.n1 = C2f(
            c1=int(filters[1] + filters[2]),
            c2=int(filters[1]),
            n=round(3 * d),
            shortcut=False,
        )
        self.n2 = C2f(
            c1=int(filters[0] + filters[1]),
            c2=int(filters[0]),
            n=round(3 * d),
            shortcut=False,
        )
        self.n3 = Conv(
            c1=int(filters[0]), c2=int(filters[0]), kernel_size=3, stride=2, padding=1
        )
        self.n4 = C2f(
            c1=int(filters[0] + filters[1]),
            c2=int(filters[1]),
            n=round(3 * d),
            shortcut=False,
        )
        self.n5 = Conv(
            c1=int(filters[1]), c2=int(filters[1]), kernel_size=3, stride=2, padding=1
        )
        self.n6 = C2f(
            c1=int(filters[1] + filters[2]),
            c2=int(filters[2]),
            n=round(3 * d),
            shortcut=False,
        )

    def forward(self, p3, p4, p5):
        """Executes YOLOv8 neck.

        Arguments
        ---------
        x : tuple
            Input to the neck.

        Returns
        -------
            Three intermediate representations with different resolutions : list
        """
        x = self.up(p5)
        x = torch.cat((x, p4), dim=1)
        x = self.n1(x)
        h1 = self.up(x)
        h1 = torch.cat((h1, p3), dim=1)
        head_1 = self.n2(h1)
        h2 = self.n3(head_1)
        h2 = torch.cat((h2, x), dim=1)
        head_2 = self.n4(h2)
        h3 = self.n5(head_2)
        h3 = torch.cat((h3, p5), dim=1)
        head_3 = self.n6(h3)
        return [head_1, head_2, head_3]


class DetectionHead(nn.Module):
    """Implements YOLOv8's detection head"""

    def __init__(self, nc=80, filters=()):
        """Defines the structure of a YOLOv8 detection head.

        Arguments
        ---------
        nc : int
            Number of classes to predict.
        filters : tuple
            Number of channels of the three inputs of the detection head.
        """
        super().__init__()
        self.reg_max = 16
        self.nc = nc
        self.nl = len(filters)
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor([8.0, 16.0, 32.0], dtype=torch.float16)
        c2, c3 = max((16, filters[0] // 4, self.reg_max * 4)), max(
            filters[0], min(self.nc, 104)
        )  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in filters
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in filters
        )
        self.dfl = DFL(self.reg_max)

    def forward(self, x):
        """Executes YOLOv8 detection head.

        Arguments
        ---------
        x : list
            Input to the detection head.

        Returns
        -------
            Output of the detection head : torch.Tensor
        """
        for i in range(self.nl):
            a = self.cv2[i](x[i])
            b = self.cv3[i](x[i])
            x[i] = torch.cat((a, b), dim=1)
        self.anchors, self.strides = (
            xl.transpose(0, 1) for xl in make_anchors(x, self.stride, 0.5)
        )

        y = [(i.reshape(x[0].shape[0], self.no, -1)) for i in x]
        x_cat = torch.cat((y[0], y[1], y[2]), dim=2)
        box, cls = x_cat[:, : self.reg_max * 4], x_cat[:, self.reg_max * 4 :]
        dbox = (
            dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1)
            * self.strides
        )
        z = torch.cat((dbox, nn.Sigmoid()(cls)), dim=1)
        return z, x


class YOLOv8(nn.Module):
    """Implements YOLOv8 network"""

    def __init__(self, w, r, d, num_classes=80):
        """Defines the structure of a YOLOv8 neck.

        Arguments
        ---------
        w : float
            Width multiple of the Darknet.
        r : float
            Ratio multiple of the Darknet.
        d : float
            Depth multiple of the Darknet.
        num_classes : int
            Number of classes to predict.
        """
        super().__init__()
        self.net = Darknet(w, r, d)
        self.fpn = Yolov8Neck(w, r, d)
        self.head = DetectionHead(
            num_classes, filters=(int(256 * w), int(512 * w), int(512 * w * r))
        )

    def forward(self, x):
        """Executes YOLOv8 network.

        Arguments
        ---------
        x : torch.Tensor
            Input to the YOLOv8 network.

        Returns
        -------
            Output of the YOLOv8 network : torch.Tensor
        """
        x = self.net(x)
        x = self.fpn(*x)
        x = self.head(x)
        return x
