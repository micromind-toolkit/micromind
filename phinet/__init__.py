from .networks import PhiNet
from .blocks import PhiNetConvBlock


datasets_info = {
    "CIFAR-100": {"Nclasses": 100, "NChannels": 3, "ext": ".pth.tar"},
    "CIFAR-10": {"Nclasses": 10, "NChannels": 3, "ext": ".pth.tar"},
    "ImageNet-1k": {"Nclasses": 1000, "NChannels": 3, "ext": ".pth.tar"},
}
