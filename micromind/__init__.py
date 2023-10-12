from .core import MicroMind, Metric, Stage

# Package version
__version__ = "0.1.1"


"""datasets_info is a dictionary that contains information about the attributes
of the datasets.
This dictionary is used in networks.py inside the from_pretrained class method
in order to examine the inputs and initialize the PhiNet or, in case of
mismatching between dataset and Nclasses, raise an AssertionError."""
datasets_info = {
    "CIFAR-100": {"Nclasses": 100, "NChannels": 3, "ext": ".pth.tar"},
    "CIFAR-10": {"Nclasses": 10, "NChannels": 3, "ext": ".pth.tar"},
    "ImageNet-1k": {"Nclasses": 1000, "NChannels": 3, "ext": ".pth.tar"},
    "MNIST": {"Nclasses": 10, "NChannels": 1, "ext": ".pth.tar"},
}
