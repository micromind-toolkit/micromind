import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import types
from functools import reduce


def sum_arr(arr):
    sum = 0.0
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()


def no_op(self, x):
    return x


def disable_batchnorm(net: nn.Module) -> nn.Module:
    """
    This function disables batch normalization layers in input `net`
    :param net: Network in which to disable normalization layers
    :return: Network in which batch normalization layers have been disabled
    """
    # Disable batch norm layers
    for layer in net.modules():
        if isinstance(layer, _BatchNorm):
            layer._old_forward = layer.forward
            layer.forward = types.MethodType(no_op, layer)

    return net


@torch.no_grad()
def linearize(net: nn.Module) -> dict:
    """
    This function stores the sign of each parameter in a given network. This is useful if one has to consider absolute
    values of weights for a given moment and then re-consider the weights themselves.
    :param net: Actual network
    :return: Alternative state_dict in which each element stores the sign of previous state_dict
    """
    # signs will be stored in signs.
    signs = {}
    for name, param in net.state_dict().items():
        signs[name] = torch.sign(param)
        param.abs_()

    return signs


@torch.no_grad()
def nonlinearize(net, signs):
    """Opposide of linearize, reconverts parameters to their original sign"""
    for name, param in net.state_dict().items():
        if "weight_mask" not in name:
            param.mul_(signs[name])


def logarithmic_synflow(layer) -> torch.tensor:
    """
    This function computes logsynflow for a given layer in an architecture. Computation of log-synflow is broken down
    to individual layers to be able to deactivate the batch normalization layers that are possibly part of a given archi-
    tecture.
    :param layer: Single layer of neural network (TODO: data type?)
    :return: tensor representing logsynflow output
    """

    # select the gradients that we want to use for search/prune
    if layer.weight.grad is not None:
        g = torch.log(
            layer.weight.grad + 1e-6
        )  # summing a small epsilon to avoid having log(0)
        return torch.abs(layer.weight * g)
    else:
        return torch.zeros_like(layer.weight)


def compute_logsynflow(
    net: nn.Module,
    inputs: torch.tensor,
    device: torch.device = torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu"),
    mode: str = "param",
) -> float:
    # set network in training mode
    net = net.to(device).train()
    # disable batch normalization
    net_nobatchnorm = disable_batchnorm(net=net)

    # convert params to their abs. Keep sign for converting it back.
    signs = linearize(net=net_nobatchnorm)

    # Compute gradients with input of 1s
    net_nobatchnorm.zero_grad()
    net_nobatchnorm.double()
    # inputs is used only to retrieve the dimensionality of input elements
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)
    # remove batchnorm
    output = net_nobatchnorm(inputs)

    if isinstance(output, tuple):
        output = output[1]
    # synflow fictional scalar loss - sum of all elements in tensor
    torch.sum(output).backward()

    # only appending the results of Convolutional or Linear layers, since they are the only
    # ones with weights
    grads_abs = [
        torch.sum(logarithmic_synflow(layer).detach())
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)
        else 0
        for layer in net_nobatchnorm.modules()
    ]

    # apply signs of all params
    nonlinearize(net, signs)

    # Enable batch norm again
    for layer in net.modules():
        if isinstance(layer, _BatchNorm):
            layer.forward = layer._old_forward
            del layer._old_forward

    net.float()

    return reduce(lambda g1, g2: g1 + g2, grads_abs).item()
