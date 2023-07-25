import torch
from torch.nn.modules.batchnorm import _BatchNorm
import types
from typing import Union, Text
import torch.nn as nn


def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()

# Try considering bn too.
def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array

def _no_op(self, x):
    return x


# LogSynflow
def compute_synflow_per_weight(net, inputs, device, mode='param', remap: Union[Text, None] = 'log'):
    net = net.train()

    # Disable batch norm
    for layer in net.modules():
        if isinstance(layer, _BatchNorm):
            # TODO: this could be done with forward hooks
            layer._old_forward = layer.forward
            layer.forward = types.MethodType(_no_op, layer)

    # Convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # Convert to original values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # Keep signs of all params
    signs = linearize(net)

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)
    output = net(inputs)
    if isinstance(output, tuple):
        output = output[1]
    torch.sum(output).backward()

    # Select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            if remap:
                remap_fun = {
                    'log': lambda x: torch.log(x + 1),
                    # Other reparametrizations can be added here
                    # 'atan': torch.arctan,
                    # 'sqrt': torch.sqrt
                }
                # LogSynflow
                g = remap_fun[remap](layer.weight.grad)
            else:
                # Traditional synflow
                g = layer.weight.grad
            return torch.abs(layer.weight * g)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # Apply signs of all params
    nonlinearize(net, signs)

    # Enable batch norm again
    for layer in net.modules():
        if isinstance(layer, _BatchNorm):
            layer.forward = layer._old_forward
            del layer._old_forward

    net.float()
    return sum_arr(grads_abs)