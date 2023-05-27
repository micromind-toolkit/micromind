import numpy as np
import torch


def score_network(model, batch_size, loader_train):
    network = model
    batch_size = batch_size
    loader_train = loader_train

    def counting_forward_hook(module, inp, out):
        inp = inp[0].view(inp[0].size(0), -1)
        x = (inp > 0).float()  # binary indicator
        K = x @ x.t()
        K2 = (1.0 - x) @ (1.0 - x.t())
        network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()  # hamming distance

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    # this is the logarithm of the determinant of K
    def hooklogdet(K, labels=None):
        s, ld = np.linalg.slogdet(K)
        return ld

    # initialize K and forward and backward hook so that they are called in forward pass
    network.K = np.zeros((batch_size, batch_size))
    for name, module in network.named_modules():
        if "ReLU" in str(type(module)):
            module.register_full_backward_hook(counting_backward_hook)
            module.register_forward_hook(counting_forward_hook)

    # run batch through network
    x, target = next(iter(loader_train))
    x2 = torch.clone(x)

    x, target = x, target
    network(x2)
    score = hooklogdet(network.K, target)
    return score
