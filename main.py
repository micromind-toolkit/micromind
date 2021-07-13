from model.phinet import PhiNet

from torch.autograd import Variable
import numpy as np
import torch

from torchinfo import summary


if __name__ == "__main__":
    pn = PhiNet(res=128, B0=7, alpha=0.2, beta=1, t_zero=6, squeeze_excite=True, h_swish=True)
    
    rand_x = Variable(torch.Tensor(np.random.rand(1, 3, 128, 128)))
    print(pn)

    print(pn.forward(rand_x).shape)

    model_parameters = filter(lambda p: p.requires_grad, pn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(summary(pn, input_shape=(1, 3, 128, 128)))
