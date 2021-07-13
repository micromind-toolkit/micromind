from model.phinet import PhiNet

from torch.autograd import Variable
import numpy as np
import torch

from torchinfo import summary


if __name__ == "__main__":
    pn = PhiNet(res=128, B0=7, alpha=0.2, beta=1, t_zero=6, squeeze_excite=True, h_swish=True)
    
    rand_x = Variable(torch.Tensor(np.random.rand(1, 3, 128, 128)))

    print(pn.forward(rand_x).shape)

    summary(pn, input_shape=(1, 3, 128, 128))
