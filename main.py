from model.phinet import PhiNet

from torch.autograd import Variable
import numpy as np
import torch

if __name__ == "__main__":
    pn = PhiNet(h_swish=True)
    rand_x = Variable(torch.Tensor(np.random.rand(1, 3, 96, 96)))

    print(pn(rand_x).shape)
