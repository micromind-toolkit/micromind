from phinet_pl.phinet import PhiNet

from torch.autograd import Variable
import numpy as np
import torch

from torchinfo import summary


if __name__ == "__main__":
    pn = PhiNet(res=128, B0=7, alpha=0.25, beta=1, t_zero=6, squeeze_excite=True, h_swish=True, include_top=False).to("cpu")
    
    rand_x = Variable(torch.Tensor(np.random.rand(1, 3, 128, 128))).to("cpu")

    # print(pn.forward(rand_x).shape)
    summary(pn, input_data=rand_x, col_names = ["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1, device="cpu")
