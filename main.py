from model.phinet_convblock import PhiNetConvBlock

from torch.autograd import Variable
import numpy as np
import torch


rand_x = Variable(torch.Tensor(np.random.rand(1, 4, 48, 48)))
conv_block = PhiNetConvBlock(in_shape = rand_x.shape[1:], expansion = 6, stride = 1, filters = 4, block_id = 1, has_se = True, res=True, h_swish=True, k_size=3)

print(conv_block.forward(rand_x).shape)