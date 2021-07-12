from model.phinets import PhiNetConvBlock

from torch.autograd import Variable
import numpy as np
import torch


rand_x = Variable(torch.Tensor(np.random.rand(1, 3, 224, 224)))
conv_block = PhiNetConvBlock(in_shape = rand_x.shape, expansion = 1, stride = 1, filters = 16, block_id = 1, has_se = True, res=True, h_swish=True, k_size=3)

# print(conv_block.forward(rand_x))