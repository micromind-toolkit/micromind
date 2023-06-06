# Ultralytics YOLO custom model DetectionHeadModule.py

import torch
import torch.nn as nn

from micromind import PhiNet

from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo.utils import LOGGER

from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment)
from ultralytics.yolo.utils.torch_utils import (initialize_weights)

class DetectionHeadModel(DetectionModel):
    """YOLOv8 detection model."""

    def __init__(self, ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        super().__init__()
        
        # Read custom hardcoded model
        self.model, self.save = parse_model_custom(ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

def get_output_dim(data_config, model):     
    """Returns intermediate representations shapes."""     
    x = torch.randn(*[1] + list(data_config["input_size"]))
    out_dim = [model._layers[0](x)]     
    names = [model._layers[0].__class__]     
    for layer in model._layers[1:]:         
        out_dim.append(layer(out_dim[-1]))
        names.append(layer.__class__)
    return [list(o.shape)[1:] for o in out_dim], names

def parse_model_custom(ch, verbose=True):

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # START layers phinet ------------------------------------
    model = PhiNet(
            input_shape=(3,320,320),
            alpha=2.67,
            num_layers=6,
            beta=1,
            t_zero=4,
            include_top=False,
            num_classes=nc,
            compatibility=False,
        )


    # maybe we can take away this code?
    data_config = {"input_size":(3,320,320)}
    res = get_output_dim(data_config, model)    

    for i, layer in enumerate(model._layers):        
        f = -1
        n = n_ = 1
        args = [2.67, 1, 4] + res[0][i]
        t = str(layer.__class__).replace("<class '",'').replace("'>",'')       
        # get the shape
        
        
        #t = str(m)[8:-2].replace('__main__.', '')  # module type
        layer.np = sum(x.numel() for x in layer.parameters())  # number params
        layer.i, layer.f, layer.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{layer.np:10.0f}  {t:<45}{str(args):<30}')  # print            

        c2 = ch[f]
        if i == 0:
            ch = []

        # da refactorizzare
        try:
            ch.append(res[0][i-1][0])
        except:
            ch.append(c2) # take the number of layers

    layers = list(model._layers)
    # END layers phinet ------------------------------------ 

    # START HARDCODED detection head ------------------------------------
    
    layers.append(SPPF(256,256, 5))
    layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

    #https://discuss.pytorch.org/t/how-to-concatenate-layers-in-pytorch-similar-to-tf-keras-layers-concatenate/33736



    pass