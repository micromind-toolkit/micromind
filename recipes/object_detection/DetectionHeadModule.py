# Ultralytics YOLO custom model DetectionHeadModule.py

import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from micromind import PhiNet

from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo.utils import LOGGER

from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment)
from ultralytics.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.yolo.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights,
                                                intersect_dicts, make_divisible, model_info, scale_img, time_sync)

try:
    import thop
except ImportError:
    thop = None

class DetectionHeadModel(DetectionModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        
        super(DetectionModel, self).__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        
        # Read custom hardcoded model
        self.model, self.save = parse_model_custom(nc, ch=ch, verbose=verbose)  # model, savelist
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

def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_stem(new_stem)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d

def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale.
    The function uses regular expression matching to find the pattern of the model scale in the YAML file name,
    which is denoted by n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str) or (Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ''


def parse_model_custom(nc, ch, verbose=True):

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
    layer9 = SPPF(256,256, 5)
    layer9.i, layer9.f, layer9.type = 9, -1, 'ultralytics.nn.modules.block.SPPF'
    layers.append(layer9)
    save.extend(x % layer9.i for x in ([layer9.f] if isinstance(layer9.f, int) else layer9.f) if x != -1)  # append to savelist

    layer10 = nn.Upsample(scale_factor=2, mode='nearest')
    layer10.i, layer10.f, layer10.type = 10, -1, 'torch.nn.modules.upsampling.Upsample'
    layers.append(layer10)
    save.extend(x % layer10.i for x in ([layer10.f] if isinstance(layer10.f, int) else layer10.f) if x != -1)  # append to savelist

    #layer11 = Concat((layers[-1], layers[6]))
    layer11 = Concat(1)
    layer11.i, layer11.f, layer11.type = 11, [-1,6], 'ultralytics.nn.modules.conv.Concat'
    c2 = sum(ch[x] for x in layer11.f)
    save.extend(x % layer11.i for x in ([layer11.f] if isinstance(layer11.f, int) else layer11.f) if x != -1)  # append to savelist
    layers.append(layer11)    

    layer12 = C2f(384, 128, 1)
    layer12.i, layer12.f, layer12.type = 12, -1, 'ultralytics.nn.modules.block.C2f'
    layers.append(layer12)
    save.extend(x % layer12.i for x in ([layer12.f] if isinstance(layer12.f, int) else layer12.f) if x != -1)  # append to savelist

    layer13 = nn.Upsample(scale_factor=2, mode='nearest')
    layer13.i, layer13.f, layer13.type = 13, -1, 'torch.nn.modules.upsampling.Upsample'
    layers.append(layer13)   
    save.extend(x % layer13.i for x in ([layer13.f] if isinstance(layer13.f, int) else layer13.f) if x != -1)  # append to savelist

    layer14 = Concat(1)
    layer14.i, layer14.f, layer14.type = 14, [-1,4], 'ultralytics.nn.modules.conv.Concat'
    layers.append(layer14)
    save.extend(x % layer14.i for x in ([layer14.f] if isinstance(layer14.f, int) else layer14.f) if x != -1)  # append to savelist

    layer15 = C2f(192, 64, 1)
    layer15.i, layer15.f, layer15.type = 15, -1, 'ultralytics.nn.modules.block.C2f'
    layers.append(layer15)
    save.extend(x % layer15.i for x in ([layer15.f] if isinstance(layer15.f, int) else layer15.f) if x != -1)  # append to savelist

    layer16 = Conv(64,64,3,2)
    layer16.i, layer16.f, layer16.type = 16, -1, 'ultralytics.nn.modules.conv.Conv'
    layers.append(layer16)
    save.extend(x % layer16.i for x in ([layer16.f] if isinstance(layer16.f, int) else layer16.f) if x != -1)  # append to savelist

    layer17 = Concat(1)
    layer17.i, layer17.f, layer17.type = 17, [-1,12], 'ultralytics.nn.modules.conv.Concat'
    layers.append(layer17)
    save.extend(x % layer17.i for x in ([layer17.f] if isinstance(layer17.f, int) else layer17.f) if x != -1)  # append to savelist

    layer18 = C2f(192, 128, 1)
    layer18.i, layer18.f, layer18.type = 18, -1, 'ultralytics.nn.modules.block.C2f'
    layers.append(layer18)
    save.extend(x % layer18.i for x in ([layer18.f] if isinstance(layer18.f, int) else layer18.f) if x != -1)  # append to savelist
    
    layer19 = Conv(128,128,3,2)
    layer19.i, layer19.f, layer19.type = 19, -1, 'ultralytics.nn.modules.conv.Conv'
    layers.append(layer19)
    save.extend(x % layer19.i for x in ([layer19.f] if isinstance(layer19.f, int) else layer19.f) if x != -1)  # append to savelist

    layer20 = Concat(1)
    layer20.i, layer20.f, layer20.type = 20, [-1,8], 'ultralytics.nn.modules.conv.Concat'
    layers.append(layer20)
    save.extend(x % layer20.i for x in ([layer20.f] if isinstance(layer20.f, int) else layer20.f) if x != -1)  # append to savelist

    layer21 = C2f(384, 256, 1)
    layer21.i, layer21.f, layer21.type = 21, -1, 'ultralytics.nn.modules.block.C2f'
    layers.append(layer21)
    save.extend(x % layer21.i for x in ([layer21.f] if isinstance(layer21.f, int) else layer21.f) if x != -1)  # append to savelist

    head = Detect(80, [64,128,256])
    head.i, head.f, head.type = 22, [15,18,21], 'ultralytics.nn.modules.conv.Detect'
    save.extend(x % head.i for x in ([head.f] if isinstance(head.f, int) else head.f) if x != -1)  # append to savelist

    layers.append(head)    

    return nn.Sequential(*layers), sorted(save)