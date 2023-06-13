import torch
from modules.model import microYOLO

modelmicro = microYOLO()
modelmicro.load_state_dict(torch.load("./runs/microyolo-500-epochs/best.pt")).eval()
modelmicro.export(format="tflite", imgsz=320, int8=True)
modelmicro.export(format="onnx", imgsz=320, simplify=True, int8=True)
