# from ultralytics import YOLO
from micromind import microYOLO

# import torch

# select the file .pt that you want to export
filename = "./benchmark/weights/033_testing_new_refactor/weights/best.pt"

# load the network once
print("Loading network: " + filename)
model = microYOLO(model=filename, task="detect")
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# apparently update the network
# print("Saving updated network: " + filename.replace("best.pt", "model-updated.pt"))
# torch.load(filename)
# torch.save(model, filename.replace("best.pt", "model-updated.pt"))

# export the network
# Args:
#     frame: The image to be preprocessed.
#     imgsz: is the size of the input image at inference time
#     format: can be tflite or onnx
#     half: (floating point using 16 bit)
#     int8: (integer using 8 bit) (only for tflite)
#     device: can be cpu or cuda (0,1... )
export_filename = model.export(
    imgsz=160, format="tflite", half=True, int8=True, device="cpu"
)

# the exported network is saved in the same folder of the original network
print("Networks exported.")
