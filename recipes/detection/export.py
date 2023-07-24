from ultralytics import YOLO
from micromind import microYOLO

# select the file .pt that you want to export
filename = "./file/path/best.pt"

# select also the task associated to the neural network
task = "detect"

# load the network once
model = microYOLO(model=filename, task=task)
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

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
print("Exported network: " + export_filename)
