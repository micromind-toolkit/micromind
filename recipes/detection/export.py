# from ultralytics import YOLO
from micromind import microYOLO

# Export

# select the file .pt that you want to export
filename = "./benchmark/weights/_new_start/deeper033_3heads/weights/best.pt"
# select also the task associated to the neural network
task = "detect"

# load the network once
model = microYOLO(model=filename, task=task)
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# export the network
export_filename = model.export(
    imgsz=160, format="tflite", half=True, int8=True, device="cpu"
)

# the exported network is saved in the same folder of the original network
print("Exported network: " + export_filename)
