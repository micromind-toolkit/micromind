from micromind import microYOLO

# Export

# select the file .pt that you want to export
filename = "./benchmark/weights/_new_start/deeper033_3heads/weights/best.pt"
# select also the task associated to the neural network
task = "detect"

# load the network once
model = microYOLO(model=filename, task=task)

# export the network
export_filename = model.export(
    imgsz=320, format="onnx", half=True, int8=False, device="cpu"
)

# the exported network is saved in the same folder of the original network
print("Exported network: " + export_filename)
