from modules.model import microYOLO

# load a model
model = microYOLO()  # build a new model from scratch DEFAULT_CFG
# model = microYOLO('weights.pt') # load a pretrained model (recommended for training)

# Train the model
model.train(data="coco128.yaml", epochs=5, imgsz=320, device="cpu")
model.export()
