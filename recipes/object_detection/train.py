from micromind.yolo.model import microYOLO

# define backbone
backbone = PhiNet(
    input_shape=(3, 320, 320),
    alpha=0.67,
    num_layers=6,
    beta=1,
    t_zero=4,
    include_top=False,
    num_classes=nc,
    compatibility=False,
)
# define head
head = Microhead()

# load a model
model = microYOLO()  # build a new model from scratch DEFAULT_CFG
# model = microYOLO('weights.pt') # load a pretrained model (recommended for training)

# Train the model
model.train(data="coco128.yaml", epochs=5, imgsz=320, device="cpu", task="detect")
model.export()
