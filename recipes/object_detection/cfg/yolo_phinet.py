"""
YOLOPhiNet training configuration.

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023
"""
# Data configuration
batch_size = 8
data_cfg = "cfg/data/coco.yaml"
data_dir = "data/coco"

# Model configuration
input_shape = (3, 672, 672)
alpha = 2.3
num_layers = 7
beta = 0.75
t_zero = 5
divisor = 8
downsampling_layers = [5, 7]
return_layers = [4, 6, 7]

# Placeholder for inference
ckpt_pretrained = ""
output_dir = "detection_output"
coco_names = "cfg/data/coco.names"
