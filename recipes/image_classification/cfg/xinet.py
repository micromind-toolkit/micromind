"""
Configuration file image classification with PhiNet.

Authors:
    - Francesco Paissan, 2023
"""

# Model configuration
model = "xinet"
input_shape = (3, 128, 128)
alpha = 1
num_layers = 7
return_layers = None
gamma = 4

ckpt_pretrained = ""

# Basic training loop
epochs = 50

# Basic data
data_dir = "data/cifar10/"
dataset = "torch/cifar10"
batch_size = 256
dataset_download = True

# Dataloading config
num_workers = 4
pin_memory = True
persistent_workers = True

# Loss function
bce_loss = False
bce_target_thresh = None

# Data augmentation config
aa = "rand-m8-inc1-mstd101"
aug_repeats = 0
aug_splits = 0
class_map = ""
color_jitter = 0.4
cutmix = 0.0
cutmix_minmax = None
drop = 0.0
drop_block = None
drop_connect = None
drop_path = 0.1
epoch_repeats = 0.0
hflip = 0.5
img_size = None
in_chans = None
initial_checkpoint = ""
interpolation = "bilinear"
jsd_loss = False
layer_decay = 0.65
local_rank = 0
log_interval = 50
log_wandb = False
lr = 0.001
lr_base = 0.1
lr_base_scale = ""
lr_base_size = 256
lr_cycle_decay = 0.5
lr_cycle_limit = 1
lr_cycle_mul = 1.0
lr_k_decay = 1.0
lr_noise = None
lr_noise_pct = 0.67
lr_noise_std = 1.0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mixup = 0.0
mixup_mode = "batch"
mixup_off_epoch = 0
mixup_prob = 1.0
mixup_switch_prob = 0.5
no_aug = False
num_classes = 100
ratio = [0.75, 1.3333333333333333]
recount = 1
recovery_interval = 0
remode = "pixel"
reprob = 0.3
scale = [0.08, 1.0]
smoothing = 0.1
train_interpolation = "bilinear"
train_split = "train"
use_multi_epochs_loader = False
val_split = "validation"
vflip = 0.0
