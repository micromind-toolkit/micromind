from huggingface_hub import hf_hub_download
from pathlib import Path
import yaml

# Data configuration
batch_size = 8
data_cfg="cfg/data/coco8.yaml"

# Architecture definition
REPO_ID = "micromind/ImageNet"
FILENAME = "v1/state_dict.pth.tar"

model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
args = Path(FILENAME).parent.joinpath("args.yaml")
args_path = hf_hub_download(repo_id=REPO_ID, filename=str(args))
with open(args_path, "r") as f:
    dat = yaml.safe_load(f)

input_shape = (3, 672, 672)
alpha = dat["alpha"]
num_layers = dat["num_layers"]
beta = dat["beta"]
t_zero = dat["t_zero"]
divisor = 8
downsampling_layers = [5, 7]
return_layers = [4, 6, 7]
