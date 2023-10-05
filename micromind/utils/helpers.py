from typing import Union, Dict, Tuple
from pathlib import Path

from loguru import logger
import torch
import os


def get_value_from_key(s:str, key: str, cast=float) -> float:
    dat = s.split(f"{key}_")[-1]

    if "ckpt" in dat:
        dat = dat.split(".ckpt")[0]

    return cast(dat)

def select_and_load_checkpoint(path: Union[Path, str]) -> Tuple[Dict, str]:
    checkpoints = os.listdir(path)
    checkpoints = [
        os.path.join(path, c) for c in checkpoints
    ]

    dat = torch.load(checkpoints[0])
    selected_key, selected_mode = dat["key"], dat["mode"]

    values = [
         get_value_from_key(str(c), selected_key) for c in checkpoints
    ]

    best_key = min(values) if selected_mode == "min" else max(values)
    best_checkpoint = checkpoints[values.index(best_key)]

    return torch.load(best_checkpoint), best_checkpoint

