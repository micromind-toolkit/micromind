"""
micromind helper functions.

Authors:
    - Francesco Paissan, 2023
"""
import os
import random
import string
import sys
from pathlib import Path
from typing import Dict, Tuple, Union
from argparse import Namespace

import torch
from loguru import logger
import micromind as mm


def parse_configuration(cfg: Union[str, Path]):
    with open(cfg, "r") as f:
        conf = f.read()

    global_vars = {}
    local_vars = {}

    exec(conf, global_vars, local_vars)
    for key in mm.core.default_cfg:
        if not key in local_vars:
            local_vars[key] = mm.core.default_cfg[key]

    return Namespace(**local_vars)
    

def get_value_from_key(s: str, key: str, cast=float) -> float:
    dat = s.split(f"{key}_")[-1]

    if "ckpt" in dat:
        dat = dat.split(".ckpt")[0]

    return cast(dat)


def select_and_load_checkpoint(path: Union[Path, str]) -> Tuple[Dict, str]:
    checkpoints = os.listdir(path)
    checkpoints = [os.path.join(path, c) for c in checkpoints]

    dat = torch.load(checkpoints[0])
    selected_key, selected_mode = dat["key"], dat["mode"]

    values = [get_value_from_key(str(c), selected_key) for c in checkpoints]

    best_key = min(values) if selected_mode == "min" else max(values)
    best_checkpoint = checkpoints[values.index(best_key)]

    return torch.load(best_checkpoint, map_location="cpu"), best_checkpoint


def get_random_string(length=10):
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))

    return result_str


def get_logger():
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | \
            <level>{level: <8}</level> |  \
            <level>{message}</level>"
    logger.remove()
    logger.add(sys.stderr, format=fmt)

    return logger
