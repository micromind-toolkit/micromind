"""
micromind helper functions.

Authors:
    - Francesco Paissan, 2023
"""
import sys
from pathlib import Path
from typing import Dict, Union
from argparse import Namespace

from loguru import logger
import micromind as mm
import argparse


def override_conf(hparams: Dict):
    """Handles command line overrides. Takes as input a configuration
    and defines all the keys as arguments. If passed from command line,
    these arguments override the default configuration.

    Arguments
    ---------
    hparams : Dict
        Dictionary containing current configuration.

    Returns
    -------
    Configuration agumented with overrides. : Namespace

    """
    parser = argparse.ArgumentParser(description="MicroMind experiment configuration.")
    for key, value in hparams.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

    args, extra_args = parser.parse_known_args()
    for key, value in vars(args).items():
        if value is not None:
            hparams[key] = value

    return Namespace(**hparams)


def parse_configuration(cfg: Union[str, Path]):
    """Parses default configuration and compares it with user defined.
    It processes a user-defined python file that creates the configuration.
    Additionally, it handles eventual overrides from command line.

    Arguments
    ---------
    cfg : Union[str, Path]
        Configuration file defined by the user

    Returns
    -------
    Configuration Namespace. : argparse.Namespace

    """
    with open(cfg, "r") as f:
        conf = f.read()

    local_vars = {}

    exec(conf, {}, local_vars)
    for key in mm.core.default_cfg:
        if key not in local_vars:
            local_vars[key] = mm.core.default_cfg[key]

    return override_conf(local_vars)


def get_logger():
    """Default loguru logger config. It is called inside micromind's files."""
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | \
            <level>{level: <8}</level> |  \
            <level>{message}</level>"
    logger.remove()
    logger.add(sys.stderr, format=fmt)

    return logger
