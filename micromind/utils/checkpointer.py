"""
micromind checkpointer. Unwraps models and saves the to disk with optimizer's
state etc.

Authors:
    - Francesco Paissan, 2023
"""
from typing import Union, Dict, Optional
from datetime import datetime
from loguru import logger
from pathlib import Path
import shutil
import yaml
import os

import torch


def create_experiment_folder(
    output_folder: Union[Path, str], exp_name: Union[Path, str]
) -> Path:
    exp_folder = os.path.join(output_folder, exp_name)

    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(os.path.join(exp_folder, "save"), exist_ok=True)

    return exp_folder


class Checkpointer:
    def __init__(
        self,
        experiment_folder: Union[str, Path],
        key: Optional[str] = "loss",
        mode: Optional[str] = "min",
    ) -> None:
        assert experiment_folder != "", "You should pass a valid experiment folder."
        assert os.path.exists(
            os.path.join(experiment_folder, "save")
        ), "Invalid experiment folder."
        assert mode in ["max", "min"], "Checkpointer mode can be only max or min."
        self.key = "val_" + key
        self.mode = mode

        self.bests = torch.inf if mode == "min" else -torch.inf
        self.check_paths = ""
        self.root_dir = experiment_folder
        self.save_dir = os.path.join(self.root_dir, "save")
        self.last_dir = "default"

    # def recover_from_checkpoint(self):

    @staticmethod
    def dump_modules(modules, out_folder):
        base_save = {k: v.state_dict() for k, v in modules.items()}

        torch.save(base_save, os.path.join(out_folder, "state-dict.pth.tar"))

    @staticmethod
    def dump_status(status, out_dir):
        yaml_status = yaml.dump(status)

        with open(os.path.join(out_dir, "status.yaml"), "w") as f:
            f.write(yaml_status)

    def __call__(
        self,
        mind,
        train_metrics: Dict,
        metrics: Dict,
    ) -> Union[Path, str]:
        current_folder = datetime.now().strftime("%Y-%m-%d+%H-%M-%S")
        current_folder = os.path.join(self.save_dir, current_folder)
        os.makedirs(current_folder, exist_ok=True)

        status_dict = {
            "epoch": mind.current_epoch,
            "metric": metrics[self.key],
            "metric_key": self.key,
        }

        self.fstream = open(os.path.join(self.root_dir, "train_log.txt"), "a")
        s_out = (
            f"Epoch {mind.current_epoch}: "
            + " - ".join([f"{k}: {v:.2f}" for k, v in train_metrics.items()])
            + "; "
        )
        s_out += " - ".join([f"{k2}: {v2:.4f}" for k2, v2 in metrics.items()]) + ".\n"
        self.fstream.write(s_out)
        logger.info(s_out)

        mind.accelerator.save_state(os.path.join(current_folder, "accelerate_dump"))
        self.dump_modules(mind.modules, current_folder)
        self.dump_status(status_dict, current_folder)

        # remove previous last dir after saving the current version
        if os.path.exists(self.last_dir) and self.last_dir != self.check_paths:
            shutil.rmtree(self.last_dir)

        self.last_dir = current_folder

        to_remove = None
        if self.mode == "min":
            if metrics[self.key] <= self.bests:
                to_remove = self.check_paths

                mind.accelerator.save_state(
                    os.path.join(current_folder, "accelerate_dump")
                )
                self.dump_modules(mind.modules, current_folder)
                self.dump_status(status_dict, current_folder)

                self.bests = metrics[self.key]
                self.check_paths = current_folder

        elif self.mode == "max":
            if metrics[self.key] >= self.bests:
                to_remove = self.check_paths

                mind.accelerator.save_state(
                    os.path.join(current_folder, "accelerate_dump")
                )
                self.dump_modules(mind.modules, current_folder)
                self.dump_status(status_dict, current_folder)

                self.bests = metrics[self.key]
                self.check_paths = current_folder

        if to_remove is not None and to_remove != "":
            logger.info(f"Generated better checkpoint. Deleting {to_remove}.")
            if os.path.exists(to_remove):
                shutil.rmtree(to_remove)

        self.fstream.close()

        return self.check_paths
