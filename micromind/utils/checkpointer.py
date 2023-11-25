"""
micromind checkpointer. Unwraps models and saves the to disk with optimizer's
state etc.

Authors:
    - Francesco Paissan, 2023
"""
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import yaml

from .helpers import get_logger

logger = get_logger()


def create_experiment_folder(
    output_folder: Union[Path, str], exp_name: Union[Path, str]
) -> Path:
    """Creates the experiment folder used to log data.

    Arguments
    ---------
    output_folder : Union[Path, str]
        General output folder (can be shared between more experiments).
    exp_name  : Union[Path, str]
        Name of the experiment, to be concatenated to the output_folder.

    Returns
    -------
    Experiment folder : Union[Path, str]
    """
    exp_folder = os.path.join(output_folder, exp_name)

    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(os.path.join(exp_folder, "save"), exist_ok=True)

    return exp_folder


class Checkpointer:
    """Checkpointer class. Supports min/max modes for arbitrary keys (Metrics or loss).
    Always saves best and last in the experiment folder.

    Arguments
    ---------

    experiment_folder : Union[str, Path]
        Experiment folder. Used to load / store checkpoints.
    key: Optional[str]
        Key to be logged. It should be the name of the Metric, or "loss".
        Defaults to "loss".
    mode: Optional[str]
        Either `min` or `max`. If min, will store the checkpoint with the lowest
        value for key. If max, it does the opposite.

    Example
    -------
    .. doctest::
        >>> from micromind.utils.checkpointer import Checkpointer
        >>> from micromind.utils.checkpointer import create_experiment_folder
        >>> exp_folder = create_experiment_folder("/tmp", "test_mm")
        >>> check = Checkpointer(exp_folder)
    """

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

        # if true, does not write on disk
        self.debug = False

    def recover_state(self):
        """Recovers last corrected state of the training. If found, returns
        the accelerate dump folder (for recovery) and the last epoch logged.

        Returns
        -------
        Checkpoint path and last epoch logged. : Tuple[str, int]
        """
        available_ckpts = list(Path(self.save_dir).iterdir())
        if len(available_ckpts) < 1:
            return
        dates = [
            datetime.strptime(str(ckpt.name), "%Y-%m-%d+%H-%M-%S")
            for ckpt in available_ckpts
        ]
        dates = sorted(dates, reverse=True)

        for date in dates:
            oldest_name = os.path.join(
                self.save_dir, date.strftime("%Y-%m-%d+%H-%M-%S")
            )
            try:
                print(os.path.join(oldest_name, "status.yaml"))
                with open(os.path.join(oldest_name, "status.yaml"), "r") as f:
                    dat = yaml.safe_load(f)

                epoch = dat["epoch"]
                self.bests = dat["metric"]
                self.key = dat["metric_key"]

                accelerate_path = os.path.join(oldest_name, "accelerate_dump")
                logger.info(
                    f"Recovered info from checkpoint {oldest_name} at epoch {epoch}."
                )
                logger.info(f"{self.key} was {self.bests:.4f} for this checkpoint.")

                return accelerate_path, epoch
            except Exception as e:
                logger.info(
                    " ".join(
                        f"Tried to recover checkpoint {oldest_name}, \
                    but it appears corrupted.".split()
                    )
                )
                logger.debug(str(e))
        return

    @staticmethod
    def dump_modules(modules, out_folder):
        """Dumps state dict for all elements in the modules."""
        base_save = {k: v.state_dict() for k, v in modules.items()}

        torch.save(base_save, os.path.join(out_folder, "state-dict.pth.tar"))

    @staticmethod
    def dump_status(status, out_dir):
        """Dumps the status of the training."""
        yaml_status = yaml.dump(status)

        with open(os.path.join(out_dir, "status.yaml"), "w") as f:
            f.write(yaml_status)

    def __call__(
        self,
        mind,
        train_metrics: Dict,
        metrics: Dict,
    ) -> Union[Path, str]:
        """Does one checkpointing step.
        Arguments
        ---------
        mind : mm.Micromind
            Mind to be saved, eventually.
        train_metrics : Dict
            Training metrics, used only for the `train_log.txt` and the `stdout`.
        metrics : Dict
            Validation metrics, used to check if the checkpoint improved.

        Returns
        -------
        Current best checkpoint : Union[str, Path]
        """
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
        if not self.debug:
            self.fstream.write(s_out)
        logger.info(s_out)

        if not self.debug:
            mind.accelerator.save_state(os.path.join(current_folder, "accelerate_dump"))
            self.dump_modules(mind.modules, current_folder)
            self.dump_status(status_dict, current_folder)

        # remove previous last dir after saving the current version
        if (
            os.path.exists(self.last_dir)
            and self.last_dir != self.check_paths
            and not self.debug
        ):
            shutil.rmtree(self.last_dir)

        self.last_dir = current_folder

        to_remove = None
        if self.mode == "min":
            if metrics[self.key] <= self.bests:
                to_remove = self.check_paths

                if not self.debug:
                    mind.accelerator.save_state(
                        os.path.join(current_folder, "accelerate_dump")
                    )
                    self.dump_modules(mind.modules, current_folder)
                    self.dump_status(status_dict, current_folder)

                self.bests = metrics[self.key]
                self.check_paths = current_folder
                logger.info(
                    f"Generated better checkpoint at epoch {mind.current_epoch}."
                )

        elif self.mode == "max":
            if metrics[self.key] >= self.bests:
                to_remove = self.check_paths

                if not self.debug:
                    mind.accelerator.save_state(
                        os.path.join(current_folder, "accelerate_dump")
                    )
                    self.dump_modules(mind.modules, current_folder)
                    self.dump_status(status_dict, current_folder)

                self.bests = metrics[self.key]
                self.check_paths = current_folder
                logger.info(
                    f"Generated better checkpoint at epoch {mind.current_epoch}."
                )

        if to_remove is not None and to_remove != "" and not self.debug:
            logger.info(f"Deleting {to_remove}.")
            if os.path.exists(to_remove):
                shutil.rmtree(to_remove)

        self.fstream.close()

        return self.check_paths
