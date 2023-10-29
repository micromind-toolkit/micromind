"""
micromind checkpointer. Unwraps models and saves the to disk with optimizer's
state etc.

Authors:
    - Francesco Paissan, 2023
"""
from typing import Union, Dict, Callable
from loguru import logger
from pathlib import Path
import os

import torch


class Checkpointer:
    def __init__(
        self,
        key: str,
        mode: str = "min",
        top_k: int = 1,
        checkpoint_path: Union[str, Path] = ".",
    ) -> None:
        assert mode in ["max", "min"], "Checkpointer mode can be only max or min."
        self.key = key
        self.mode = mode
        self.top_k = top_k

        self.bests = [torch.inf] * self.top_k
        self.check_paths = [""] * self.top_k
        self.root_dir = checkpoint_path
        self.save_dir = os.path.join(self.root_dir, "save")
        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(
        self,
        mind,
        epoch: int,
        train_metrics: Dict,
        metrics: Dict,
        unwrap: Callable = lambda x: x,
    ) -> Union[Path, str]:
        self.fstream = open(os.path.join(self.root_dir, "train_log.txt"), "a")
        s_out = (
            f"Epoch {epoch}: "
            + " - ".join([f"{k}: {v:.2f}" for k, v in train_metrics.items()])
            + "; "
        )
        s_out += " - ".join([f"{k2}: {v2:.4f}" for k2, v2 in metrics.items()]) + ".\n"
        self.fstream.write(s_out)
        logger.info(s_out)
        base_save = {
            "key": self.key,
            "mode": self.mode,
            "epoch": epoch,
            "optimizer": mind.opt,
            "lr_scheduler": mind.lr_sched,
        }
        to_remove = None
        if self.mode == "min":
            if metrics[self.key] <= min(self.bests):
                id_best = self.bests.index(min(self.bests))
                to_remove = self.check_paths[id_best]

                self.check_paths[id_best] = os.path.join(
                    self.save_dir,
                    f"epoch_{epoch}_{self.key}_{metrics[self.key]:.4f}.ckpt",
                )

                base_save.update(
                    {k: unwrap(v).state_dict() for k, v in mind.modules.items()}
                ),
                self.bests[id_best] = metrics[self.key]

                torch.save(base_save, self.check_paths[id_best])
        elif self.mode == "max":
            if metrics[self.key] >= max(self.bests):
                id_best = self.bests.index(min(self.bests))
                to_remove = self.check_paths[id_best]

                self.check_paths[id_best] = os.path.join(
                    self.save_dir,
                    f"epoch_{epoch}_{self.key}_{metrics[self.key]:.4f}.ckpt",
                )

                base_save.update(
                    {k: unwrap(v).state_dict() for k, v in mind.modules.items()}
                ),
                self.bests[id_best] = metrics[self.key]

                torch.save(base_save, self.check_paths[id_best])

        if to_remove is not None and to_remove != "":
            logger.info(f"Generated better checkpoint. Deleting {to_remove}.")
            os.remove(to_remove)

        self.fstream.close()

        if self.mode == "max":
            return self.check_paths[self.bests.index(max(self.bests))]
        elif self.mode == "min":
            return self.check_paths[self.bests.index(min(self.bests))]
