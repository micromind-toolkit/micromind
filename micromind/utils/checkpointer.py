from typing import Union, Dict
from loguru import logger
from pathlib import Path
import os

import torch

class Checkpointer():   # should look if something is inside this folder, in case start the exp from there.
    def __init__(
        self,
        key: str,
        mode: str = "min",
        top_k: int = 5,
        checkpoint_path: Union[str, Path] = "."
    ) -> None:
        assert mode in ["max", "min"], "Checkpointer mode can be only max or min."
        self.key = key
        self.mode = mode
        self.top_k = 5

        self.bests = [torch.inf] * self.top_k
        self.check_paths = [""] * self.top_k
        self.root_dir = checkpoint_path
        self.save_dir = os.path.join(
            self.root_dir, "save"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.fstream = open(
            os.path.join(self.root_dir, "train_log.txt"), "a"
        )

    def __call__(self, mind, epoch: int, metrics: Dict) -> Union[Path, str]:
        self.fstream.write(
            f"Epoch {epoch}: " + " - ".join([f"{k}: {v:.4f}" for k,v in metrics.items()]) + ".\n"
        )
        base_save = {
            "key": self.key,
            "mode": self.mode,
            "epoch": epoch,
            "optimizer": mind.opt,
            "lr_scheduler": mind.lr_sched,
        }
        if self.mode == "min":
            if metrics[self.key] <= min(self.bests):
                id_best = self.bests.index(min(self.bests))
                to_remove = self.check_paths[id_best]

                self.check_paths[id_best] = os.path.join(
                    self.save_dir,
                    f"epoch_{epoch}_{self.key}_{metrics[self.key]:.4f}.ckpt"
                )

                base_save.update({k: v.state_dict() for k, v in mind.modules.items()}),
                torch.save(
                    base_save,
                    self.check_paths[id_best]
                )
        elif self.mode == "max":
            if metrics[self.key] >= max(self.bests):
                id_best = self.bests.index(min(self.bests))
                to_remove = self.check_paths[id_best]

                self.check_paths[id_best] = os.path.join(
                    self.save_dir,
                    f"epoch_{epoch}_{key}_{metrics[key]:.4f}.ckpt"
                )

                base_save.update({k: v.state_dict() for k, v in mind.modules.items()}),
                torch.save(
                    base_save,
                    self.check_paths[id_best]
                )

        if self.mode == "max":
            return self.check_paths[self.bests.index(max(self.bests))]
        elif self.mode == "min":
            return self.check_paths[self.bests.index(min(self.bests))]

    def close(self):
        self.fstream.close()

