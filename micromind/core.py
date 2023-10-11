from typing import Dict, Union, Tuple, Callable, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path

from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm
import os

from torch.cuda import device
import torch.nn as nn
import torch

from .utils.helpers import select_and_load_checkpoint
from .utils.checkpointer import Checkpointer

@dataclass
class Stage:
    train: int = 0
    val: int = 1
    test: int = 2

class Metric():
    def __init__(self, name: str, fn: Callable, reduction="mean"):
        self.name = name
        self.fn = fn
        self.reduction = reduction
        self.history = {
            s: torch.empty(1,) for s in [Stage.train, Stage.val, Stage.test]
        }

    def __call__(self, pred, batch, stage, device="cpu"):
        if self.history[stage].device != device: self.history[stage] = self.history[stage].to(device)

        self.history[stage] = torch.cat(
            (self.history[stage], self.fn(pred, batch))
        )

    def reduce(self, stage, clear=False):
        if self.reduction == "mean":
            tmp = self.history[stage].mean()
        elif self.reduction == "sum":
            tmp = self.history[stage].sum()

        if clear:
            del self.history[stage]
            self.history[stage] = torch.empty(1,)
        return tmp.item()

class MicroMind(ABC):
    def __init__(self, hparams):
        # here we should handle devices etc.
        self.modules = torch.nn.ModuleDict({}) # init empty modules dict
        self.hparams = hparams
        self.input_shape = None

        self.device = "cpu"
        self.accelerator = Accelerator()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def compute_loss(self, pred, batch):
        pass

    def set_input_shape(self, input_shape: Tuple = (3, 224, 224)):
        self.input_shape = input_shape

    def load_modules(self, checkpoint_path: Union[Path, str]):
        """ Loads models for path. """
        dat = torch.load(checkpoint_path)

        modules_keys = list(self.modules.keys())
        for k in self.modules:
            self.modules[k].load_state_dict(dat[k])

            modules_keys.remove(k)

        if len(modules_keys) != 0:
            print(modules_keys)
            breakpoint()
            logger.info(f"Couldn')t find a state_dict for modules {modules_keys}.")

    def export(self, save_dir: Union[Path, str], out_format: str = "onnx", input_shape=None) -> None:
        from micromind import convert
        if not isinstance(save_dir, Path): save_dir = Path(save_dir)
        save_dir = save_dir.joinpath(self.hparams.experiment_name)

        self.set_input_shape(input_shape)
        assert self.input_shape is not None, "You should pass the input_shape of the model."

        if out_format == "onnx":
            convert.convert_to_onnx(self, save_dir.joinpath("model.onnx"))
        elif out_format == "openvino":
            convert.convert_to_openvino(self, save_dir)
        elif out_format == "tflite":
            convert.convert_to_tflite(self, save_dir)

    def configure_optimizers(self):
        assert self.hparams.opt in ["adam", "sgd"], f"Optimizer {self.hparams.opt} not supported."
        if self.hparams.opt == "adam":
            opt = torch.optim.Adam(self.modules.parameters(), self.hparams.lr)
        elif self.hparams.opt == "sgd":
            opt = torch.optim.SGD(self.modules.parameters(), self.hparams.lr)
        return opt, None    # None is for learning rate sched

    def __call__(self, *x, **xv):
        return self.forward(*x, **xv)

    def on_train_start(self):
        self.experiment_folder = os.path.join(
            self.hparams.output_folder,
            self.hparams.experiment_name
        )

        save_dir = os.path.join(
                self.experiment_folder, "save"
            )
        if os.path.exists(save_dir):
            if len(os.listdir(save_dir)) != 0:
                # select which checkpoint and load it.
                checkpoint, path = select_and_load_checkpoint(save_dir)
                self.opt = checkpoint["optimizer"]
                self.lr_sched = checkpoint["lr_scheduler"]
                self.start_epoch = checkpoint["epoch"] + 1

                self.load_modules(path)

                if self.accelerator.is_local_main_process:
                    self.checkpointer = Checkpointer(
                        checkpoint["key"],
                        mode=checkpoint["mode"],
                        checkpoint_path=self.experiment_folder
                    )

                    logger.info(f"Loaded existing checkpoint from {path}.")
            else:
                self.opt, self.lr_sched = self.configure_optimizers()
                self.start_epoch = 0
    
                self.checkpointer = Checkpointer("val_loss", checkpoint_path=self.experiment_folder)
        else:
            os.makedirs(self.experiment_folder, exist_ok=True)

            self.opt, self.lr_sched = self.configure_optimizers()
            self.start_epoch = 0

            self.checkpointer = Checkpointer("val_loss", checkpoint_path=self.experiment_folder)


        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.modules.to(self.device)
        print("Set device to ", self.device)

        convert = [self.modules, self.opt, self.lr_sched] + list(self.datasets.values())
        accelerated = self.accelerator.prepare(convert)
        self.modules, self.opt, self.lr_sched = accelerated[:3]
        for i, key in enumerate(self.datasets):
            self.datasets[key] = accelerated[-(i + 1)]

    def on_train_end(self):
        if self.accelerator.is_local_main_process:
            self.checkpointer.close()

    def train(
            self,
            epochs: int = 1,
            datasets: Dict = {},
            metrics: List[Metric] = [],
            debug: bool = False
        ) -> None:
        self.datasets = datasets
        self.metrics = metrics
        assert "train" in self.datasets, "Training dataloader was not specified."
        assert epochs > 0, "You must specify at least one epoch."

        self.debug = debug

        self.on_train_start()

        if self.accelerator.is_local_main_process:
            logger.info(f"Starting from epoch {self.start_epoch}. Training is scheduled for {epochs} epochs.")
        with self.accelerator.autocast():
            for e in range(self.start_epoch, epochs):
                pbar = tqdm(self.datasets["train"], unit="batches", ascii=True, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process)
                loss_epoch = 0
                pbar.set_description(f"Running epoch {e + 1}/{epochs}")
                self.modules.train()
                for idx, batch in enumerate(pbar):
                    if isinstance(batch, list): 
                        batch = [b.to(self.device) for b in batch]

                    self.opt.zero_grad()
        
                    model_out = self(batch)
                    loss = self.compute_loss(model_out, batch)

                    self.accelerator.backward(loss)
                    self.opt.step()

                    for m in self.metrics:
                        m(model_out, batch, Stage.train, self.device)

                    
                    running_train = {
                        "train_" + m.name: m.reduce(Stage.train) for m in self.metrics
                    }

                    running_train.update({"train_loss": loss_epoch/(idx+1)})
        
                    loss_epoch += loss.item()
                    pbar.set_postfix(**running_train)
        
                    if self.debug and idx > 10: break
    
                pbar.close()
    
                train_metrics = {
                    "train_" + m.name: m.reduce(Stage.train, True) for m in self.metrics
                }
                train_metrics.update({"train_loss": loss_epoch/(idx+1)})

                if "val" in datasets:
                    val_metrics = self.validate()
                    if self.accelerator.is_local_main_process:
                        self.checkpointer(self, e, train_metrics, val_metrics, lambda x: self.accelerator.unwrap_model(x))
                else:
                    val_metrics = train_metrics.update({"val_loss": loss_epoch / (idx + 1)})
    
                if e >= 1 and self.debug: break


        self.on_train_end()
        return None

    @torch.no_grad()
    def validate(self) -> Dict:
        assert "val" in self.datasets, "Validation dataloader was not specified."
        self.modules.eval()

        pbar = tqdm(self.datasets["val"], unit="batches", ascii=True, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process)
        loss_epoch = 0
        pbar.set_description(f"Validation...")
        with self.accelerator.autocast():
            for idx, batch in enumerate(pbar):
                if isinstance(batch, list):
                    batch = [b.to(self.device) for b in batch]

                self.opt.zero_grad()
    
                model_out = self(batch)
                loss = self.compute_loss(model_out, batch)
                for m in self.metrics:
                    m(model_out, batch, Stage.val, self.device)
    
                loss_epoch += loss.item()
                pbar.set_postfix(loss=loss_epoch/(idx + 1))
    
                if self.debug and idx > 10: break
    
        val_metrics = {
            "val_" + m.name: m.reduce(Stage.val, True) for m in self.metrics
        }
        val_metrics.update({"val_loss": loss_epoch/(idx+1)})

        pbar.close()

        return val_metrics

    @torch.no_grad()
    def test(self, datasets: Dict = {}) -> None:
        assert "test" in self.datasets, "Test dataloader was not specified."
        self.modules.eval()

        pbar = tqdm(self.datasets["test"], unit="batches", ascii=True, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process)
        loss_epoch = 0
        pbar.set_description(f"Testing...")
        with self.accelerator.autocast():
            for idx, batch in enumerate(pbar):
                if isinstance(batch, list): 
                    batch = [b.to(self.device) for b in batch]
                self.opt.zero_grad()
    
                loss = self.compute_loss(self(batch), batch)
    
                loss_epoch += loss.item()
                pbar.set_postfix(loss=loss_epoch/(idx + 1))

        pbar.close()

        return None

