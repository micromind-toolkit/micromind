from abc import ABC, abstractmethod
from tqdm import tqdm

from typing import Dict, Union
from torch.cuda import device

import torch.nn as nn
import torch

class MicroMind(ABC):
    def __init__(self):
        # here we should handle devices etc.
        self.modules = torch.nn.ModuleList([]) # init empty modules list

        self.device = "cpu"

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def compute_loss(self, pred, batch):
        pass

    def configure_optimizers(self):
        opt_conf = {"lr": 0.001, "momentum": 0.9}
        opt = torch.optim.SGD(self.modules.parameters(), **opt_conf)
        return opt, None    # None is for learning rate sched

    def __call__(self, *x, **xv):
        return self.forward(*x, **xv)

class MicroTrainer():
    def __init__(
        self,
        mind: MicroMind,
        epochs: int = 1,
        datasets: Dict = {},
        device: Union[str, device] = "cpu",
        debug: bool = False
    ) -> None:
        # Sanity checks
        assert datasets != {}, "You must specify at least one DataLoader."
        assert epochs > 0, "You must specify at least one epoch."

        # Put everything on the right device.
        mind.modules.to(device)
        self.device = device

        # Get optimizer
        self.opt, self.lr_sched = mind.configure_optimizers()
        mind.device = device

        # Store useful things
        self.epochs = epochs
        self.datasets = datasets
        self.debug = debug
        self.mind = mind

    def train(self) -> None:
        assert "train" in self.datasets, "Training dataloader was not specified."

        for e in range(self.epochs):
            pbar = tqdm(self.datasets["train"], unit="batches", ascii=True, dynamic_ncols=True) #, disable=not accelerator.is_local_main_process)
            loss_epoch = 0
            pbar.set_description(f"Running epoch {e + 1}/{self.epochs}")
            for idx, batch in enumerate(pbar):
                self.opt.zero_grad()
    
                loss = self.mind.compute_loss(self.mind(batch), batch)

                loss.backward()

                # accelerator.backward(loss)
                self.opt.step()
    
                loss_epoch += loss.item()
                pbar.set_postfix(loss=loss_epoch/(idx + 1))
    
                if self.debug and idx > 10: break

            pbar.close()

            if e >= 1 and self.debug: break     # not sure this is getting called

        return None



