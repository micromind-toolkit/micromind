"""
Core class for micromind. Supports helper function for exports. Out-of-the-box
multi-gpu and FP16 training with HF Accelerate and much more.

Authors:
    - Francesco Paissan, 2023
"""
from typing import Dict, Union, Tuple, Callable, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from argparse import Namespace
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import shutil

from accelerate import Accelerator
import torch
import os

from .utils.helpers import select_and_load_checkpoint, get_random_string
from .utils.checkpointer import Checkpointer

# This is used ONLY if you are not using argparse to get the hparams
default_cfg = {
    "output_folder": "results",
    "experiment_name": "micromind_exp",
    "opt": "adam",  # this is ignored if you are overriding the configure_optimizers
    "lr": 0.001,  # this is ignored if you are overriding the configure_optimizers
    "debug": False,
}


@dataclass
class Stage:
    """enum to track training stage"""

    train: int = 0
    val: int = 1
    test: int = 2


class Metric:
    """
    Class for tracking evaluation metrics during training.

    This class allows you to create custom evaluation metrics by providing a
    function to compute the metric and specifying a reduction method.

    Arguments
    ---------
        name : str
            The name of the metric.
        fn : Callable
            A function that computes the metric given predictions and batch data.
        reduction : Optional[str]
            The reduction method for the metric ('sum' or 'mean'). Default is 'mean'.

    Returns
    -------
        Reduced metric. Optionally, you can access the metric history
        before call reduce(clear=True) : torch.Tensor

    Example
    -------
    .. doctest::

        >>> from micromind import Metric, Stage
        >>> import torch

        >>> def custom_metric(pred, batch):
        ...     # Replace this with your custom metric calculation
        ...     return pred - batch

        >>> metric = Metric("Custom Metric", custom_metric, reduction="mean")
        >>> pred = torch.tensor([1.0, 2.0, 3.0])
        >>> batch = torch.tensor([0.5, 1.5, 2.5])
        >>> metric(pred, batch, stage=Stage.train)
        >>> metric.history
        {0: [tensor([0.5000, 0.5000, 0.5000])], 1: [], 2: []}
        >>> metric.reduce(Stage.train)
        0.5
    """

    def __init__(self, name: str, fn: Callable, reduction="mean"):
        self.name = name
        self.fn = fn
        self.reduction = reduction
        self.history = {s: [] for s in [Stage.train, Stage.val, Stage.test]}

    def __call__(self, pred, batch, stage, device="cpu"):
        if pred.device != device:
            pred = pred.to(device)
        dat = self.fn(pred, batch)
        if dat.ndim == 0:
            dat = dat.unsqueeze(0)

        self.history[stage].append(self.fn(pred, batch))

    def reduce(self, stage, clear=False):
        """
        Compute and return the metric for a given prediction and batch data.

        Arguments
        -------
            pred : torch.Tensor
                The model's prediction.
            batch : torch.Tensor
                The ground truth or target values.
            stage : Stage
                The current stage (e.g., Stage.train).
            device Optional[str]
                The device on which to perform the computation. Default is 'cpu'.
        """

        if self.reduction == "mean":
            if clear or (
                self.history[stage][-1].shape[0] != self.history[stage][0].shape[0]
            ):
                tmp = torch.stack(self.history[stage][:-1]).mean()
            else:
                tmp = torch.stack(self.history[stage]).mean()
        elif self.reduction == "sum":
            if (
                clear
                or self.history[stage][-1].shape[0] != self.history[stage][0].shape[0]
            ):
                tmp = torch.stack(self.history[stage][:-1]).sum()
            else:
                tmp = torch.stack(self.history[stage]).sum()

        if clear:
            self.history[stage] = []
        return tmp.item()


class MicroMind(ABC):
    """
    MicroMind is an abstract base class for creating and training deep learning
    models. Handles training on multi-gpu via accelerate (using DDP and other
    distributed training strategies). It automatically handles the device
    management for the training and the micromind's export capabilities to onnx,
    OpenVino and TFLite.

    Arguments
    ---------
        hparams : Optional[Namespace]
            Hyperparameters for the model. Default is None.

    """

    def __init__(self, hparams=None):
        if hparams is None:
            hparams = Namespace(**default_cfg)

        # here we should handle devices etc.
        self.modules = torch.nn.ModuleDict({})  # init empty modules dict
        self.hparams = hparams
        self.input_shape = None

        self.device = "cpu"  # used just to init the models
        self.accelerator = Accelerator()

    @abstractmethod
    def forward(self, batch):
        """
        Forward step of the class. It gets called during inference and optimization.
        This method should be overwritten for specific applications.

        Arguments
        ---------
            batch : torch.Tensor
                Batch as output from the defined DataLoader.

        Returns
        -------
            pred : Union[torch.Tensor, Tuple]
                Predictions - this depends on the task.
        """
        pass

    @abstractmethod
    def compute_loss(self, pred, batch):
        """
        Computes the cost function for the optimization process.  It return a
        tensor on which backward() is called. This method should be overwritten
        for the specific application.

        Arguments
        ---------
            pred : Union[torch.Tensor, Tuple]
                Output of the forward() function
            batch : torch.Tensor
                Batch as defined from the DataLoader.

        Returns
        -------
            loss : torch.Tensor
                Compute cost function.
        """
        pass

    def set_input_shape(self, input_shape: Tuple = (3, 224, 224)):
        """Setter function for input_shape.

        Arguments
        ---------
            input_shape : Tuple
                Input shape of the forward step.

        """
        self.input_shape = input_shape

    def load_modules(self, checkpoint_path: Union[Path, str]):
        """Loads models for path.

        Arguments
        ---------
            checkpoint_path : Union[Path, str]
                Path to the checkpoint where the modules are stored.

        """
        dat = torch.load(checkpoint_path)

        modules_keys = list(self.modules.keys())
        for k in self.modules:
            self.modules[k].load_state_dict(dat[k])

            modules_keys.remove(k)

        if len(modules_keys) != 0:
            print(modules_keys)
            breakpoint()
            logger.info(f"Couldn't find a state_dict for modules {modules_keys}.")

    def export(
        self, save_dir: Union[Path, str], out_format: str = "onnx", input_shape=None
    ) -> None:
        """
        Export the model to a specified format for deployment.
        TFLite and OpenVINO need a Linux machine to be exported.


        Arguments
        ---------
        save_dir : Union[Path, str]
            The directory where the exported model will be saved.
        out_format : Optional[str]
            The format for exporting the model. Default is 'onnx'.
        input_shape : Optional[Tuple]
            The input shape of the model. If not provided, the input shape
            specified during model creation is used.

        """
        from micromind import convert

        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        save_dir = save_dir.joinpath(self.hparams.experiment_name)

        self.set_input_shape(input_shape)
        assert (
            self.input_shape is not None
        ), "You should pass the input_shape of the model."

        if out_format == "onnx":
            convert.convert_to_onnx(
                self, save_dir.joinpath("model.onnx"), replace_forward=True
            )
        elif out_format == "openvino":
            convert.convert_to_openvino(self, save_dir, replace_forward=True)
        elif out_format == "tflite":
            convert.convert_to_tflite(self, save_dir, replace_forward=True)

    def configure_optimizers(self):
        """Configures and defines the optimizer for the task. Defaults to adam
        with lr=0.001; It can be overwritten by either passing arguments from the
        command line, or by overwriting this entire method.

        Returns
        ---------
           Optimizer and learning rate scheduler
           (not implemented yet). : Tuple[torch.optim.Adam, None]

        """
        assert self.hparams.opt in [
            "adam",
            "sgd",
        ], f"Optimizer {self.hparams.opt} not supported."
        if self.hparams.opt == "adam":
            opt = torch.optim.Adam(self.modules.parameters(), self.hparams.lr)
        elif self.hparams.opt == "sgd":
            opt = torch.optim.SGD(self.modules.parameters(), self.hparams.lr)
        return opt, None  # None is for learning rate sched

    def __call__(self, *x, **xv):
        """Just forwards everything to the forward method."""
        return self.forward(*x, **xv)

    def on_train_start(self):
        """Initializes the optimizer, modules and puts the networks on the right
        devices. Optionally loads checkpoint if already present.

        This function gets executed at the beginning of every training.
        """
        self.experiment_folder = os.path.join(
            self.hparams.output_folder, self.hparams.experiment_name
        )
        if self.hparams.debug:
            self.experiment_folder = "tmp_" + get_random_string()
            logger.info(f"Created temporary folder for debug {self.experiment_folder}.")

        save_dir = os.path.join(self.experiment_folder, "save")
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
                        checkpoint_path=self.experiment_folder,
                    )

                    logger.info(f"Loaded existing checkpoint from {path}.")
            else:
                self.opt, self.lr_sched = self.configure_optimizers()
                self.start_epoch = 0

                self.checkpointer = Checkpointer(
                    "val_loss", checkpoint_path=self.experiment_folder
                )
        else:
            os.makedirs(self.experiment_folder, exist_ok=True)

            self.opt, self.lr_sched = self.configure_optimizers()
            self.start_epoch = 0

            self.checkpointer = Checkpointer(
                "val_loss", checkpoint_path=self.experiment_folder
            )

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
        """Runs at the end of each training. Cleans up before exiting."""
        if self.hparams.debug:
            logger.info(f"Removed temporary folder {self.experiment_folder}.")
            shutil.rmtree(self.experiment_folder)

        if self.accelerator.is_local_main_process:
            self.checkpointer.close()

    def train(
        self,
        epochs: int = 1,
        datasets: Dict = {},
        metrics: List[Metric] = [],
        debug: bool = False,
    ) -> None:
        """
        This method trains the model on the provided training dataset for the
        specified number of epochs. It tracks training metrics and can
        optionally perform validation during training, if the validation set is
        provided.

        Arguments
        ---------
        epochs : int
            The number of training epochs.
        datasets : Dict
            A dictionary of dataset loaders. Dataloader should be mapped to keys
            "train", "val", and "test".
        metrics : Optional[List[Metric]]
            A list of metrics to track during training. Default is an empty list.
        debug : bool
            Whether to run in debug mode. Default is False. If in debug mode,
            only runs for few epochs
            and with few batches.

        """
        self.datasets = datasets
        self.metrics = metrics
        assert "train" in self.datasets, "Training dataloader was not specified."
        assert epochs > 0, "You must specify at least one epoch."

        self.debug = debug

        self.on_train_start()

        if self.accelerator.is_local_main_process:
            logger.info(
                f"Starting from epoch {self.start_epoch}."
                + f" Training is scheduled for {epochs} epochs."
            )
        with self.accelerator.autocast():
            for e in range(self.start_epoch, epochs):
                pbar = tqdm(
                    self.datasets["train"],
                    unit="batches",
                    ascii=True,
                    dynamic_ncols=True,
                    disable=not self.accelerator.is_local_main_process,
                )
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

                    running_train.update({"train_loss": loss_epoch / (idx + 1)})

                    loss_epoch += loss.item()
                    pbar.set_postfix(**running_train)

                    if self.debug and idx > 10:
                        break

                pbar.close()

                train_metrics = {
                    "train_" + m.name: m.reduce(Stage.train, True) for m in self.metrics
                }
                train_metrics.update({"train_loss": loss_epoch / (idx + 1)})

                if "val" in datasets:
                    val_metrics = self.validate()
                    if self.accelerator.is_local_main_process:
                        self.checkpointer(
                            self,
                            e,
                            train_metrics,
                            val_metrics,
                            lambda x: self.accelerator.unwrap_model(x),
                        )
                else:
                    val_metrics = train_metrics.update(
                        {"val_loss": loss_epoch / (idx + 1)}
                    )

                if e >= 1 and self.debug:
                    break

        self.on_train_end()
        return None

    @torch.no_grad()
    def validate(self) -> Dict:
        """Runs the validation step."""
        assert "val" in self.datasets, "Validation dataloader was not specified."
        self.modules.eval()

        pbar = tqdm(
            self.datasets["val"],
            unit="batches",
            ascii=True,
            dynamic_ncols=True,
            disable=not self.accelerator.is_local_main_process,
        )
        loss_epoch = 0
        pbar.set_description("Validation...")
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
                pbar.set_postfix(loss=loss_epoch / (idx + 1))

                if self.debug and idx > 10:
                    break

        val_metrics = {"val_" + m.name: m.reduce(Stage.val, True) for m in self.metrics}
        val_metrics.update({"val_loss": loss_epoch / (idx + 1)})

        pbar.close()

        return val_metrics

    @torch.no_grad()
    def test(self, datasets: Dict = {}) -> None:
        """Runs the test steps."""
        assert "test" in self.datasets, "Test dataloader was not specified."
        self.modules.eval()

        pbar = tqdm(
            self.datasets["test"],
            unit="batches",
            ascii=True,
            dynamic_ncols=True,
            disable=not self.accelerator.is_local_main_process,
        )
        loss_epoch = 0
        pbar.set_description("Testing...")
        with self.accelerator.autocast():
            for idx, batch in enumerate(pbar):
                if isinstance(batch, list):
                    batch = [b.to(self.device) for b in batch]
                self.opt.zero_grad()

                model_out = self(batch)
                loss = self.compute_loss(model_out, batch)
                for m in self.metrics:
                    m(model_out, batch, Stage.test, self.device)

                loss_epoch += loss.item()
                pbar.set_postfix(loss=loss_epoch / (idx + 1))

        pbar.close()

        test_metrics = {
            "test_" + m.name: m.reduce(Stage.test, True) for m in self.metrics
        }
        test_metrics.update({"test_loss": loss_epoch / (idx + 1)})
        s_out = (
            "Testing "
            + " - ".join([f"{k}: {v:.2f}" for k, v in test_metrics.items()])
            + "; "
        )

        logger.info(s_out)

        return None
