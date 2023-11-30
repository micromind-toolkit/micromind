"""
Core class for micromind. Supports helper function for exports. Out-of-the-box
multi-gpu and FP16 training with HF Accelerate and much more.

Authors:
    - Francesco Paissan, 2023
"""
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from tqdm import tqdm
import warnings

from .utils.helpers import get_logger

logger = get_logger()

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

    def __init__(
        self,
        name: str,
        fn: Callable,
        reduction: Optional[str] = "mean",
        eval_only: Optional[bool] = False,
        eval_period: Optional[int] = 1,
    ):
        self.name = name
        self.fn = fn
        self.reduction = reduction
        self.eval_only = eval_only
        self.eval_period = eval_period

        self.history = {s: [] for s in [Stage.train, Stage.val, Stage.test]}

    def __call__(self, pred, batch, stage, device="cpu"):
        dat = self.fn(pred, batch)
        if dat.ndim == 0:
            dat = dat.unsqueeze(0)

        self.history[stage].append(dat)

    def reduce(self, stage, clear=False):
        """
        Compute and return the metric for a given prediction and batch data.

        Arguments
        ---------
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
            tmp = torch.cat(self.history[stage], dim=0).mean()
        elif self.reduction == "sum":
            tmp = torch.cat(self.history[stage], dim=0).sum()

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

        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.current_epoch = 0

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
        Scheduler step is called every optimization step.

        Returns
        -------
        Optimizer and learning rate scheduler.
            : Union[Tuple[torch.optim.Adam, None], torch.optim.Adam]

        """
        assert self.hparams.opt in [
            "adam",
            "sgd",
        ], f"Optimizer {self.hparams.opt} not supported."
        if self.hparams.opt == "adam":
            opt = torch.optim.Adam(self.modules.parameters(), self.hparams.lr)
        elif self.hparams.opt == "sgd":
            opt = torch.optim.SGD(self.modules.parameters(), self.hparams.lr)

        return opt

    def __call__(self, *x, **xv):
        """Just forwards everything to the forward method."""
        return self.forward(*x, **xv)

    def on_train_start(self):
        """Initializes the optimizer, modules and puts the networks on the right
        devices. Optionally loads checkpoint if already present.

        This function gets executed at the beginning of every training.
        """

        # pass debug status to checkpointer
        self.checkpointer.debug = self.hparams.debug

        init_opt = self.configure_optimizers()
        if isinstance(init_opt, list) or isinstance(init_opt, tuple):
            self.opt, self.lr_sched = init_opt
        else:
            self.opt = init_opt

        self.init_devices()

        self.start_epoch = 0
        if self.checkpointer is not None:
            # recover state
            ckpt = self.checkpointer.recover_state()
            if ckpt is not None:
                accelerate_path, self.start_epoch = ckpt
                self.accelerator.load_state(accelerate_path)
        else:
            tmp = """
                You are not passing a checkpointer to the training function, \
                thus no status will be saved. If this is not the intended behaviour \
                please check https://micromind-toolkit.github.io/docs/").
            """
            warnings.warn(" ".join(tmp.split()))

    def init_devices(self):
        """Initializes the data pipeline and modules for DDP and accelerated inference.
        To control the device selection, use `accelerate config`."""

        # pass each module through DDP independently
        convert = list(self.modules.values())
        if hasattr(self, "opt"):
            convert += [self.opt]

        if hasattr(self, "lr_sched"):
            convert += [self.lr_sched]

        if hasattr(self, "datasets"):
            # if the datasets are store here, prepare them for DDP
            convert += list(self.datasets.values())

        accelerated = self.accelerator.prepare(*convert)
        for idx, key in enumerate(self.modules):
            print(accelerated[idx])
            self.modules[key] = accelerated[idx]
        self.accelerator.register_for_checkpointing(self.modules)

        if hasattr(self, "opt"):
            self.opt = accelerated[1]
            self.accelerator.register_for_checkpointing(self.opt)

        if hasattr(self, "lr_sched"):
            self.lr_sched = accelerated[2]
            self.accelerator.register_for_checkpointing(self.lr_sched)

        if hasattr(self, "datasets"):
            for i, key in enumerate(list(self.datasets.keys())[::-1]):
                self.datasets[key] = accelerated[-(i + 1)]

        self.modules.to(self.device)

    def on_train_end(self):
        """Runs at the end of each training. Cleans up before exiting."""
        pass

    def eval(self):
        self.modules.eval()

    def train(
        self,
        epochs: int = 1,
        datasets: Dict = {},
        metrics: List[Metric] = [],
        checkpointer=None,  # fix type hints
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
        self.checkpointer = checkpointer
        assert "train" in self.datasets, "Training dataloader was not specified."
        assert epochs > 0, "You must specify at least one epoch."

        self.debug = debug

        self.on_train_start()

        if self.accelerator.is_local_main_process:
            logger.info(
                f"Starting from epoch {self.start_epoch + 1}."
                + f" Training is scheduled for {epochs} epochs."
            )
        with self.accelerator.autocast():
            for e in range(self.start_epoch + 1, epochs + 1):
                self.current_epoch = e
                pbar = tqdm(
                    self.datasets["train"],
                    unit="batches",
                    ascii=True,
                    dynamic_ncols=True,
                    disable=not self.accelerator.is_local_main_process,
                )
                loss_epoch = 0
                pbar.set_description(f"Running epoch {self.current_epoch}/{epochs}")
                self.modules.train()
                for idx, batch in enumerate(pbar):
                    if isinstance(batch, list):
                        batch = [b.to(self.device) for b in batch]

                    self.opt.zero_grad()

                    model_out = self(batch)
                    loss = self.compute_loss(model_out, batch)
                    loss_epoch += loss.item()

                    self.accelerator.backward(loss)
                    self.opt.step()
                    if hasattr(self, "lr_sched"):
                        # ok for cos_lr
                        self.lr_sched.step()

                    for m in self.metrics:
                        if (
                            self.current_epoch + 1
                        ) % m.eval_period == 0 and not m.eval_only:
                            m(model_out, batch, Stage.train, self.device)

                    running_train = {}
                    for m in self.metrics:
                        if (
                            self.current_epoch + 1
                        ) % m.eval_period == 0 and not m.eval_only:
                            running_train["train_" + m.name] = m.reduce(Stage.train)

                    running_train.update({"train_loss": loss_epoch / (idx + 1)})

                    pbar.set_postfix(**running_train)

                    if self.debug and idx > 10:
                        break

                pbar.close()

                train_metrics = {}
                for m in self.metrics:
                    if (
                        self.current_epoch + 1
                    ) % m.eval_period == 0 and not m.eval_only:
                        train_metrics["train_" + m.name] = m.reduce(Stage.train, True)

                train_metrics.update({"train_loss": loss_epoch / (idx + 1)})

                if "val" in datasets:
                    val_metrics = self.validate()
                    if (
                        self.accelerator.is_local_main_process
                        and self.checkpointer is not None
                    ):
                        self.checkpointer(
                            self,
                            train_metrics,
                            val_metrics,
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
                    if (self.current_epoch + 1) % m.eval_period == 0:
                        m(model_out, batch, Stage.val, self.device)

                loss_epoch += loss.item()
                pbar.set_postfix(loss=loss_epoch / (idx + 1))

                if self.debug and idx > 10:
                    break

        val_metrics = {}
        for m in self.metrics:
            if (self.current_epoch + 1) % m.eval_period == 0:
                val_metrics["val_" + m.name] = m.reduce(Stage.val, True)

        val_metrics.update({"val_loss": loss_epoch / (idx + 1)})

        pbar.close()

        return val_metrics

    @torch.no_grad()
    def test(self, datasets: Dict = {}, metrics: List[Metric] = []) -> None:
        """Runs the test steps.

        Arguments
        ---------
        datasets : Dict
            Dictionary with the test DataLoader. Should be present in the key
            `test`.
        metrics : List[Metric]
            List of metrics to compute during test step.

        Returns
        -------
        Metrics computed on test set. : Dict[torch.Tensor]
        """
        assert "test" in datasets, "Test dataloader was not specified."
        self.modules.eval()

        pbar = tqdm(
            datasets["test"],
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

                model_out = self(batch)
                loss = self.compute_loss(model_out, batch)
                for m in metrics:
                    m(model_out, batch, Stage.test, self.device)

                loss_epoch += loss.item()
                pbar.set_postfix(loss=loss_epoch / (idx + 1))

        pbar.close()

        test_metrics = {"test_" + m.name: m.reduce(Stage.test, True) for m in metrics}
        test_metrics.update({"test_loss": loss_epoch / (idx + 1)})
        s_out = (
            "Testing "
            + " - ".join([f"{k}: {v:.2f}" for k, v in test_metrics.items()])
            + "; "
        )

        logger.info(s_out)

        return test_metrics
