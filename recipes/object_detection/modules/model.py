# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
from pathlib import Path
from typing import Union

from ultralytics import YOLO
from ultralytics.nn.tasks import (attempt_load_one_weight, guess_model_task, yaml_model_load)
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, callbacks,
                                    yaml_load)
from ultralytics.yolo.utils.checks import check_file, check_pip_update_available, check_yaml
from ultralytics.yolo.utils.downloads import GITHUB_ASSET_STEMS

from ultralytics.yolo.utils import DEFAULT_CFG, RANK
from modules.detectionheadmodule import DetectionHeadModel
from modules.detectionheadtrainer import DetectionHeadTrainer

class microYOLO(YOLO):
    """
    YOLO (You Only Look Once) object detection model.

    Args:
        model (str, Path): Path to the model file to load or create.
        task (Any, optional): Task type for the YOLO model. Defaults to None.

    Attributes:
        predictor (Any): The predictor object.
        model (Any): The model object.
        trainer (Any): The trainer object.
        task (str): The type of model task.
        ckpt (Any): The checkpoint object if the model loaded from *.pt file.
        cfg (str): The model configuration if loaded from *.yaml file.
        ckpt_path (str): The checkpoint file path.
        overrides (dict): Overrides for the trainer object.
        metrics (Any): The data for metrics.

    Methods:
        __call__(source=None, stream=False, **kwargs):
            Alias for the predict method.
        _new(cfg:str, verbose:bool=True) -> None:
            Initializes a new model and infers the task type from the model definitions.
        _load(weights:str, task:str='') -> None:
            Initializes a new model and infers the task type from the model head.
        _check_is_pytorch_model() -> None:
            Raises TypeError if the model is not a PyTorch model.
        reset() -> None:
            Resets the model modules.
        info(verbose:bool=False) -> None:
            Logs the model info.
        fuse() -> None:
            Fuses the model for faster inference.
        predict(source=None, stream=False, **kwargs) -> List[ultralytics.yolo.engine.results.Results]:
            Performs prediction using the YOLO model.

    Returns:
        list(ultralytics.yolo.engine.results.Results): The prediction results.
    """

    def __init__(self, model: Union[str, Path] = 'yolov8n.yaml', task=None) -> None:
        """
        Initializes the YOLO model.

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
            task (Any, optional): Task type for the YOLO model. Defaults to None.
        """
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.task = None  # task type
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        model = str(model).strip()  # strip spaces        

        # Load or create new YOLO model
        suffix = Path(model).suffix
        if suffix == '.yaml': # new
            self._new(model, task)
        else:
            self._load(model, task)

    def _new(self, cfg: str, task=None, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str) or (None): model task
            verbose (bool): display model info on load
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = DetectionHeadModel(cfg=cfg, nc=cfg_dict["nc"], verbose=verbose and RANK == -1) ## code added
        self.overrides['model'] = self.cfg

        # Below added to allow export from yamls
        args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine model and default args, preferring model args
        self.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
        self.model.task = self.task

    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str) or (None): model task
        """
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args['task']
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides['model'] = weights
        self.overrides['task'] = self.task


    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides['mode'] = 'export'
        if overrides.get('imgsz') is None:
            overrides['imgsz'] = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        if overrides.get('batch') is None:
            overrides['batch'] = 1  # default to 1 if not modified
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        self._check_is_pytorch_model()
        if self.session:  # Ultralytics HUB session
            if any(kwargs):
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
        check_pip_update_available()
        overrides = self.overrides.copy()
        if kwargs.get('cfg'):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs['cfg']))
        overrides.update(kwargs)
        overrides['mode'] = 'train'
        if not overrides.get('data'):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get('resume'):
            overrides['resume'] = self.ckpt_path
        self.task = overrides.get('task') or self.task
        self.trainer = DetectionHeadTrainer(overrides=overrides, _callbacks=self.callbacks)
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP
