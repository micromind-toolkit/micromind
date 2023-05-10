import logging
import os
import torch
import torch.nn as nn
import torchinfo
import sys
import numpy as np
from recipes.image_classification.classification import (
    optimized_params,
    train_one_epoch,
    validate,
)
from naswot import score_network
from orion.client import build_experiment, get_experiment
from micromind import PhiNet, configlib
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.data import create_loader, create_transform, create_dataset
from utils import plot, plot_table

group = configlib.add_parser("Orion experiments")

group.add_argument("--w", default=-0.07, type=float)
group.add_argument("--algo", default="random", type=str)
group.add_argument("--dset", default="cifar10", type=str)
group.add_argument("--save_path", default="./orion_exp/", type=str)
group.add_argument("--classifier", default=None, type=bool)
group.add_argument("--device", default=0, type=int)
group.add_argument("--target", default=5e6, type=int)
group.add_argument("--opt-goal", default="params", type=str)


def data_loader(image_size, batch_size):
    if nas_args.dset == "cifar10":
        # mean and std of cifar10 dataset
        data_mean = (0.49139968, 0.48215827, 0.44653124)
        data_std = (0.24703233, 0.24348505, 0.26158768)

    elif nas_args.dset == "cifar100":
        # mean and std of cifar100 dataset
        data_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        data_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    train_transforms = create_transform(
        input_size=image_size,
        is_training=True,
        mean=data_mean,
        std=data_std,
        auto_augment="rand-m3-mstd0.55",
    )

    eval_transforms = create_transform(
        input_size=image_size, mean=data_mean, std=data_std
    )

    train_dataset = create_dataset(
        "torch/" + nas_args.dset,
        "~/data/" + nas_args.dset,
        download=True,
        split="train",
        transform=train_transforms,
    )
    eval_dataset = create_dataset(
        "torch/" + nas_args.dset,
        "~/data/" + nas_args.dset,
        download=True,
        split="val",
        transform=eval_transforms,
    )

    loader_train = create_loader(
        train_dataset,
        input_size=image_size,
        batch_size=batch_size,
        is_training=True,
        # num_aug_splits=num_aug_splits,
        # interpolation='random',
        mean=data_mean,
        std=data_std,
        num_workers=0,
        distributed=False,
        # collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False,
    )

    loader_eval = create_loader(
        eval_dataset,
        input_size=image_size,
        batch_size=batch_size,
        is_training=False,
        # use_prefetcher=arg.prefetcher,
        # use_prefetcher=nas_args.prefetcher,
        # interpolation=random,
        mean=data_mean,
        std=data_std,
        num_workers=0,
        distributed=False,
        # crop_pct=data_config['crop_pct'],
        pin_memory=False,
        persistent_workers=False,
    )

    return loader_train, loader_eval


def train_and_evaluate(model, epochs, loader_train, loader_eval):
    nas_args.opt = "sgd"
    nas_args.lr = 0.05
    nas_args.weight_decay = 0.02
    nas_args.prefetcher = not nas_args.no_prefetcher
    nas_args.distributed = False
    nas_args.device = "cuda:0"
    nas_args.world_size = 1
    nas_args.rank = 0  # global rank
    nas_args.epochs = epochs
    nas_args.warmup_epochs = 0
    nas_args.warmup_lr = None
    nas_args.cooldown_epochs = 0

    train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = train_loss_fn
    optimizer = create_optimizer(nas_args, model)
    lr_scheduler, num_epochs = create_scheduler(nas_args, optimizer)
    start_epoch = 0
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    logging.info("Scheduled epochs: {}".format(num_epochs))

    best_metric = None
    best_epoch = None
    saver = None

    output_dir = os.path.join(
        nas_args.save_path, nas_args.dset + "/", nas_args.algo + "/"
    )

    try:
        for epoch in range(start_epoch, num_epochs):
            train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                nas_args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
            )

            eval_metrics = validate(model, loader_eval, validate_loss_fn, nas_args)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        logging.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))
    return eval_metrics["loss"]


def EA_objective(loss, params, target, w):
    obj = -1 * loss * (params / nas_args.target) ** w
    # obj = loss - 100 * np.abs(nas_args.target - params)
    return obj


def score_objective(score, params, target, w):
    # obj = score * (params / nas_args.target) ** w
    # obj = score - (np.abs(params - nas_args.target)*1E-6)
    obj = score * 1e-3 - np.abs(params - nas_args.target) * 1e-6
    # obj= score + ((nas_args.target - params)**2 * 1e-3)
    # obj = score - 100 * np.abs(nas_args.target - params)
    return obj


def define_model(alpha, beta, B0, t_zero, res, batch_size):
    batch_size = batch_size
    model = PhiNet(
        input_shape=[3, res, res],
        alpha=alpha,
        beta=beta,
        t_zero=t_zero,
        num_layers=B0,
        squeeze_excite=True,
        h_swish=False,
        include_top=True,
        num_classes=nas_args.num_classes,
    )
    model.cuda()
    return model


def get_params(model, res):
    return torchinfo.summary(
        model, (1, 3, res, res), device=nas_args.device, verbose=0
    ).total_params


def objective(alpha, beta, B0, t_zero, res, epochs):
    batch_size = 32
    w = nas_args.w
    model = define_model(alpha, beta, B0, t_zero, res, batch_size)

    loader_train, loader_eval = data_loader(
        image_size=(3, res, res), batch_size=batch_size
    )
    logging.info(
        "Phinet params: %s",
        {"alpha": alpha, "beta": beta, "t_zero": t_zero, "B0": B0, "res": res},
    )

    params = get_params(model, res)

    if nas_args.algo == "evo":
        loss = train_and_evaluate(model, epochs, loader_train, loader_eval)
        obj = EA_objective(loss, params, nas_args.target, w)
    else:
        score = score_network(model, batch_size, loader_train)
        obj = score_objective(score, params, nas_args.target, w)

    # obj = -1 * score - phi(params - nas_args.target)**w
    # target_obj = (params / nas_args.target) ** w
    # target_obj = 100 * np.abs(nas_args.target - params)
    target_obj = np.abs(params - nas_args.target) * 1e-6
    print(target_obj)

    logging.info("w: %f", w)
    logging.info("Total params: %f", params / 1e6)
    if nas_args.algo == "evo":
        logging.info("Loss: %f", loss)
    else:
        logging.info("Score: %f", score)
    logging.info("Objective: %f", -obj)

    objs.append(-obj)
    param.append(params / 1e6)
    if nas_args.algo == "evo":
        losses.append(loss)
    else:
        scores.append(score)
    target_objs.append(target_obj)

    return [{"name": "objective", "type": "objective", "value": -obj}]


def run_hpo(exp_name):
    # Specify the database where the experiments are stored
    storage = {
        "type": "legacy",
        "database": {
            "type": "pickleddb",
            "host": "./orion_exp/exp_database.pkl",
        },
    }

    # Load the data for the specified experiment
    if nas_args.algo == "random":
        algorithm = {
            "random": {
                "seed": 1,
            },
        }
    elif nas_args.algo == "tpe":
        algorithm = {
            "tpe": {
                "seed": 1,
                "n_initial_points": 20,
                "n_ei_candidates": 25,
                "gamma": 0.25,
                "equal_weight": False,
                "prior_weight": 1.0,
                "full_weight_num": 25,
                "parallel_strategy": {
                    "of_type": "StatusBasedParallelStrategy",
                    "strategy_configs": {
                        "broken": {
                            "of_type": "MaxParallelStrategy",
                            "default_result": 100,
                        },
                    },
                    "default_strategy": {
                        "of_type": "meanparallelstrategy",
                        "default_result": 50,
                    },
                },
            },
        }
    elif nas_args.algo == "evo":
        algorithm = {
            "EvolutionES": {
                "seed": 1,
                "repetitions": 1,
                "nums_population": 10,
                "mutate": {
                    "function": "orion.algo.mutate_functions.default_mutate",
                    "multiply_factor": 3.0,
                    "add_factor": 1,
                },
            }
        }

    experiment = build_experiment(
        exp_name,
        version=1,
        # space={
        #     "alpha": "loguniform(3, 4)",
        #     "beta": "loguniform(0.75, 1.3)",
        #     "B0": "uniform(7, 8, discrete=True)",
        #     "t_zero": "uniform(2, 6, discrete=True)",
        #     "res": "choices([32,64,128,256])",
        #     "epochs": "fidelity(1, 10, base=2)",
        # },
        space={
            "alpha": "loguniform(1, 5)",
            "beta": "loguniform(0.5, 1.3)",
            "B0": "uniform(5, 8, discrete=True)",
            "t_zero": "uniform(2, 6, discrete=True)",
            "res": "choices([32,64,128,160,240,256])",
            "epochs": "fidelity(1, 10, base=2)",
        },
        algorithms=algorithm,
        storage=storage,
        max_trials=100,
    )

    trials = 1
    while not experiment.is_done:
        print("trial", trials)
        trial = experiment.suggest()
        if trial is None and experiment.is_done:
            break
        obj = objective(**trial.params)
        experiment.observe(trial, obj)
        trials += 1
    return experiment


if __name__ == "__main__":
    nas_args = configlib.parse()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(nas_args.device)

    objs = []
    param = []
    scores = []
    target_objs = []
    losses = []

    if nas_args.dset == "cifar10":
        nas_args.num_classes = 10

    if nas_args.dset == "cifar100":
        nas_args.num_classes = 100

    exp_name = (
        "obj1_seed_1_"
        + nas_args.algo
        + "_w_"
        + str(nas_args.w)
        + "_"
        + str(int(nas_args.target / 1e6))
        + "M"
    )
    # exp_name = "tcdcdv"
    base_path = os.path.join(nas_args.save_path, nas_args.dset, nas_args.algo, exp_name)
    os.makedirs(
        base_path,
        exist_ok=True,
    )

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = "%(message)s"
    logger = logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )

    fh = logging.FileHandler(os.path.join(base_path + "_log.txt"), "w")

    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    experiment = run_hpo(exp_name)
    experiment = get_experiment(name=exp_name)
    best_trial = experiment.stats.best_trials_id
    best_params = experiment.get_trial(uid=best_trial).params
    best_obj = experiment.stats.best_evaluation
    logging.info("Best params: {}".format(best_params))
    logging.info("Best obj value: {:.3f}".format(best_obj))

    model = PhiNet(
        input_shape=[3, best_params["res"], best_params["res"]],
        alpha=best_params["alpha"],
        beta=best_params["beta"],
        t_zero=best_params["t_zero"],
        num_layers=best_params["B0"],
        squeeze_excite=True,
        h_swish=False,
        include_top=True,
        num_classes=nas_args.num_classes,
    )
    model.cuda()
    total_param = torchinfo.summary(
        model,
        (1, 3, best_params["res"], best_params["res"]),
        device=nas_args.device,
        verbose=0,
    ).total_params
    logging.info("Best trial total params: {:.3f}".format(total_param / 1e6))

    plot_table(exp_name, param, scores, target_objs, base_path)

    plot(param, objs, base_path, "objective")
    if nas_args.algo == "evo":
        plot(param, losses, base_path, "loss")
    else:
        plot(param, scores, base_path, "naswot")
    plot(param, target_objs, base_path, "target")

    save_path = base_path
    print(nas_args.classifier)
    # nas_args.classifier=False
    if nas_args.classifier or nas_args.algo == "evo":
        print(nas_args.classifier)
        optimized_params(nas_args, best_params, nas_args.num_classes, save_path)
