import logging
import os
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torchinfo
import sys
import numpy as np
sys.path.append("../")
from image_classification.classification import optimized_params
from naswot import score_network
from orion.client import build_experiment, get_experiment
from micromind import PhiNet, configlib
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.utils import AverageMeter, accuracy


group = configlib.add_parser("Orion experiments")

group.add_argument("--w", default=-0.07, type=float)
group.add_argument("--algo", default="random", type=str)
group.add_argument("--dset", default="cifar10", type=str)
group.add_argument("--save_path", default="./orion_exp/", type=str)
group.add_argument("--classifier", default=None, type=bool)
group.add_argument("--device", default=0, type=int)
group.add_argument("--target", default=5e6, type=int)
group.add_argument("--opt-goal", default="params", type=str)


def create_loader(image_size, batch_size, data_mean, data_std):
    train_transforms = timm.data.create_transform(
        input_size=image_size,
        is_training=True,
        mean=data_mean,
        std=data_std,
        auto_augment="rand-m3-mstd0.55",
    )

    eval_transforms = timm.data.create_transform(
        input_size=image_size, mean=data_mean, std=data_std
    )

    train_dataset = timm.data.create_dataset(
        "torch/" + nas_args.dset,
        "~/data/" + nas_args.dset,
        download=True,
        split="train",
        transform=train_transforms,
    )
    eval_dataset = timm.data.create_dataset(
        "torch/" + nas_args.dset,
        "~/data/" + nas_args.dset,
        download=True,
        split="val",
        transform=eval_transforms,
    )

    loader_train = timm.data.create_loader(
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

    loader_eval = timm.data.create_loader(
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
    model_ema = None
    use_amp = False

    train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = train_loss_fn

    # optimizer = create_optimizer(model)

    optimizer = create_optimizer_v2(model, opt="cosine", lr=0.005, weight_decay=0.02)

    # lr_scheduler, num_epochs = create_scheduler(
    #     optimizer, sched="cosine", num_epochs=epochs
    # )
    nas_args.epochs = epochs
    nas_args.warmup_epochs = 0
    nas_args.warmup_lr = None
    nas_args.cooldown_epochs = 0
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
    # output_dir = get_outdir(output_dir, exp_name)

    # saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=decreasing)

    try:
        for epoch in range(start_epoch, num_epochs):
            train_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                use_amp=use_amp,
                model_ema=model_ema,
            )

            eval_metrics = validate(
                model,
                loader_eval,
                validate_loss_fn,
            )

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        logging.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))
    return eval_metrics["loss"]


def train_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    lr_scheduler=None,
    saver=None,
    output_dir="",
    use_amp=False,
    model_ema=None,
):
    log_interval = 200
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        input, target = input.cuda(), target.cuda()

        output = model(input)

        loss = loss_fn(output, target)
        losses_m.update(loss.item(), input.size(0))
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            # world_size = torch.distributed.get_world_size()

            logging.info(
                "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                "Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  "
                "Time: {batch_time.val:.3f}s  "
                "({batch_time.avg:.3f}s)  "
                "LR: {lr:.3e}  "
                "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                    epoch,
                    batch_idx,
                    len(loader),
                    100.0 * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    lr=lr,
                    data_time=data_time_m,
                )
            )

        end = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(model, loader, loss_fn, log_suffix=""):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()
    log_interval = 50
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))

            batch_time_m.update(time.time() - end)

            if last_batch or batch_idx % log_interval == 0:
                log_name = "Test" + log_suffix
                logging.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=prec1_m,
                        top5=prec5_m,
                    )
                )
            end = time.time()

    metrics = OrderedDict(
        [("loss", losses_m.avg), ("prec1", prec1_m.avg), ("prec5", prec5_m.avg)]
    )
    return metrics


def objective(alpha, beta, B0, t_zero, res, epochs):
    batch_size = 32
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

    if nas_args.dset == "cifar10":
        # mean and std of cifar10 dataset
        TRAIN_MEAN = (0.49139968, 0.48215827, 0.44653124)
        TRAIN_STD = (0.24703233, 0.24348505, 0.26158768)

    elif nas_args.dset == "cifar100":
        # mean and std of cifar100 dataset
        TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    loader_train, loader_eval = create_loader(
        image_size=(3, res, res),
        batch_size=batch_size,
        data_mean=TRAIN_MEAN,
        data_std=TRAIN_STD,
    )
    logging.info(
        "Phinet params: %s",
        {"alpha": alpha, "beta": beta, "t_zero": t_zero, "B0": B0, "res": res},
    )
    w = nas_args.w
    params = torchinfo.summary(
        model, (1, 3, res, res), device=nas_args.device, verbose=0
    ).total_params
    if nas_args.algo == "evo":
        loss = train_and_evaluate(model, epochs, loader_train, loader_eval)
        #obj = -1 * loss * (params / nas_args.target) ** w
        #obj = loss - 100 * np.abs(nas_args.target - params)
    else:
        score = score_network(model, batch_size, loader_train)
        obj = score * (params / nas_args.target) ** w
        #obj = score - (np.abs(params - nas_args.target)*1E-6)
        #obj = score*1e-3 - np.abs(params - nas_args.target)*1e-6 
        #obj= score + ((nas_args.target - params)**2 * 1e-3)
        #obj = score - 100 * np.abs(nas_args.target - params)
    # obj = -1 * score - phi(params - nas_args.target)**w
    #target_obj = (params / nas_args.target) ** w
    #target_obj = 100 * np.abs(nas_args.target - params)
    target_obj = np.abs(params - nas_args.target)*1E-6
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


def plot(x, y, path, plot_type):
    path = os.path.join(path + "_" + plot_type + ".jpg")
    plt.scatter(x, y, color="blue", alpha=0.5)
    if plot_type == "objective":
        plt.ylabel("Objective")
    elif plot_type == "naswot":
        plt.ylabel("NASWOT Score")
    elif plot_type == "loss":
        plt.ylabel("Val loss")
    elif plot_type == "target":
        plt.ylabel("Target Objective")
    else:
        "This type of plot is not defined!"
    plt.savefig(path)
    plt.cla()


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

    exp_name = "obj1_seed_1_" + nas_args.algo + "_w_" + str(nas_args.w) + "_" + str(int(nas_args.target/1e6)) + "M_cosine"
    #exp_name = "test_obj_1"
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
        input_shape=[3, best_params['res'], best_params['res']],
        alpha=best_params['alpha'],
        beta=best_params['beta'],
        t_zero=best_params['t_zero'],
        num_layers=best_params['B0'],
        squeeze_excite=True,
        h_swish=False,
        include_top=True,
        num_classes=nas_args.num_classes,
    )
    model.cuda()
    total_param = torchinfo.summary(
        model, (1, 3, best_params['res'], best_params['res']), device=nas_args.device, verbose=0
    ).total_params
    logging.info("Best trial total params: {:.3f}".format(total_param/1e6))

    df = experiment.to_pandas(True)
    print(df)
    df.drop(df.iloc[:, 0:6], inplace=True, axis=1)
    df["Parameters"] = param
    df["Score"] = scores
    df["Target Objective"] = target_objs
    df.to_excel(base_path + ".xlsx", index=False)

    plot(param, objs, base_path, "objective")
    if nas_args.algo == "evo":
       plot(param, losses, base_path, "loss")
    else:
       plot(param, scores, base_path, "naswot")
    plot(param, target_objs, base_path, "target")

    save_path = base_path
    print(nas_args.classifier)
    #nas_args.classifier=False
    if nas_args.classifier or nas_args.algo == "evo":
        print(nas_args.classifier)
        optimized_params(nas_args, best_params, nas_args.num_classes, save_path)
