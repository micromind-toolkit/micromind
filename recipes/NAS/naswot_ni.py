"""
Hyperparamater Optimization of Phinets using Orion

Authors:
    - Mariam Jamal, 2023
"""
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
from naswot_naslib import compute_nwot
from epe_nas import compute_epe_score
from synflow import compute_synflow_per_weight, sum_arr
from logsynflow import compute_synflow_per_weight as compute_logsynflow
from orion.client import build_experiment, get_experiment
from micromind import PhiNet, configlib
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.data import create_loader, create_transform, create_dataset
from utils import plot, plot_table, topk_net, get_mod_data, add_noise, plot_pareto
import multiprocessing as mp

group = configlib.add_parser("Orion experiments")

group.add_argument("--w", default=-0.07, type=float)
group.add_argument(
    "--algo", default="random", type=str, help="Algorithm for optimization"
)
group.add_argument("--dset", default="cifar100", type=str, help="Dataset")
group.add_argument(
    "--save_path", default="./orion_exp/", type=str, help="Path to save results"
)
group.add_argument(
    "--classifier",
    default=False,
    type=bool,
    help="True to train the network with found hyperparameters",
)
group.add_argument("--device", default=0, type=int, help="GPU/CPU")
group.add_argument(
    "--target", default=1e6, type=int, help="Target value of the metric (default: 1M)"
)
group.add_argument(
    "--opt-goal",
    default="params",
    type=str,
    help="Metric to optimize (default: params)",
)
group.add_argument(
    "--k", default=1, type=int, help="No. of top k networks to train (default: 1)"
)
group.add_argument(
    "--predictor",
    default="naswot",
    type=str,
    help="Zero-cost proxy to use (default: naswot)",
)
group.add_argument(
    "--diff_weight",
    default=1.,
    type=float,
    help="Weight for diff term",
)


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

    # TODO: change this fixed image res and batch size for now
    loader_train, loader_eval = data_loader(image_size=64, batch_size=64)
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
    return eval_metrics["top1"]


def EA_objective(top1, params, target, w):
    # obj = -1 * loss * (params / nas_args.target) ** w
    power = np.floor(np.log10(nas_args.target))
    obj = top1 * 1e-3 - (np.abs(params - nas_args.target) / 10**power)
    # obj = loss - 100 * np.abs(nas_args.target - params)
    return obj


def score_objective(score, param, target, w):
    # obj = score * (params / nas_args.target) ** w
    # obj = score - (np.abs(params - nas_args.target)*1E-6)
    #power = np.floor(np.log10(nas_args.target))
   
    #alpha = 1e-6 if nas_args.predictor == "synflow" else 1e-3
    #obj = score * alpha - (np.abs(params - nas_args.target) / 10**power)
   # diff = (np.abs(params - nas_args.target))
    #diff = normalize_score(diff, noise=False, score_2=True, synflow=False) 
    #logging.info("Diff :%f",diff)
    #beta = nas_args.diff_weight
    #obj = score  - (beta * (diff / 10**4))
    w1 = 0.8
    w2 = 0.2
    norm_score = normalize_score(score, noise=False, score_2=True, synflow=False)
    logging.info("Normalized Score: %f", norm_score)

    #norm_param = normalize_avg(param/target, 1, norm_params)
    #logging.info("Normalized Params: %f", norm_param)

    obj = (w1 * norm_score) - (w2 * param/target)
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

def normalize_avg(value, window_size, data_points):
    data_points.append(value)
    print(data_points)
    if len(data_points) == window_size:
        min_val = np.min(data_points)
        max_val = np.max(data_points)
    else:
        min_val = min(data_points)
        max_val = max(data_points)

    if min_val == max_val:
        return 0.0
    else:
        return (value - min_val) / (max_val - min_val)

def normalize_score(score, noise, score_2, synflow):
    # Update min and max scorese
    if noise:
        global min_ni, max_ni
        min_ni = min(min_ni, score)
        max_ni = max(max_ni, score)
        normalized_score = (score - min_ni) / (max_ni - min_ni)
    elif score_2:
        global min_score_2, max_score_2
        min_score_2 = min(min_score_2, score)
        max_score_2 = max(max_score_2, score)
        normalized_score = (score - min_score_2) / (max_score_2 - min_score_2)
    elif synflow:
        global min_synflow, max_synflow
        min_synflow = min(min_synflow, score)
        max_synflow = max(max_synflow, score)
        normalized_score = (score - min_synflow) / (max_synflow - min_synflow)
    else:
        global min_score, max_score
        min_score = min(min_score, score)
        max_score = max(max_score, score)
        normalized_score = (score - min_score) / (max_score - min_score)

    return normalized_score

def normalize_score_free_rea(score, noise, score_2, synflow):
    # Update min and max scorese
    if noise:
        global max_ni
        min_ni = 0
        max_ni = max(max_ni, score)
        normalized_score = (score - min_ni) / (max_ni - min_ni)
    elif score_2:
        global min_score_2, max_score_2
        min_score_2 = 0
        max_score_2 = max(max_score_2, score)
        normalized_score = (score - min_score_2) / (max_score_2 - min_score_2)
    elif synflow:
        global min_synflow, max_synflow
        min_synflow = 0
        max_synflow = max(max_synflow, score)
        normalized_score = (score - min_synflow) / (max_synflow - min_synflow)
    else:
        global min_score, max_score
        min_score = 0
        max_score = max(max_score, score)
        normalized_score = (score - min_score) / (max_score - min_score)
    logging.info("Normalized score: %f", normalized_score)
    return normalized_score


def objective(noise, inputs, inputs_noisy, targets, alpha, beta, B0, t_zero, res, epochs):
    batch_size = 64
    w = nas_args.w
    true_targets= targets
   
    model = define_model(alpha, beta, B0, t_zero, res, batch_size)
    if noise:
        with torch.no_grad():
            target_pred = model(inputs)
            target_noise = model(inputs_noisy)
            diff = target_pred - target_noise
            noise_immunity = -torch.sum(diff ** 2)
            noise_immunity_hist.append(noise_immunity)
            normalized_ni = (noise_immunity - torch.mean(torch.tensor(noise_immunity_hist)))/torch.std(torch.tensor(noise_immunity_hist))
            print("Normalized NI: %f", normalized_ni)
            #normalized_ni = normalize_score(noise_immunity, noise, score_2=False)

    # loader_train, loader_eval = data_loader(
    #     image_size=(3, res, res), batch_size=batch_size
    # )
    logging.info(
        "Phinet params: %s",
        {"alpha": alpha, "beta": beta, "t_zero": t_zero, "B0": B0, "res": res},
    )

    total_params = get_params(model, res)

    if nas_args.algo == "evo":
        # top1 = train_and_evaluate(model, epochs, loader_train, loader_eval)
        # obj = EA_objective(top1, total_params, nas_args.target, w)
        score = compute_nwot(model, inputs, targets, split_data=1, loss_fn=None)
        obj = score_objective(score, total_params, nas_args.target, w)
    else:
        # score = score_network(model, batch_size, loader_train)
        if nas_args.predictor == 'synflow':
            score = compute_logsynflow(model, inputs, nas_args.device, mode="param")
        elif nas_args.predictor == 'epenas':
            score = compute_epe_score(model, inputs, targets, split_data=1, loss_fn=None)
        elif nas_args.predictor == 'freerea':
            score_syn = compute_logsynflow(model, inputs, nas_args.device, mode="param")
            logging.info("Logsynflow:%f", score_syn)
            score_nwot = compute_nwot(model, inputs, targets, split_data=1, loss_fn=None) 
            logging.info("Naswot:%f", score_nwot)
            score =  normalize_score_free_rea(score_nwot, noise=False, score_2=True, synflow=False) + normalize_score_free_rea(score_syn, noise=False, score_2=False, synflow=True)
            #score = score_syn + score_nwot
            #score = normalize_score(score, noise=False, score_2=False, synflow=True)
        else:
            score = compute_nwot(model, inputs, targets, split_data=1, loss_fn=None) 
            #logging.info("Naswot:%f", score)

        #logging.info("Score: %f", score)
        #score = normalize_score(score, noise=False, score_2=True, synflow=False)
        #logging.info("Normalize Score:%f", score)
        scores.append(score)
         # Normalize the score
        #score = normalize_score(score, noise=False, score_2=False)
        #logging.info("Normalized Score: %f", score)
        if noise:
            normalized_score = (score - torch.mean(torch.tensor(scores)))/torch.std(torch.tensor(scores))
            print("Normalized score: %f", normalized_score)
            score = 2*normalized_score + normalized_ni
            #score = normalize_score(score, noise=False, score_2=True)
            logging.info("Final Score: %f", score)

        obj = score_objective(score, total_params, nas_args.target, w)

    # obj = -1 * score - phi(total_params - nas_args.target)**w
    # target_obj = (total_params / nas_args.target) ** w
    # target_obj = 100 * np.abs(nas_args.target - total_params)

    #power = np.floor(np.log10(nas_args.target))
    #target_obj = np.abs(total_params - nas_args.target) / 10**power
    #logging.info("w: %f", w)
    logging.info("Total params: %f", total_params / 1e6)
    if nas_args.algo == "evo":
        # logging.info("Loss: %f", loss)
        logging.info("Score: %f", score)
    else:
        logging.info("Score: %f", score)
    logging.info("Objective: %f", -obj)
    target_obj = total_params/nas_args.target
    logging.info("Target obj: %f", target_obj)
    
    objs.append(-obj)
    params.append(total_params / 1e6)
    target_objs.append(target_obj)
    # if nas_args.algo == "evo":
    #     losses.append(loss)
    #     scores.append(score)
    # else:
    #     scores.append(score)
    #target_objs.append(target_obj)

    return [
        {"name": "objective", "type": "objective", "value": -obj},
        {"name": "params", "type": "statistic", "value": total_params / 1e6},
        {
            "name": "loss" if nas_args.algo == "loss" else "score",
            "type": "statistic",
            "value": loss if nas_args.algo == "loss" else score,
        },
        {"name": "target_obj", "type": "statistic", "value": target_obj},
    ]


def run_hpo(exp_name):
    # Specify the database where the experiments are stored
    storage = {
        "type": "legacy",
        "database": {
            "type": "pickleddb",
            "host": "/home/majam001/mj/micromind/recipes/NAS/orion_exp/exp_database.pkl",
        },
    }

    # Load the data for the specified experiment
    if nas_args.algo == "random":
        algorithm = {
            "random": {
                "seed": 42,
            },
        }
    elif nas_args.algo == "grid_search":
        algorithm = {
            "gridsearch": {"n_values": 100},
        }

    elif nas_args.algo == "hyperband":
        algorithm = {
            "hyperband": {"seed": 42, "repetitions": 1},
        }

    elif nas_args.algo == "asha":
        algorithm = {
            "asha": {"seed": 42, "repetitions": 1, "num_brackets": 1},
        }

    elif nas_args.algo == "dehb":
        algorithm = {
            "dehb": {
                "seed": 42,
                "mutation_factor": 0.5,
                "crossover_prob": 0.5,
                "mutation_strategy": "rand1",
                "crossover_strategy": "bin",
                "boundary_fix_type": "random",
            },
        }

    elif nas_args.algo == "pbt":
        algorithm = {
            "pbt": {
                "seed": 42,
                "population_size": 50,
                "generations": 10,
                "fork_timeout": 60,
                "exploit": {
                    "of_type": "PipelineExploit",
                    "exploit_configs": [
                        {
                            "of_type": "BacktrackExploit",
                            "min_forking_population": 5,
                            "truncation_quantile": 0.9,
                            "candidate_pool_ratio": 0.2,
                        },
                        {
                            "of_type": "TruncateExploit",
                            "min_forking_population": 5,
                            "truncation_quantile": 0.8,
                            "candidate_pool_ratio": 0.2,
                        },
                    ],
                },
                "explore": {
                    "of_type": "PipelineExplore",
                    "explore_configs": [
                        {"of_type": "ResampleExplore", "probability": 0.2},
                        {
                            "of_type": "PerturbExplore",
                            "factor": 1.2,
                            "volatility": 0.0001,
                        },
                    ],
                },
            },
        }

    elif nas_args.algo == "pb2":
        algorithm = {
            "pb2": {
                "seed": 42,
                "population_size": 50,
                "generations": 10,
                "fork_timeout": 60,
                "exploit": {
                    "of_type": "PipelineExploit",
                    "exploit_configs": [
                        {
                            "of_type": "BacktrackExploit",
                            "min_forking_population": 5,
                            "truncation_quantile": 0.9,
                            "candidate_pool_ratio": 0.2,
                        },
                        {
                            "of_type": "TruncateExploit",
                            "min_forking_population": 5,
                            "truncation_quantile": 0.8,
                            "candidate_pool_ratio": 0.2,
                        },
                    ],
                },
            },
        }

    elif nas_args.algo == "mofa":
        algorithm = {
            "mofa": {
                "seed": 42,
                "index": 1,
                "n_levels": 5,
                "strength": 2,
                "threshold": 0.1,
            },
        }

    elif nas_args.algo == "nevergrad":
        algorithm = {
            "nevergradoptimizer": {
                "seed": 42,
                "budget": 1000,
                "num_workers": 10,
                "model_name": "NGOpt",
            },
        }

    elif nas_args.algo == "hebo":
        algorithm = {
            "hebo": {
                "seed": 42,
                "parameters": 1000,
                "model_name": "catboost",
                "random_samples": 5,
                "acquisition_class": "hebo.acquisitions.acq.MACE",
                "evolutionary_strategy": "nsga2",
                "model_config": "null",
            },
        }

    elif nas_args.algo == "ax":
        algorithm = {
            "ax": {
                "seed": 42,
                "n_initial_trials": 5,
                "parallel_strategy": {
                    "of_type": "StatusBasedParallelStrategy",
                    "strategy_configs": {
                        "broken": {
                            "of_type": "MaxParallelStrategy",
                        },
                    },
                },
            },
        }

    elif nas_args.algo == "evo":
        algorithm = {
            "EvolutionES": {
                "seed": 42,
                "repetitions": 1,
                "nums_population": 20,
                "mutate": {
                    "function": "orion.algo.evolution_es.mutate_functions.default_mutate",
                    "multiply_factor": 3.0,
                    "add_factor": 1,
                },
            }
        }

    elif nas_args.algo == "bohb":
        algorithm = {
            "bohb": {
                "min_points_in_model": 20,
                "top_n_percent": 15,
                "num_samples": 64,
                "random_fraction": 0.33,
                "bandwidth_factor": 3,
                "min_bandwidth": 1e-3,
                "parallel_strategy": {
                    "of_type": "StatusBasedParallelStrategy",
                    "strategy_configs": {"broken": {"of_type": "MaxParallelStrategy"}},
                },
            }
        }

    elif nas_args.algo == "tpe":
        algorithm = {
            "tpe": {
                "seed": 42,
                "n_initial_points": 20,
                "n_ei_candidates": 25,
                "gamma": 0.1,
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
            # "res": "choices([32,64,128,160,240,256])",
            "res": "choices([64])",
            "epochs": "fidelity(1, 10, base=2)",
        },
        algorithm=algorithm,
        storage=storage,
        max_trials=100,
    )
    #experiment.executor.n_workers = 10
    
    # fixed res=64 and batch_size=64
    loader_train, loader_eval = data_loader(image_size=(3, 64, 64), batch_size=64)

    inputs, targets = get_mod_data(
        loader_train, nas_args.num_classes, 6, nas_args.device
    )
    #print(targets)
    trials = 1
    
    inputs_noisy = add_noise(inputs, noise_factor=0.1)
    noise = False
     
    while not experiment.is_done:
        print("trial", trials)
        trial = experiment.suggest(pool_size=5)
        if trial is None and experiment.is_done:
            break
        obj = objective(noise, inputs, inputs_noisy, targets, **trial.params)
        experiment.observe(trial, obj)
        trials += 1
    
    return experiment


if __name__ == "__main__":
    nas_args = configlib.parse()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(nas_args.device)
    # Set the start method to 'spawn'
    mp.set_start_method('spawn')

    objs = []
    params = []
    scores = []
    noise_immunity_hist = []
    target_objs = []
    losses = []
    min_score = 0
    max_score = 1
    min_score_2 = 0
    max_score_2 = 0
    min_ni = 0
    max_ni = 1
    min_synflow = 0
    max_synflow = 1
    norm_scores = []
    norm_params = []
    running_sum = 0.0

    if nas_args.dset == "cifar10":
        nas_args.num_classes = 10

    if nas_args.dset == "cifar100":
        nas_args.num_classes = 100

    # exp_name = (
    #     "obj1_seed_42_" + nas_args.algo + "_" + str(float(nas_args.target / 1e6)) + "M"
    # )
    # exp_name = (
    #     nas_args.algo
    #     + "_gamma_0.1_ei_10"
    #     + "_"
    #     + str(float(nas_args.target / 1e6))
    #     + "M"
    # )

    exp_name = (
        nas_args.dset + "_"  + "obj2_norm_score" + nas_args.predictor + "_" + nas_args.algo + "_" + str(float(nas_args.target / 1e6)) + "M"
    )
    #exp_name="trial"
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

    fh = logging.FileHandler(os.path.join(base_path, "info.log"), "w")

    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    experiment = run_hpo(exp_name)
    experiment = get_experiment(name=exp_name)
    best_trial = experiment.stats.best_trials_id
    best_params = experiment.get_trial(uid=best_trial).params
    best_obj = experiment.stats.best_evaluation
    logging.info("Best params: {}".format(best_params))
    logging.info("Best obj value: {:.3f}".format(best_obj))
    best_loss = experiment.get_trial(uid=best_trial).statistics[1].value
    if nas_args.algo == "evo":
        # logging.info("Best loss value: {:.3f}".format(best_loss))
        logging.info("Best score value: {:.3f}".format(best_loss))
    else:
        logging.info("Best score value: {:.3f}".format(best_loss))

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
    logging.info("Best trial total params: {:.5f}".format(total_param / 1e6))

    plot_table(exp_name, base_path)

    plot(exp_name, base_path, "objective")
    if nas_args.algo == "evo":
        # plot(exp_name, base_path, "loss")
        plot(exp_name, base_path, "score")
    else:
        plot(exp_name, base_path, "score")
    plot(exp_name, base_path, "target_obj")

    plot_pareto(exp_name, base_path, "objective", nas_args.target/1e6)
    plot_pareto(exp_name, base_path, "score", nas_args.target/1e6)

    save_path = base_path
    #nas_args.classifier = False
    if nas_args.classifier or nas_args.algo == "evo":
        top_3_net = topk_net(base_path, exp_name, nas_args.k)

        for index, row in top_3_net.iterrows():
            # Create a dictionary with the extracted values
            best_params = {
                "B0": int(row["B0"]),
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "res": int(row["res"]),
                "t_zero": float(row["t_zero"]),
            }
            print(best_params)
            print(type(best_params))

            optimized_params(
                nas_args,
                best_params,
                float(row["objective"]),
                float(row["score"]),
                nas_args.num_classes,
                save_path,
                exp_name + "_" + str(index),
            )

