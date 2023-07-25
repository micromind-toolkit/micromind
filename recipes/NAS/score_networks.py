import yaml
import torchinfo
import os

# Model interface
from micromind import PhiNet

# For argparse from multiple files
from micromind.utils import configlib

from timm.data import create_loader, create_transform, create_dataset
from recipes.image_classification.classification import optimized_params

from naswot import score_network
from epe_nas import compute_epe_score
from naswot_naslib import compute_nwot

# Model parameters
group = configlib.add_parser("Phinet parameters")
# group.add_argument(
#     "--alpha",
#     default=0.5,
#     type=float,
#     help="alpha parameter for phinet. Defaults to 0.5",
# )
# group.add_argument(
#     "--beta",
#     default=1.0,
#     type=float,
#     help="beta parameter for phinet. Defaults to 1.",
# )
# group.add_argument(
#     "--t_zero",
#     default=4,
#     type=float,
#     help="t_zero parameter for phinet. Defaults to 4.",
# )
# group.add_argument(
#     "--num_layers",
#     default=4,
#     type=int,
#     help="Number of layers for phinet. Defaults to 4.",
# )
# group.add_argument(
#     "--num_classes",
#     default=4,
#     type=int,
#     help="Number of classes for phinet. Defaults to 10.",
# ),
group.add_argument(
    "--batch_size",
    default=64,
    type=int,
    help="Batch size for the train loader",
)
group.add_argument(
    "--res",
    default=160,
    type=int,
    help="Resolution of the image",
)
# group.add_argument(
#     "--dset",
#     default="cifar10",
#     type=str,
#     help="Dataset",
# )
group.add_argument(
    "--train",
    default="False",
    type=bool,
    help="Model training on/off",
)


def get_params(model, res):
    return torchinfo.summary(model, (1, 3, res, res), device=0, verbose=0).total_params


def score_predict(alpha, beta, B0, t_zero, res, epochs, batch_size, loader_train):
    # utils.setup_default_logging()
    score_args = configlib.parse()
    # score_args.input_size = 3, 160, 160
    score_args.res = res
    print(score_args.res)
    score_args.alpha = alpha
    score_args.num_layers = B0
    score_args.beta = beta
    score_args.t_zero = t_zero
    score_args.num_classes = 10
    score_args.batch_size = batch_size

    phinet_params = {
        "input_shape": (3, vars(score_args)["res"], vars(score_args)["res"]),
        "alpha": vars(score_args)["alpha"],
        "num_layers": vars(score_args)["num_layers"],
        "beta": vars(score_args)["beta"],
        "t_zero": vars(score_args)["t_zero"],
        "squeeze_excite": True,
        "include_top": True,
        "num_classes": vars(score_args)["num_classes"],
        "compatibility": False,
        "h_swish": False,
    }

    model = PhiNet(**phinet_params)
    params = get_params(model, score_args.res)
    score = score_network(
        model, batch_size=score_args.batch_size, loader_train=loader_train
    )

    print(f"Params: {params}")
    print("Score: %f" % score)

    save_path = "./NASWOT_Phinet"
    exp_name = "test"
    base_path = os.path.join(save_path, score_args.dset, exp_name)
    os.makedirs(
        base_path,
        exist_ok=True,
    )

    # if score_args.train:
    # optimized_params(score_args, phinet_params, 10, save_path=base_path)


if __name__ == "__main__":
    score_predict()
