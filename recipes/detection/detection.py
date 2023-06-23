import os
import yaml

from micromind import PhiNet

# For argparse from multiple files
from micromind.utils import configlib
from micromind.utils.configlib import config as args

from yolo.model import microYOLO
from yolo.microyolohead import Microhead

# Dataset parameters
group = configlib.add_parser("Dataset parameters")
# Keep this argument outside of the dataset group because it is positional.
group.add_argument("data_dir", metavar="DIR", help="path to dataset")
group.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
group.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="Input batch size for training (default: 128)",
)
group.add_argument(
    "--epochs",
    type=int,
    default=300,
    metavar="N",
    help="number of epochs to train (default: 300)",
)
group.add_argument(
    "--num-classes",
    "-nc",
    type=int,
    metavar="NC",
    default="",
    help="number of classes (default: number of directory names)",
)
group.add_argument(
    "--img-size",
    type=int,
    default=320,
    metavar="N",
    help="Image patch size (default: None => model default)",
)
group.add_argument(
    "--device",
    type=str,
    default="0",
    metavar="N",
    help="Image patch size (default: None => model default)",
)


def _parse_args():

    # Do we have a config file to parse?
    # args_config, remaining = config_parser.parse_known_args()
    # if args_config.config:
    # with open(args_config.config, "r") as f:
    # cfg = yaml.safe_load(f)
    # parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    # args = parser.parse_args(remaining)
    configlib.parse()

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    args, args_text = _parse_args()
    print(args)
    print(args.epochs)
    train_nn()
    with open(os.path.join("./runs/", "args.yaml"), "w") as f:
        f.write(args_text)


def train_nn():
    # how to tell people that the net has to be this way in order for
    # the whole task to work?

    # define backbone
    backbone = PhiNet(
        input_shape=(3, 320, 320),
        alpha=0.67,
        num_layers=6,
        beta=1,
        t_zero=4,
        include_top=False,
        num_classes=args.num_classes,
        compatibility=False,
    )
    # define head
    head = Microhead()

    # load a model
    model = microYOLO(
        backbone=backbone, head=head, task="detect", nc=args.num_classes
    )  # build a new model from scratch DEFAULT_CFG

    # Train the model
    model.train(
        data=args.dataset,
        epochs=args.epochs,
        imgsz=args.img_size,
        device=(int(args.device) if args.device.isdigit() else args.device),
        task="detect",
    )
    model.export()


if __name__ == "__main__":
    main()
