"""
This code prepares the DataLoader compatible with HF accelerate and exploiting
timm data augmentation.
For compatibility, the prefetcher, JSDLoss and re_split options where disabled.

Authors:
    - Francesco Paissan, 2023

"""
import torch

from timm.data import (
    AugMixDataset,
    Mixup,
    create_dataset,
    create_transform,
)
from argparse import Namespace


def setup_mixup(args: Namespace):
    """Setup of Mixup data augmentation based on input configuration.

    Arguments
    ---------
    args : Namespace
        Input configuration for the experiment.

    Returns
    -------
    Mixup function and respective collate_fn. : Union[Callable, Callable]"""
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        mixup_fn = Mixup(**mixup_args)

    return mixup_fn, collate_fn


def create_loaders(args: Namespace):
    """Creates DataLoaders for dataset specified in the configuration file.
    Refer to ... for how to select the proper configuration.

    Arguments
    ---------
    args : Namespace
        Input configuration for the experiment.
    """
    # args.prefetcher = not args.no_prefetcher
    args.prefetcher = False
    args.distributed = False

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        # split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats,
    )
    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
    )

    mixup_fn, collate_fn = setup_mixup(args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = args.interpolation
    re_num_splits = 0
    dataset_train.transform = create_transform(
        input_size=args.input_shape,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=train_interpolation,
        mean=args.mean,
        std=args.std,
        tf_preprocessing=False,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )

    dataset_eval.transform = create_transform(
        input_size=args.input_shape,
        is_training=False,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=train_interpolation,
        mean=args.mean,
        std=args.std,
        tf_preprocessing=False,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=args.persistent_workers,
    )
    try:
        loader_train = loader_class(dataset_train, **loader_args)
        loader_args["drop_last"] = False
        loader_eval = loader_class(dataset_eval, **loader_args)
    except TypeError:
        loader_args.pop("persistent_workers")  # only in Pytorch 1.7+
        loader_train = loader_class(dataset_train, **loader_args)
        loader_args["drop_last"] = False
        loader_eval = loader_class(dataset_eval, **loader_args)

    return loader_train, loader_eval
