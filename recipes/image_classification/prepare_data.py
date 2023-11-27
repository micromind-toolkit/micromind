from typing import Dict
import os
import torch

from timm.data import (
    AugMixDataset,
    FastCollateMixup,
    Mixup,
    create_dataset,
    create_loader,
    resolve_data_config,
    create_transform,
)
from argparse import Namespace
from torch.utils.data import DataLoader


def setup_mixup(args: Namespace):
    # setup mixup / cutmix
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
        if args.prefetcher:
            assert (
                not num_aug_splits
            )  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    return mixup_fn, collate_fn

def create_loaders(args: Namespace):
    """Creates DataLoaders for dataset specified in the configuration file.
    Refer to ... for how to select the proper configuration.

    Arguments
    ---------
    m_cfg : Dict
        Contains information about the training process (e.g., data augmentation).
    data_cfg : Dict
        Contains details about the data configurations (e.g., image size, etc.).
    batch_size : int
        Batch size for the training process.

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
    # if args.re_split:
        # # apply RE to second half of batch if no aug split otherwise line up with aug split
        # re_num_splits = num_aug_splits or 2
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
        # num_workers=args.num_workers,
        num_workers=4,
        # sampler=sampler,
        collate_fn=collate_fn,
        # pin_memory=args.pin_memory,
        pin_memory=True,
        drop_last=True,
        # worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        # persistent_workers=args.persistent_workers
        persistent_workers=True
    )
    try:
        loader_train = loader_class(dataset_train, **loader_args)
        loader_eval = loader_class(dataset_eval, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader_train = loader_class(dataset_train, **loader_args)
        loader_eval = loader_class(dataset_eval, **loader_args)

    return loader_train, loader_eval

