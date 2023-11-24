from typing import Dict

from torch.utils.data import DataLoader
from ultralytics.data import build_yolo_dataset


def create_loaders(m_cfg: Dict, data_cfg: Dict, batch_size: int):
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
    mode = "train"
    train_set = build_yolo_dataset(
        m_cfg,
        "datasets/coco8/images/train",
        batch_size,
        data_cfg,
        mode=mode,
        rect=mode == "val",
    )

    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=getattr(train_set, "collate_fn", None),
    )

    mode = "val"
    val_set = build_yolo_dataset(
        m_cfg,
        "datasets/coco8/images/val",
        batch_size,
        data_cfg,
        mode=mode,
        rect=mode == "val",
    )

    val_loader = DataLoader(
        val_set,
        batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=getattr(val_set, "collate_fn", None),
    )

    return train_loader, val_loader
