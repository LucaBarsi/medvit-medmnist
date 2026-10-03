"""Parameterized loading of MedMNIST subsets.

Switching subsets (e.g. 'pneumoniamnist' -> 'dermamnist') only requires
changing the `flag`; the number of classes, channels and the task type
come from the MedMNIST INFO dictionary.
"""
import os
from dataclasses import dataclass

import medmnist
import torch
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms


@dataclass
class DatasetSpec:
    flag: str
    task: str          # 'binary-class', 'multi-class', 'multi-label, binary-class', ...
    n_channels: int
    n_classes: int
    labels: dict       # {'0': 'normal', '1': 'pneumonia', ...}


def get_spec(flag: str) -> DatasetSpec:
    """Build a DatasetSpec from the MedMNIST INFO dictionary."""
    info = INFO[flag]
    return DatasetSpec(
        flag=flag,
        task=info["task"],
        n_channels=info["n_channels"],
        n_classes=len(info["label"]),
        labels=info["label"],
    )


def get_dataset(flag: str, split: str, size: int = 28,
                root: str = "data", transform=None):
    """Return the MedMNIST Dataset for a split ('train', 'val' or 'test')."""
    os.makedirs(root, exist_ok=True)
    DataClass = getattr(medmnist, INFO[flag]["python_class"])
    return DataClass(split=split, transform=transform, download=True,
                     size=size, root=root)


def default_transform(n_channels: int) -> transforms.Compose:
    """ToTensor + normalization to [-1, 1] (the benchmark's own convention).

    Kept mainly as a fallback / sanity-check baseline; prefer
    `compute_normalization_stats` + `default_transform(..., mean, std)` for
    actual training, since it adapts to each subset's real pixel statistics
    (see the PneumoniaMNIST EDA: mean 0.57 / std 0.17, not 0.5 / 0.5).
    """
    mean, std = [0.5] * n_channels, [0.5] * n_channels
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std)])


def compute_normalization_stats(flag: str, size: int = 28,
                                root: str = "data") -> tuple[list, list]:
    """Compute per-channel mean/std from the TRAINING split only.

    Images are read raw (no transform) and scaled to [0, 1] before the
    statistics are computed, so the result can be plugged straight into
    transforms.Normalize. Only using the training split avoids leaking
    information from validation/test into the normalization.
    """
    train_ds = get_dataset(flag, "train", size, root, transform=None)
    imgs = train_ds.imgs.astype("float32") / 255.0  # (N, H, W) or (N, H, W, C)
    if imgs.ndim == 3:  # grayscale: add a channel axis
        imgs = imgs[..., None]
    mean = imgs.mean(axis=(0, 1, 2)).tolist()
    std = imgs.std(axis=(0, 1, 2)).tolist()
    return mean, std


def normalized_transform(mean: list, std: list) -> transforms.Compose:
    """ToTensor + normalization using explicit per-channel mean/std,
    typically produced by `compute_normalization_stats`.
    """
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std)])


def get_dataloaders(flag: str, size: int = 28, batch_size: int = 128,
                    num_workers: int = 2, root: str = "data",
                    use_train_stats: bool = True):
    """Create train/val/test DataLoaders plus the dataset specification.

    When `use_train_stats` is True (default), normalization uses the mean/std
    computed from the training split (see `compute_normalization_stats`).
    Set it to False to fall back to the fixed 0.5/0.5 benchmark convention.
    """
    spec = get_spec(flag)
    if use_train_stats:
        mean, std = compute_normalization_stats(flag, size, root)
        tf = normalized_transform(mean, std)
    else:
        tf = default_transform(spec.n_channels)

    train_ds = get_dataset(flag, "train", size, root, tf)
    val_ds = get_dataset(flag, "val", size, root, tf)
    test_ds = get_dataset(flag, "test", size, root, tf)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers),
    }
    return loaders, spec


def prepare_targets(labels: torch.Tensor, spec: DatasetSpec) -> torch.Tensor:
    """Adjust the label format according to the task.

    MedMNIST returns labels with shape (B, 1). For multi-class and binary
    tasks (treated as 2 classes) we use integer labels of shape (B,)
    (CrossEntropyLoss); for multi-label tasks we keep (B, n_classes) as
    floats (BCEWithLogitsLoss).
    """
    if spec.task == "multi-label, binary-class":
        return labels.float()
    return labels.squeeze(-1).long()
