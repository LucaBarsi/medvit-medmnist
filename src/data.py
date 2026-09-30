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
    """ToTensor + normalization to [-1, 1] (same convention as the benchmark)."""
    mean, std = [0.5] * n_channels, [0.5] * n_channels
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std)])


def get_dataloaders(flag: str, size: int = 28, batch_size: int = 128,
                    num_workers: int = 2, root: str = "data"):
    """Create train/val/test DataLoaders plus the dataset specification."""
    spec = get_spec(flag)
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
