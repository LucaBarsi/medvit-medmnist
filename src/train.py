"""Training loop shared by every model (SimpleCNN now, ResNet/ViT/MedViT later).

Keeping this generic means swapping architectures is just a different
`model`, not a different training script.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .data import DatasetSpec, prepare_targets
from .metrics import EvalResult, evaluate


def _loss_fn(spec: DatasetSpec) -> nn.Module:
    if spec.task == "multi-label, binary-class":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def _logits_to_pred_proba(logits: torch.Tensor, spec: DatasetSpec):
    """Convert raw logits into (predicted labels, probabilities) as numpy arrays."""
    if spec.task == "multi-label, binary-class":
        proba = torch.sigmoid(logits)
        pred = (proba >= 0.5).long()
        return pred.cpu().numpy(), proba.detach().cpu().numpy()
    proba = torch.softmax(logits, dim=1)
    pred = proba.argmax(dim=1)
    return pred.cpu().numpy(), proba.detach().cpu().numpy()


def run_epoch(model, loader, spec: DatasetSpec, device, criterion,
             optimizer=None) -> dict:
    """Run one pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_seen = 0.0, 0
    all_true, all_pred, all_proba = [], [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            targets = prepare_targets(labels, spec).to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_seen += batch_size

            pred, proba = _logits_to_pred_proba(logits, spec)
            all_true.append(targets.cpu().numpy())
            all_pred.append(pred)
            all_proba.append(proba)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_proba = np.concatenate(all_proba)
    result = evaluate(y_true, y_pred, y_proba, spec.task)
    return {"loss": total_loss / n_seen, **result.as_dict()}, result


def train_model(model, loaders: dict, spec: DatasetSpec, device,
                epochs: int = 15, lr: float = 1e-3, out_dir: str = "../results",
                run_name: str = "run", monitor: str = "auc",
                early_stopping_patience: int | None = 5) -> dict:
    """Train `model`, tracking the best checkpoint by validation `monitor`.

    Saves, under `out_dir`:
    - `{run_name}_history.json`: per-epoch train/val metrics
    - `{run_name}_best.pt`: weights of the best epoch (by validation `monitor`)
    - `{run_name}_test_metrics.json`: final metrics on the test split

    Returns the history dict (also written to disk).
    """
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)
    criterion = _loss_fn(spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train": [], "val": []}
    best_score, best_epoch, epochs_without_improvement = -float("inf"), -1, 0

    for epoch in tqdm(range(1, epochs + 1), desc=f"{run_name}"):
        start = time.time()
        train_metrics, _ = run_epoch(model, loaders["train"], spec, device,
                                     criterion, optimizer)
        val_metrics, _ = run_epoch(model, loaders["val"], spec, device, criterion)
        train_metrics["epoch"] = val_metrics["epoch"] = epoch
        train_metrics["time_s"] = round(time.time() - start, 1)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        score = val_metrics.get(monitor, val_metrics["accuracy"])
        improved = score > best_score
        if improved:
            best_score, best_epoch, epochs_without_improvement = score, epoch, 0
            torch.save(model.state_dict(), os.path.join(out_dir, f"{run_name}_best.pt"))
        else:
            epochs_without_improvement += 1

        print(f"[{run_name}] epoch {epoch:02d} | "
             f"train loss {train_metrics['loss']:.4f} | "
             f"val {monitor} {score:.4f}"
             f"{' (best)' if improved else ''}")

        if (early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience):
            print(f"[{run_name}] no improvement in {early_stopping_patience} "
                 f"epochs, stopping early at epoch {epoch}.")
            break

    with open(os.path.join(out_dir, f"{run_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(os.path.join(out_dir, f"{run_name}_best.pt"),
                                      map_location=device))
    print(f"[{run_name}] loaded best checkpoint from epoch {best_epoch} "
         f"(val {monitor}={best_score:.4f})")
    return history


def evaluate_on_test(model, loaders: dict, spec: DatasetSpec, device,
                     out_dir: str = "../results", run_name: str = "run") -> EvalResult:
    """Evaluate `model` (assumed already loaded with the best weights) on the
    test split, print a summary and save the metrics to disk.
    """
    criterion = _loss_fn(spec)
    test_metrics, result = run_epoch(model, loaders["test"], spec, device, criterion)
    print(f"[{run_name}] TEST metrics: {result}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{run_name}_test_metrics.json"), "w") as f:
        json.dump(result.as_dict(), f, indent=2)
    return result
