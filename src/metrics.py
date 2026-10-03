"""Evaluation metrics for MedMNIST classification tasks.

Supports binary-class, multi-class and multi-label tasks, and includes a
majority-class baseline so every reported metric can be compared against
"the simplest possible model" (see the PneumoniaMNIST EDA notebook).
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    task: str
    accuracy: float
    balanced_accuracy: float
    auc: float
    f1: float
    sensitivity: float | None = None   # binary-class only
    specificity: float | None = None   # binary-class only
    confusion: np.ndarray | None = None
    majority_baseline_accuracy: float | None = None

    def as_dict(self) -> dict:
        d = {
            "task": self.task,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "auc": self.auc,
            "f1": self.f1,
        }
        if self.sensitivity is not None:
            d["sensitivity"] = self.sensitivity
        if self.specificity is not None:
            d["specificity"] = self.specificity
        if self.majority_baseline_accuracy is not None:
            d["majority_baseline_accuracy"] = self.majority_baseline_accuracy
        return d

    def __repr__(self) -> str:
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in self.as_dict().items()]
        return f"EvalResult({', '.join(parts)})"


def majority_class_baseline(y_true: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class.

    Every trained model should beat this; see the EDA notebook for the
    PneumoniaMNIST train/val (~0.742) and test (~0.625) baselines.
    """
    values, counts = np.unique(y_true, return_counts=True)
    return counts.max() / counts.sum()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
            task: str) -> EvalResult:
    """Compute the full metric set for one split.

    Args:
        y_true: shape (N,) of integer class labels (binary/multi-class) or
            (N, n_classes) of 0/1 labels (multi-label).
        y_pred: predicted class labels, same shape convention as y_true.
        y_proba: predicted probabilities, shape (N, n_classes) — for binary
            tasks this may also be (N,) with the probability of class 1.
        task: one of 'binary-class', 'multi-class', 'multi-label, binary-class'.
    """
    if task == "multi-label, binary-class":
        return _evaluate_multilabel(y_true, y_pred, y_proba)

    accuracy = float(np.mean(y_true == y_pred))
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro" if task == "multi-class" else "binary")
    baseline = majority_class_baseline(y_true)

    n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2
    if task == "binary-class":
        proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        auc = roc_auc_score(y_true, proba_pos)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return EvalResult(task=task, accuracy=accuracy, balanced_accuracy=balanced_acc,
                          auc=auc, f1=f1, sensitivity=sensitivity, specificity=specificity,
                          confusion=cm, majority_baseline_accuracy=baseline)

    # multi-class: one-vs-rest AUC
    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                        labels=list(range(n_classes)))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return EvalResult(task=task, accuracy=accuracy, balanced_accuracy=balanced_acc,
                      auc=auc, f1=f1, confusion=cm, majority_baseline_accuracy=baseline)


def _evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: np.ndarray) -> EvalResult:
    accuracy = float(np.mean(y_true == y_pred))  # per-label accuracy, averaged
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_proba, average="macro")
    # balanced accuracy is not well-defined per-label in aggregate; report NaN
    return EvalResult(task="multi-label, binary-class", accuracy=accuracy,
                      balanced_accuracy=float("nan"), auc=auc, f1=f1)
