"""
Classification metrics for multi-class source separation.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

CLASS_NAMES = ["star", "galaxy", "artifact", "cosmic_ray"]


@dataclass
class ClassificationMetrics:
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    macro_auc: float = 0.0
    per_class_precision: dict = field(default_factory=dict)
    per_class_recall: dict = field(default_factory=dict)
    per_class_f1: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_auc": self.macro_auc,
        }
        for name in CLASS_NAMES:
            d[f"precision_{name}"] = self.per_class_precision.get(name, 0.0)
            d[f"recall_{name}"]    = self.per_class_recall.get(name, 0.0)
            d[f"f1_{name}"]        = self.per_class_f1.get(name, 0.0)
        return d


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> ClassificationMetrics:
    """
    Compute full classification metrics.

    Parameters
    ----------
    y_true  : (N,) integer class labels.
    y_pred  : (N,) predicted class labels.
    y_prob  : (N, C) class probabilities for AUC computation.
    class_names : list of class name strings.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)
    labels    = list(range(n_classes))

    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall    = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1        = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    macro_auc = 0.0
    if y_prob is not None:
        try:
            macro_auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=labels,
            )
        except ValueError:
            macro_auc = 0.0

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        macro_auc=float(macro_auc),
        per_class_precision={name: float(precision[i]) for i, name in enumerate(class_names)},
        per_class_recall={name: float(recall[i]) for i, name in enumerate(class_names)},
        per_class_f1={name: float(f1[i]) for i, name in enumerate(class_names)},
    )
