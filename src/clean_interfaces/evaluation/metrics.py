"""Metric computation helpers for evaluation workflows."""

from __future__ import annotations

import typing
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Aggregate metrics for binary classification evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation of the metrics."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def compute_classification_metrics(
    true_labels: typing.Sequence[bool],
    predicted_labels: typing.Sequence[bool],
) -> ClassificationMetrics:
    """Compute accuracy, precision, recall, and F1 score."""
    if len(true_labels) != len(predicted_labels):
        msg = "True and predicted label sequences must have the same length"
        raise ValueError(msg)

    if not true_labels:
        msg = "At least one label is required to compute metrics"
        raise ValueError(msg)

    tp = fp = tn = fn = 0
    for truth, prediction in zip(true_labels, predicted_labels, strict=False):
        if truth and prediction:
            tp += 1
        elif truth and not prediction:
            fn += 1
        elif not truth and prediction:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall
        else 0.0
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


__all__ = ["ClassificationMetrics", "compute_classification_metrics"]
