"""Tests for classification metric computation."""

from __future__ import annotations

import pytest

from clean_interfaces.evaluation.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
)


class TestComputeClassificationMetrics:
    """Validate metric calculations for binary predictions."""

    def test_metrics_are_computed_correctly(self) -> None:
        """Accuracy, precision, recall, and F1 should match expected values."""
        truth = [True, False, True, False]
        predictions = [True, True, False, False]

        metrics = compute_classification_metrics(truth, predictions)
        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.accuracy == pytest.approx(0.5)
        assert metrics.precision == pytest.approx(0.5)
        assert metrics.recall == pytest.approx(0.5)
        assert metrics.f1 == pytest.approx(0.5)

    def test_mismatched_lengths_raise(self) -> None:
        """Mismatched sequence lengths should raise ``ValueError``."""
        with pytest.raises(ValueError, match="same length"):
            compute_classification_metrics([True], [])

    def test_empty_sequences_raise(self) -> None:
        """Empty sequences should not be accepted."""
        with pytest.raises(ValueError, match="At least one label"):
            compute_classification_metrics([], [])
