"""Tests for trial executor implementations."""

from __future__ import annotations

import pytest

from clean_interfaces.evaluation.datasets import ClassificationRecord, GoldenDataset
from clean_interfaces.hpo.executors import DefaultTrialExecutor, default_trial_executor
from clean_interfaces.hpo.schemas import (
    CodingTask,
    HPOTrialRequest,
    HPOSuggestedTrial,
    HyperparameterSpec,
    HyperparameterType,
)


def _search_space() -> tuple[HyperparameterSpec, ...]:
    return (
        HyperparameterSpec(
            name="temperature",
            param_type=HyperparameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        HyperparameterSpec(
            name="max_output_tokens",
            param_type=HyperparameterType.INT,
            lower=64,
            upper=1024,
            step=64,
        ),
        HyperparameterSpec(
            name="use_tooling",
            param_type=HyperparameterType.BOOL,
        ),
    )


def test_default_executor_returns_evaluation_metadata() -> None:
    """The default executor should attach evaluation metrics and judge output."""
    request = HPOTrialRequest(
        task=CodingTask(task_id="demo", description="Demo task"),
        trial=HPOSuggestedTrial(
            trial_id="0",
            hyperparameters={
                "temperature": 0.42,
                "max_output_tokens": 512,
                "use_tooling": True,
            },
        ),
        search_space=_search_space(),
        history=(),
    )

    response = default_trial_executor(request)

    assert 0.0 <= response.score <= 1.0
    assert response.succeeded is True
    assert response.notes

    evaluation = response.metadata.get("evaluation")
    assert evaluation is not None
    assert "summary" in evaluation
    summary = evaluation["summary"]
    indicators = {indicator["name"]: indicator for indicator in summary["indicators"]}
    classification = indicators.get("classification")
    assert classification is not None
    assert set(classification["metrics"]) == {"accuracy", "precision", "recall", "f1"}
    assert "inputs" in evaluation
    assert len(evaluation["inputs"]["classification"]["predictions"]) == len(
        evaluation["inputs"]["classification"]["labels"],
    )
    assert summary.get("judge") is not None
    assert "plan" in evaluation
    assert isinstance(evaluation["plan"], str)
    assert response.score == pytest.approx(summary["composite_score"])


def test_default_executor_accepts_custom_dataset() -> None:
    """Custom datasets should be usable with the default executor wrapper."""
    dataset = GoldenDataset(
        classification=(
            ClassificationRecord(
                record_id="example",
                label=True,
                metadata={"alignment_threshold": 0.0},
            ),
        ),
        metadata={"name": "custom_dataset"},
    )
    executor = DefaultTrialExecutor(dataset=dataset)

    request = HPOTrialRequest(
        task=CodingTask(task_id="custom", description="Custom task"),
        trial=HPOSuggestedTrial(
            trial_id="0",
            hyperparameters={
                "temperature": 0.6,
                "max_output_tokens": 512,
                "use_tooling": False,
            },
        ),
        search_space=_search_space(),
        history=(),
    )

    response = executor(request)

    evaluation = response.metadata.get("evaluation")
    assert evaluation is not None
    assert evaluation["dataset"] == "custom_dataset"
    classification_inputs = evaluation["inputs"]["classification"]
    assert classification_inputs["labels"] == [record.label for record in dataset.classification]
    assert len(classification_inputs["predictions"]) == dataset.size

