"""Tests for the evaluation service and heuristic judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from clean_interfaces.evaluation.datasets import (
    ClassificationRecord,
    FileSearchRecord,
    GoldenDataset,
)
from clean_interfaces.evaluation.indicators import (
    ClassificationIndicator,
    FileSearchIndicator,
    FileSearchPrediction,
)
from clean_interfaces.evaluation.judge import HeuristicLLMJudge, LLMJudgeResult
from clean_interfaces.evaluation.service import EvaluationService

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class _PlanCall:
    plan: str


class RecordingJudge(HeuristicLLMJudge):
    """Judge that records the last plan it scored for verification."""

    def __init__(self) -> None:
        """Initialise the recording judge with an empty call log."""
        super().__init__()
        self.calls: list[_PlanCall] = []

    def score_plan(self, plan: str, references: Sequence[str]) -> LLMJudgeResult:
        """Record the plan before deferring to the parent implementation."""
        self.calls.append(_PlanCall(plan=plan))
        return super().score_plan(plan, references)


def _build_dataset() -> GoldenDataset:
    return GoldenDataset(
        classification=(
            ClassificationRecord(id="alpha", label=True),
            ClassificationRecord(id="beta", label=False),
            ClassificationRecord(id="gamma", label=True),
        ),
        reference_plans=(
            "Enable vector tooling for compliance reviews with grounded reasoning.",
            "Avoid tooling and prefer keyword search for FAQ answers.",
        ),
    )


def _build_file_search_dataset() -> GoldenDataset:
    return GoldenDataset(
        file_search=(
            FileSearchRecord(
                id="query-1",
                relevant_files=("a.txt", "b.txt"),
            ),
            FileSearchRecord(
                id="query-2",
                relevant_files=("c.txt",),
            ),
        ),
    )


class TestEvaluationService:
    """Validate behaviour of the evaluation service."""

    def test_evaluate_predictions_returns_metrics(self) -> None:
        """Metrics should be computed using the supplied predictions."""
        dataset = _build_dataset()
        service = EvaluationService(dataset)

        metrics = service.evaluate_predictions([True, False, False])
        assert metrics.accuracy == pytest.approx(2 / 3)
        assert metrics.precision == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(0.5)

    def test_judge_plan_requires_references(self) -> None:
        """Calling judge_plan without references should raise an error."""
        dataset = GoldenDataset(
            classification=(ClassificationRecord(id="only", label=True),),
        )
        service = EvaluationService(dataset)

        with pytest.raises(ValueError, match="reference plans"):
            service.judge_plan("Plan text")

    def test_evaluate_combines_metrics_and_judge(self) -> None:
        """The evaluation summary should include judge output and composite score."""
        dataset = _build_dataset()
        judge = RecordingJudge()
        service = EvaluationService(dataset, judge=judge)

        summary = service.evaluate(
            [True, False, False],
            plan="Enable vector tooling with grounded responses.",
        )

        classification_result = next(
            result
            for result in summary.indicators
            if result.name == ClassificationIndicator.name
        )
        assert classification_result.metrics["accuracy"] == pytest.approx(2 / 3)
        assert summary.judge is not None
        assert 0.0 <= summary.judge.score <= 1.0
        components = list(classification_result.score_components)
        components.append(summary.judge.score)
        composite = sum(components) / len(components)
        assert summary.composite_score == pytest.approx(composite)
        assert judge.calls
        assert "vector" in judge.calls[0].plan.lower()

    def test_evaluate_without_plan_skips_judge(self) -> None:
        """When no plan is supplied the judge should not run."""
        dataset = _build_dataset()
        judge = RecordingJudge()
        service = EvaluationService(dataset, judge=judge)

        summary = service.evaluate([True, False, False])

        assert summary.judge is None
        classification_result = next(
            result
            for result in summary.indicators
            if result.name == ClassificationIndicator.name
        )
        components = list(classification_result.score_components)
        average_component = sum(components) / len(components)
        assert summary.composite_score == pytest.approx(average_component)
        assert not judge.calls

    def test_prediction_length_validation(self) -> None:
        """Mismatched prediction length should raise a ``ValueError``."""
        dataset = _build_dataset()
        service = EvaluationService(dataset)

        with pytest.raises(ValueError, match="dataset size"):
            service.evaluate_predictions([True, False])

    def test_file_search_indicator_evaluates_retrieval(self) -> None:
        """File search indicators should compute retrieval metrics."""
        dataset = _build_file_search_dataset()
        indicator = FileSearchIndicator()
        service = EvaluationService(dataset, indicators=(indicator,))

        predictions = [
            FileSearchPrediction(
                query_id="query-1",
                retrieved_files=("a.txt", "b.txt"),
            ),
            FileSearchPrediction(
                query_id="query-2",
                retrieved_files=("c.txt",),
            ),
        ]

        summary = service.evaluate({indicator.name: predictions})

        assert not summary.judge
        assert summary.composite_score == pytest.approx(1.0)
        result = summary.indicators[0]
        assert result.name == indicator.name
        assert result.metrics["f1"] == pytest.approx(1.0)

    def test_missing_indicator_payload_raises(self) -> None:
        """Missing predictions for a configured indicator should raise an error."""
        dataset = _build_file_search_dataset()
        indicator = FileSearchIndicator()
        service = EvaluationService(dataset, indicators=(indicator,))

        with pytest.raises(ValueError, match="Missing predictions"):
            service.evaluate({})

    def test_evaluate_applies_score_weights(self) -> None:
        """Composite scores should honour configured weights."""
        dataset = _build_dataset()
        judge = RecordingJudge()
        weights = {
            "classification.accuracy": 0.25,
            "classification.precision": 0.5,
            "classification.recall": 2.0,
            "classification.f1": 0.75,
            "judge": 0.0,
        }
        service = EvaluationService(dataset, judge=judge, score_weights=weights)

        summary = service.evaluate(
            [True, False, False],
            plan="Enable vector tooling with grounded responses.",
        )

        assert summary.judge is not None
        contributions = {component.name: component for component in summary.components}
        assert contributions["judge"].weight == pytest.approx(0.0)
        assert contributions["classification.recall"].weight == pytest.approx(2.0)

        weighted_total = sum(
            component.score * component.weight
            for component in summary.components
            if component.weight
        )
        total_weight = sum(
            component.weight for component in summary.components if component.weight
        )
        assert total_weight > 0
        assert summary.composite_score == pytest.approx(weighted_total / total_weight)

    def test_dataset_metadata_configures_weights(self) -> None:
        """Score weights can be sourced from dataset metadata."""
        dataset = _build_dataset()
        dataset.metadata["score_weights"] = {"classification": 1.5, "judge": 0.0}
        judge = RecordingJudge()
        service = EvaluationService(dataset, judge=judge)

        summary = service.evaluate(
            [True, False, False],
            plan="Enable vector tooling with grounded responses.",
        )

        contributions = {component.name: component for component in summary.components}
        assert contributions["classification.accuracy"].weight == pytest.approx(1.5)
        assert contributions["classification.f1"].weight == pytest.approx(1.5)
        assert contributions["judge"].weight == pytest.approx(0.0)
