"""High-level evaluation service composing metrics and judge scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from clean_interfaces.base import BaseComponent

if TYPE_CHECKING:
    from .datasets import GoldenDataset
from .indicators import ClassificationIndicator, EvaluationIndicator, IndicatorResult
from .judge import (
    HeuristicLLMJudge,
    LLMJudgeProtocol,
    LLMJudgeResult,
    ResponsesAPIJudge,
)
from .metrics import ClassificationMetrics


@dataclass(frozen=True, slots=True)
class ScoreContribution:
    """Contribution of a specific metric or judge score to the composite."""

    name: str
    score: float
    weight: float

    def to_dict(self) -> dict[str, float | str]:
        """Return a serialisable representation of the contribution."""
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Combined evaluation artefacts returned by the service."""

    indicators: tuple[IndicatorResult, ...]
    judge: LLMJudgeResult | None
    composite_score: float
    components: tuple[ScoreContribution, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable representation of the evaluation."""
        payload: dict[str, object] = {
            "indicators": [result.to_dict() for result in self.indicators],
            "composite_score": self.composite_score,
            "components": [component.to_dict() for component in self.components],
        }

        classification_result = next(
            (
                result
                for result in self.indicators
                if result.name == ClassificationIndicator.name
            ),
            None,
        )
        if classification_result is not None:
            payload["metrics"] = dict(classification_result.metrics)

        if self.judge is not None:
            payload["judge"] = self.judge.to_dict()
        return payload


class EvaluationService(BaseComponent):
    """Service that computes metrics and optional LLM judge scores."""

    def __init__(
        self,
        dataset: GoldenDataset,
        *,
        indicators: Sequence[EvaluationIndicator] | None = None,
        judge: LLMJudgeProtocol | None = None,
        score_weights: Mapping[str, float] | None = None,
    ) -> None:
        """Initialise the service with dataset, indicators, and optional weights."""
        super().__init__()
        self._dataset = dataset
        configured = tuple(indicators or (ClassificationIndicator(),))

        weights_source: Mapping[str, object] | None
        if score_weights is not None:
            weights_source = score_weights
        else:
            metadata_weights = dataset.metadata.get("score_weights")
            if isinstance(metadata_weights, Mapping):
                weights_source = cast(Mapping[str, object], metadata_weights)
            else:
                weights_source = None

        self._score_weights = self._normalise_score_weights(weights_source)

        indicator_map: dict[str, EvaluationIndicator] = {}
        indicator_order: list[EvaluationIndicator] = []
        for indicator in configured:
            if indicator.name in indicator_map:
                msg = f"Duplicate indicator name detected: {indicator.name}"
                raise ValueError(msg)
            if not indicator.supports(dataset):
                msg = f"Indicator '{indicator.name}' is not supported by the dataset"
                raise ValueError(msg)
            indicator_map[indicator.name] = indicator
            indicator_order.append(indicator)

        self._indicators = indicator_map
        self._indicator_sequence = tuple(indicator_order)
        self._judge = judge or ResponsesAPIJudge(fallback=HeuristicLLMJudge())

        if self._score_weights:
            self.logger.debug(
                "Configured evaluation score weights",
                score_weights=self._score_weights,
            )

    @property
    def dataset(self) -> GoldenDataset:
        """Expose the underlying dataset."""
        return self._dataset

    def evaluate_predictions(
        self, predictions: Sequence[bool],
    ) -> ClassificationMetrics:
        """Compute metrics for the supplied predictions."""
        indicator = self._indicators.get(ClassificationIndicator.name)
        if indicator is None:
            msg = "Classification indicator is not configured for this service"
            raise ValueError(msg)

        payload = {ClassificationIndicator.name: list(predictions)}
        result = indicator.evaluate(self._dataset, payload)
        metrics_dict = dict(result.metrics)
        metrics = ClassificationMetrics(
            accuracy=float(metrics_dict.get("accuracy", 0.0)),
            precision=float(metrics_dict.get("precision", 0.0)),
            recall=float(metrics_dict.get("recall", 0.0)),
            f1=float(metrics_dict.get("f1", 0.0)),
        )
        self.logger.debug(
            "Computed evaluation metrics",
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
        )
        return metrics

    def judge_plan(self, plan: str) -> LLMJudgeResult:
        """Score a natural-language plan against dataset references."""
        if not self._dataset.reference_plans:
            msg = "Dataset does not define any reference plans for judging"
            raise ValueError(msg)

        result = self._judge.score_plan(plan, self._dataset.reference_plans)
        self.logger.debug("Judge produced score", score=result.score)
        return result

    def evaluate(
        self,
        predictions: Mapping[str, object] | Sequence[bool],
        *,
        plan: str | None = None,
    ) -> EvaluationSummary:
        """Evaluate predictions and optionally judge a plan."""
        payload = self._normalise_predictions(predictions)

        indicator_results: list[IndicatorResult] = []
        contributions: list[ScoreContribution] = []
        weighted_sum = 0.0
        total_weight = 0.0
        for indicator in self._indicator_sequence:
            result = indicator.evaluate(self._dataset, payload)
            indicator_results.append(result)
            indicator_contribs, sum_delta, weight_delta = self._score_indicator(result)
            contributions.extend(indicator_contribs)
            weighted_sum += sum_delta
            total_weight += weight_delta

        judge_result: LLMJudgeResult | None = None
        if plan is not None and self._dataset.reference_plans:
            judge_result = self.judge_plan(plan)

        if judge_result is not None:
            judge_weight = self._resolve_weight("judge", indicator_name=None)
            contributions.append(
                ScoreContribution(
                    name="judge",
                    score=judge_result.score,
                    weight=judge_weight,
                ),
            )
            if judge_weight:
                weighted_sum += judge_result.score * judge_weight
                total_weight += judge_weight

        composite = weighted_sum / total_weight if total_weight else 0.0
        self.logger.debug(
            "Computed composite evaluation score",
            composite_score=composite,
            total_weight=total_weight,
        )

        return EvaluationSummary(
            indicators=tuple(indicator_results),
            judge=judge_result,
            composite_score=composite,
            components=tuple(contributions),
        )

    def _normalise_predictions(
        self,
        predictions: Mapping[str, object] | Sequence[bool],
    ) -> dict[str, object]:
        """Normalise various prediction payload shapes into a mapping."""
        if isinstance(predictions, Mapping):
            return dict(predictions)

        if isinstance(predictions, (str, bytes)):
            msg = (
                "Predictions must be a mapping or a sequence of classification "
                "outputs"
            )
            raise TypeError(msg)

        return {ClassificationIndicator.name: list(predictions)}

    def _normalise_score_weights(
        self, weights: Mapping[str, object] | None,
    ) -> dict[str, float]:
        """Validate and normalise configured score weights."""
        normalised: dict[str, float] = {}
        if not weights:
            return normalised

        for key, value in weights.items():
            if not isinstance(value, (int, float)):
                msg = f"Score weight for '{key}' must be numeric"
                raise TypeError(msg)
            normalised[key] = float(value)
        return normalised

    def _resolve_weight(self, component: str, *, indicator_name: str | None) -> float:
        """Determine the weight for a specific score component."""
        if component in self._score_weights:
            return self._score_weights[component]

        if component == "judge":
            for key in ("judge", "judge.score"):
                if key in self._score_weights:
                    return self._score_weights[key]
            return self._score_weights.get("*", 1.0)

        indicator_key = indicator_name
        if indicator_key is None and "." in component:
            indicator_key = component.split(".", 1)[0]
        if indicator_key:
            for key in (indicator_key, f"{indicator_key}.*"):
                if key in self._score_weights:
                    return self._score_weights[key]

        return self._score_weights.get("*", 1.0)

    def _score_indicator(
        self, result: IndicatorResult,
    ) -> tuple[list[ScoreContribution], float, float]:
        """Convert an indicator result into contributions and weighted sums."""
        contributions: list[ScoreContribution] = []
        weighted_sum = 0.0
        total_weight = 0.0

        metrics = result.metrics
        if metrics:
            for metric_name, metric_value in metrics.items():
                component_name = f"{result.name}.{metric_name}"
                weight = self._resolve_weight(
                    component_name, indicator_name=result.name,
                )
                contributions.append(
                    ScoreContribution(
                        name=component_name,
                        score=float(metric_value),
                        weight=weight,
                    ),
                )
                if weight:
                    weighted_sum += float(metric_value) * weight
                    total_weight += weight
        elif result.score_components:
            for index, component in enumerate(result.score_components):
                component_name = f"{result.name}.component_{index}"
                weight = self._resolve_weight(
                    component_name, indicator_name=result.name,
                )
                contributions.append(
                    ScoreContribution(
                        name=component_name,
                        score=float(component),
                        weight=weight,
                    ),
                )
                if weight:
                    weighted_sum += float(component) * weight
                    total_weight += weight

        return contributions, weighted_sum, total_weight


__all__ = ["EvaluationService", "EvaluationSummary", "ScoreContribution"]
