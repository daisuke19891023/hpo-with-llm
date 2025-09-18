"""Evaluation indicators modelling diverse optimisation objectives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast
from collections.abc import Mapping, Sequence

from .metrics import ClassificationMetrics, compute_classification_metrics

if TYPE_CHECKING:
    from .datasets import GoldenDataset


def _empty_metadata() -> dict[str, Any]:
    """Return a new metadata dictionary with precise typing."""
    return {}


@dataclass(frozen=True, slots=True)
class IndicatorResult:
    """Result produced by evaluating a specific indicator."""

    name: str
    metrics: Mapping[str, float]
    score_components: tuple[float, ...]
    per_sample: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result for structured logging."""
        return {
            "name": self.name,
            "metrics": dict(self.metrics),
            "score_components": list(self.score_components),
            "per_sample": [dict(sample) for sample in self.per_sample],
            "metadata": dict(self.metadata),
            "description": self.description,
        }


class EvaluationIndicator(Protocol):
    """Protocol implemented by evaluation indicator strategies."""

    name: str

    def supports(self, dataset: GoldenDataset) -> bool:
        """Return ``True`` when the dataset contains the necessary signals."""
        ...

    def evaluate(
        self,
        dataset: GoldenDataset,
        predictions: Mapping[str, object],
    ) -> IndicatorResult:
        """Evaluate predictions against the dataset and return a result."""
        ...


class ClassificationIndicator:
    """Indicator computing standard binary classification metrics."""

    name = "classification"

    def supports(self, dataset: GoldenDataset) -> bool:
        """Return ``True`` when the dataset contains classification records."""
        return bool(dataset.classification)

    def evaluate(
        self,
        dataset: GoldenDataset,
        predictions: Mapping[str, object],
    ) -> IndicatorResult:
        """Compute metrics for boolean predictions."""
        try:
            raw_predictions = predictions[self.name]
        except KeyError as exc:  # pragma: no cover - defensive
            message = "Missing predictions for classification indicator"
            raise ValueError(message) from exc
        if not isinstance(raw_predictions, Sequence):
            msg = "Classification indicator requires a sequence of predictions"
            raise TypeError(msg)

        predictions_seq = list(cast("Sequence[object]", raw_predictions))
        if len(predictions_seq) != len(dataset.classification):
            msg = (
                "Number of classification predictions must match dataset size "
                f"({len(predictions_seq)} != {len(dataset.classification)})"
            )
            raise ValueError(msg)

        predicted_labels: list[bool] = []
        per_sample: list[dict[str, Any]] = []
        true_positive = false_positive = true_negative = false_negative = 0

        for record, candidate_obj in zip(
            dataset.classification, predictions_seq, strict=True,
        ):
            if not isinstance(candidate_obj, bool):
                msg = "Classification predictions must be boolean values"
                raise TypeError(msg)

            candidate = candidate_obj
            predicted_labels.append(candidate)
            correct = candidate == record.label
            per_sample.append(
                {
                    "record_id": record.record_id,
                    "label": record.label,
                    "prediction": candidate,
                    "correct": correct,
                    "metadata": dict(record.metadata),
                },
            )

            if record.label and candidate:
                true_positive += 1
            elif record.label and not candidate:
                false_negative += 1
            elif not record.label and candidate:
                false_positive += 1
            else:
                true_negative += 1

        metrics = compute_classification_metrics(dataset.labels, predicted_labels)

        metadata = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        }

        return IndicatorResult(
            name=self.name,
            metrics=metrics.to_dict(),
            score_components=(
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1,
            ),
            per_sample=tuple(per_sample),
            metadata=metadata,
            description="Binary classification accuracy across golden scenarios.",
        )


@dataclass(frozen=True, slots=True)
class FileSearchPrediction:
    """Prediction describing retrieved files for a specific query."""

    query_id: str
    retrieved_files: tuple[str, ...]


def _parse_file_search_predictions(
    dataset: GoldenDataset,
    raw_predictions: object,
) -> list[FileSearchPrediction]:
    """Normalise raw predictions into :class:`FileSearchPrediction` objects."""
    if not isinstance(raw_predictions, Sequence):
        msg = "File search indicator requires a sequence of predictions"
        raise TypeError(msg)

    parsed: list[FileSearchPrediction] = []
    for item in cast("Sequence[object]", raw_predictions):
        if not isinstance(item, FileSearchPrediction):
            msg = "File search predictions must be FileSearchPrediction instances"
            raise TypeError(msg)
        parsed.append(item)

    if len(parsed) != len(dataset.file_search):
        msg = (
            "Number of file search predictions must match dataset size "
            f"({len(parsed)} != {len(dataset.file_search)})"
        )
        raise ValueError(msg)

    return parsed


def _index_file_search_predictions(
    predictions: Sequence[FileSearchPrediction],
) -> dict[str, FileSearchPrediction]:
    """Build a query identifier to prediction mapping."""
    prediction_map = {prediction.query_id: prediction for prediction in predictions}
    if len(prediction_map) != len(predictions):  # pragma: no cover - defensive
        msg = "Duplicate query identifiers detected in file search predictions"
        raise ValueError(msg)
    return prediction_map


def _score_file_search(
    dataset: GoldenDataset,
    prediction_map: Mapping[str, FileSearchPrediction],
) -> tuple[ClassificationMetrics, tuple[dict[str, Any], ...], dict[str, int]]:
    """Compute retrieval metrics and per-sample breakdowns."""
    per_sample: list[dict[str, Any]] = []
    true_positive = false_positive = false_negative = 0
    exact_matches = 0

    for record in dataset.file_search:
        prediction = prediction_map.get(record.query_id)
        if prediction is None:
            msg = f"Missing prediction for query '{record.query_id}'"
            raise ValueError(msg)

        relevant = set(record.relevant_files)
        retrieved = set(prediction.retrieved_files)

        tp = len(relevant & retrieved)
        fp = len(retrieved - relevant)
        fn = len(relevant - retrieved)
        matched = relevant == retrieved

        true_positive += tp
        false_positive += fp
        false_negative += fn
        exact_matches += int(matched)

        per_sample.append(
            {
                "query_id": record.query_id,
                "relevant_files": sorted(record.relevant_files),
                "retrieved_files": sorted(prediction.retrieved_files),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "matched": matched,
            },
        )

    total_queries = len(dataset.file_search)
    accuracy = exact_matches / total_queries if total_queries else 0.0

    precision_denominator = true_positive + false_positive
    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 1.0 if false_negative == 0 else 0.0
    )

    recall_denominator = true_positive + false_negative
    recall = (
        true_positive / recall_denominator
        if recall_denominator
        else 1.0 if false_positive == 0 else 0.0
    )

    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )

    metrics = ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )

    metadata = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }

    return metrics, tuple(per_sample), metadata


class FileSearchIndicator:
    """Indicator measuring retrieval quality against golden file lists."""

    name = "file_search"

    def supports(self, dataset: GoldenDataset) -> bool:
        """Return ``True`` when file search records are available."""
        return bool(dataset.file_search)

    def evaluate(
        self,
        dataset: GoldenDataset,
        predictions: Mapping[str, object],
    ) -> IndicatorResult:
        """Evaluate retrieval predictions against the golden dataset."""
        try:
            raw_predictions = predictions[self.name]
        except KeyError as exc:  # pragma: no cover - defensive
            message = "Missing predictions for file search indicator"
            raise ValueError(message) from exc

        parsed_predictions = _parse_file_search_predictions(dataset, raw_predictions)
        prediction_map = _index_file_search_predictions(parsed_predictions)
        metrics, per_sample, metadata = _score_file_search(dataset, prediction_map)

        return IndicatorResult(
            name=self.name,
            metrics=metrics.to_dict(),
            score_components=(
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1,
            ),
            per_sample=per_sample,
            metadata=metadata,
            description="Retrieval effectiveness compared to golden file lists.",
        )


__all__ = [
    "ClassificationIndicator",
    "EvaluationIndicator",
    "FileSearchIndicator",
    "FileSearchPrediction",
    "IndicatorResult",
]
