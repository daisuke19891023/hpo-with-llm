"""Logging utilities for capturing HPO trial progress."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from .schemas import HPOTrialRequest, HPOTrialResponse, TrialObservation


class HPOTrialLogger(Protocol):
    """Protocol describing trial logging behaviour."""

    def record(
        self,
        *,
        request: HPOTrialRequest,
        response: HPOTrialResponse,
        observation: TrialObservation,
    ) -> None:
        """Persist information about a completed trial."""


@dataclass
class CSVTrialLogger:
    """Append HPO trial results to a CSV file for offline analysis."""

    path: Path | str
    include_header: bool = True

    def __post_init__(self) -> None:
        """Initialise internal state for CSV persistence."""
        self._path = Path(self.path)
        self._header_written = False
        self._fieldnames = [
            "timestamp",
            "task_id",
            "trial_id",
            "succeeded",
            "score",
            "composite_score",
            "parameters",
            "plan_description",
            "evaluation_indicators",
            "evaluation_metrics",
            "judge_score",
            "dataset",
            "notes",
        ]

    def record(
        self,
        *,
        request: HPOTrialRequest,
        response: HPOTrialResponse,
        observation: TrialObservation,
    ) -> None:
        """Build a CSV row from the observation and append it to the log."""
        row = self._build_row(
            request=request,
            response=response,
            observation=observation,
        )
        self._write_row(row)

    def _build_row(
        self,
        *,
        request: HPOTrialRequest,
        response: HPOTrialResponse,
        observation: TrialObservation,
    ) -> dict[str, object]:
        timestamp = datetime.now(tz=UTC).isoformat()

        evaluation_meta, summary = _extract_evaluation_summary(observation)
        indicator_names, indicator_metrics = _collect_indicator_details(summary)
        judge_score = _extract_judge_score(summary, evaluation_meta)
        plan_description = _resolve_plan_description(
            summary,
            evaluation_meta,
            observation,
            request,
        )
        dataset_name = _extract_dataset_name(evaluation_meta)

        row: dict[str, object] = {
            "timestamp": timestamp,
            "task_id": request.task.task_id,
            "trial_id": observation.trial_id,
            "succeeded": observation.succeeded,
            "score": observation.score,
            "composite_score": summary.get("composite_score", response.score),
            "parameters": json.dumps(observation.hyperparameters, sort_keys=True),
            "plan_description": plan_description,
            "evaluation_indicators": ",".join(indicator_names),
            "evaluation_metrics": json.dumps(indicator_metrics, sort_keys=True),
            "judge_score": judge_score if judge_score is not None else "",
            "dataset": dataset_name,
            "notes": observation.notes or "",
        }

        return row

    def _write_row(self, row: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        write_header = False
        if self.include_header and not self._header_written:
            if not self._path.exists() or self._path.stat().st_size == 0:
                write_header = True
            self._header_written = True

        with self._path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _extract_evaluation_summary(
    observation: TrialObservation,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return evaluation metadata and its summary mapping."""
    evaluation_meta_raw: object | None = None
    if observation.metadata:
        evaluation_meta_raw = cast("object", observation.metadata.get("evaluation"))
    evaluation_meta = _coerce_str_mapping(evaluation_meta_raw)

    summary_raw = _maybe_get(evaluation_meta, "summary")
    summary = _coerce_str_mapping(summary_raw)

    return evaluation_meta, summary


def _collect_indicator_details(
    summary: Mapping[str, object],
) -> tuple[list[str], dict[str, object]]:
    """Collect indicator names and metrics from the evaluation summary."""
    indicators_raw = _maybe_get(summary, "indicators")
    indicators = _coerce_sequence(indicators_raw)

    indicator_names: list[str] = []
    indicator_metrics: dict[str, object] = {}
    for indicator in indicators:
        indicator_mapping = _coerce_str_mapping(indicator)
        if not indicator_mapping:
            continue
        name = indicator_mapping.get("name")
        if not isinstance(name, str):
            continue
        indicator_names.append(name)
        metrics_payload = indicator_mapping.get("metrics")
        metrics_mapping = _coerce_str_mapping(metrics_payload)
        if metrics_mapping:
            indicator_metrics[name] = metrics_mapping

    return indicator_names, indicator_metrics


def _extract_judge_score(
    summary: Mapping[str, object],
    evaluation_meta: Mapping[str, object],
) -> int | float | str | None:
    """Extract the judge score from the evaluation payloads."""
    judge_payload = _maybe_get(summary, "judge")
    if judge_payload is None and evaluation_meta:
        judge_payload = _maybe_get(evaluation_meta, "judge")
    judge_mapping = _coerce_str_mapping(judge_payload)
    if not judge_mapping:
        return None

    score = judge_mapping.get("score")
    if isinstance(score, (int, float, str)):
        return score
    return None


def _resolve_plan_description(
    summary: Mapping[str, object],
    evaluation_meta: Mapping[str, object],
    observation: TrialObservation,
    request: HPOTrialRequest,
) -> str:
    """Determine the best available plan description for the log entry."""
    plan_description_raw = _maybe_get(summary, "plan")
    if not isinstance(plan_description_raw, str):
        plan_description_raw = _maybe_get(evaluation_meta, "plan")
    if not isinstance(plan_description_raw, str):
        plan_description_raw = observation.notes or ""
    if not plan_description_raw:
        plan_description_raw = request.task.description
    return plan_description_raw


def _extract_dataset_name(evaluation_meta: Mapping[str, object]) -> str:
    """Return the dataset identifier from evaluation metadata when present."""
    dataset_name_raw = _maybe_get(evaluation_meta, "dataset")
    return dataset_name_raw if isinstance(dataset_name_raw, str) else ""


def _maybe_get(mapping: Mapping[str, object] | None, key: str) -> object | None:
    """Safely retrieve a value from a mapping preserving ``object`` typing."""
    if mapping is None:
        return None
    return mapping.get(key)


def _coerce_str_mapping(value: object) -> dict[str, object]:
    """Convert mapping-like objects to a ``dict[str, Any]`` when possible."""
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        items = cast("Iterable[tuple[object, object]]", value.items())
        for key_obj, entry in items:
            if not isinstance(key_obj, str):
                continue
            result[key_obj] = entry
        return result
    return {}


def _coerce_sequence(value: object) -> tuple[object, ...]:
    """Return a sequence of objects excluding string-like values."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence = cast("Sequence[object]", value)
        return tuple(sequence)
    if isinstance(value, Iterable):
        iterable = cast("Iterable[object]", value)
        return tuple(iterable)
    return ()


__all__ = ["CSVTrialLogger", "HPOTrialLogger"]
