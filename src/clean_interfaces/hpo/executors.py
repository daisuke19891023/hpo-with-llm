"""Example trial executors for hyperparameter optimization."""

from __future__ import annotations

import typing
from collections.abc import Iterable
from typing import Final, cast

from clean_interfaces.evaluation.datasets import (
    GoldenDataset,
    load_default_golden_dataset,
)
from clean_interfaces.evaluation.indicators import (
    ClassificationIndicator,
    EvaluationIndicator,
    FileSearchIndicator,
    FileSearchPrediction,
)
from clean_interfaces.evaluation.service import EvaluationService

from .configuration import build_application_plan
from .schemas import HPOTrialRequest, HPOTrialResponse


DEFAULT_EXECUTOR_METADATA: Final[dict[str, object]] = {
    "executor": "default_trial_executor",
    "source": "simulated_llm",
}

_TEMPERATURE_TARGETS: Final = {
    "low": (0.2, 0.5),
    "medium": (0.35, 0.65),
    "high": (0.55, 0.85),
}

_TOKEN_TARGETS: Final = {
    "short": (0, 320),
    "medium": (280, 520),
    "long": (480, 1024),
}

_DEFAULT_GOLDEN_DATASET: Final = load_default_golden_dataset()

_GROUNDING_HIGH: Final = 0.55
_GROUNDING_LOW: Final = 0.35
_CONCISE_TOKEN_LIMIT: Final = 360
_CREATIVE_TEMPERATURE: Final = 0.6
_LOW_TEMPERATURE_THRESHOLD: Final = 0.45
_LONGFORM_TOKEN_THRESHOLD: Final = 480
_SHORT_TOKEN_THRESHOLD: Final = 320
_FILE_SEARCH_CREATIVE_TEMP: Final = 0.65
_FILE_SEARCH_CONSERVATIVE_TEMP: Final = 0.35


def _tooling_alignment(tooling_pref: object, use_tooling: bool) -> float:
    """Score alignment between tooling preference and usage."""
    if tooling_pref == "required":
        return 1.5 if use_tooling else -1.5
    if tooling_pref == "avoid":
        return 1.5 if not use_tooling else -1.5
    return 0.2


def _range_alignment(
    preference: object,
    targets: typing.Mapping[str, tuple[float, float]],
    value: float,
) -> float:
    """Return a bonus when the value falls within the preferred range."""
    if isinstance(preference, str) and preference in targets:
        low, high = targets[preference]
        return 1.0 if low <= value <= high else -1.0
    return 0.0


def _preference_bonuses(
    metadata: typing.Mapping[str, object],
    *,
    temperature: float,
    max_tokens: int,
    use_tooling: bool,
) -> float:
    """Apply additional bonuses and penalties from metadata preferences."""
    bonus = 0.0
    if metadata.get("prefer_grounding"):
        if temperature > _GROUNDING_HIGH:
            bonus -= 0.5
        elif temperature < _GROUNDING_LOW:
            bonus += 0.25

    if metadata.get("prefer_concise") and max_tokens > _CONCISE_TOKEN_LIMIT:
        bonus -= 0.5

    if metadata.get("favor_creativity") and temperature > _CREATIVE_TEMPERATURE:
        bonus += 0.5
    if metadata.get("favor_creativity") and not use_tooling:
        bonus += 0.25

    if metadata.get("prefer_keyword") and not use_tooling:
        bonus += 0.4
    elif metadata.get("prefer_keyword") and use_tooling:
        bonus -= 0.4

    return bonus


def _iter_strings(value: object) -> tuple[str, ...]:
    """Return the provided value as a tuple of strings if possible."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        iterable = cast("Iterable[object]", value)
        return tuple(item for item in iterable if isinstance(item, str))
    return ()


def _apply_file_search_preferences(
    retrieved: list[str],
    metadata: typing.Mapping[str, object],
    *,
    temperature: float,
    max_tokens: int,
    use_tooling: bool,
    aligned: bool,
    baseline: list[str],
) -> list[str]:
    """Adjust retrieved files according to metadata-driven heuristics."""
    adjusted = list(retrieved)

    if (
        (metadata.get("requires_tooling")
        and not use_tooling)
        or (metadata.get("prefers_keyword")
        and use_tooling)
    ):
        adjusted = adjusted[:1]

    if (
        metadata.get("prefer_creative")
        and temperature < _GROUNDING_HIGH
        and len(adjusted) > 1
    ):
        adjusted = adjusted[:-1]

    if not aligned:
        penalty_noise = _iter_strings(metadata.get("penalty_noise"))
        if penalty_noise:
            adjusted = adjusted[:1] if adjusted else []
            adjusted.extend(penalty_noise)

    creative_noise = _iter_strings(metadata.get("creative_noise"))
    conservative_noise = _iter_strings(metadata.get("conservative_noise"))
    if creative_noise and temperature > _FILE_SEARCH_CREATIVE_TEMP:
        adjusted.extend(creative_noise)
    elif conservative_noise and temperature < _FILE_SEARCH_CONSERVATIVE_TEMP:
        adjusted.extend(conservative_noise)

    longform_bonus = _iter_strings(metadata.get("longform_bonus"))
    concise_trim = _iter_strings(metadata.get("concise_trim"))
    if longform_bonus and max_tokens > _LONGFORM_TOKEN_THRESHOLD:
        adjusted.extend(longform_bonus)
    elif concise_trim and max_tokens < _SHORT_TOKEN_THRESHOLD:
        adjusted = adjusted[:1]

    if not adjusted:
        fallback = _iter_strings(metadata.get("fallback"))
        adjusted = list(fallback) if fallback else list(baseline)

    return list(dict.fromkeys(adjusted))


def _alignment_score(
    metadata: dict[str, object],
    *,
    temperature: float,
    max_tokens: int,
    use_tooling: bool,
) -> float:
    """Compute an alignment score between hyperparameters and scenario metadata."""
    score = _tooling_alignment(metadata.get("tooling", "optional"), use_tooling)
    score += _range_alignment(
        metadata.get("preferred_temperature"),
        _TEMPERATURE_TARGETS,
        temperature,
    )
    score += _range_alignment(
        metadata.get("token_budget"),
        _TOKEN_TARGETS,
        max_tokens,
    )
    score += _preference_bonuses(
        metadata,
        temperature=temperature,
        max_tokens=max_tokens,
        use_tooling=use_tooling,
    )
    return score


def _predict_outcomes(
    dataset: GoldenDataset,
    *,
    temperature: float,
    max_tokens: int,
    use_tooling: bool,
) -> tuple[list[bool], list[dict[str, object]]]:
    """Predict scenario success for each record based on hyperparameters."""
    predictions: list[bool] = []
    details: list[dict[str, object]] = []
    for record in dataset.classification:
        alignment = _alignment_score(
            record.metadata,
            temperature=temperature,
            max_tokens=max_tokens,
            use_tooling=use_tooling,
        )
        threshold = float(record.metadata.get("alignment_threshold", 1.0))
        predictions.append(alignment >= threshold)
        details.append(
            {
                "id": record.record_id,
                "alignment": alignment,
                "threshold": threshold,
            },
        )

    return predictions, details


def _simulate_file_search(
    dataset: GoldenDataset,
    *,
    alignment_by_id: dict[str, dict[str, float]],
    temperature: float,
    max_tokens: int,
    use_tooling: bool,
) -> tuple[list[FileSearchPrediction], list[dict[str, object]]]:
    """Simulate file retrieval outcomes for each query in the dataset."""
    predictions: list[FileSearchPrediction] = []
    details: list[dict[str, object]] = []

    for record in dataset.file_search:
        relevant = list(record.relevant_files)
        retrieved: list[str] = list(relevant)
        metadata = dict(record.metadata)
        alignment = alignment_by_id.get(record.query_id, {})
        score = float(alignment.get("alignment", 0.0))
        threshold = float(alignment.get("threshold", 1.0))
        aligned = score >= threshold

        deduped = _apply_file_search_preferences(
            retrieved,
            metadata,
            temperature=temperature,
            max_tokens=max_tokens,
            use_tooling=use_tooling,
            aligned=aligned,
            baseline=relevant,
        )

        predictions.append(
            FileSearchPrediction(
                query_id=record.query_id,
                retrieved_files=tuple(deduped),
            ),
        )
        details.append(
            {
                "query_id": record.query_id,
                "aligned": aligned,
                "alignment_score": score,
                "threshold": threshold,
                "relevant_files": list(record.relevant_files),
                "retrieved_files": deduped,
                "metadata": metadata,
            },
        )

    return predictions, details


def _build_trial_plan(temperature: float, max_tokens: int, use_tooling: bool) -> str:
    """Generate a natural-language plan reflecting the chosen hyperparameters."""
    sections: list[str] = []
    if use_tooling:
        sections.append(
            "Enable vector search tooling with a retrieval-augmented pipeline "
            "for grounded answers.",
        )
    else:
        sections.append(
            "Prefer keyword-driven responses without external tooling to stay "
            "lightweight for FAQs.",
        )

    if temperature < _LOW_TEMPERATURE_THRESHOLD:
        sections.append(
            "Maintain compliance-focused reasoning with a conservative "
            "temperature setting.",
        )
    elif temperature > _CREATIVE_TEMPERATURE:
        sections.append(
            "Encourage creative brainstorming while surfacing grounded takeaways.",
        )
    else:
        sections.append("Balance grounded analysis with concise executive summaries.")

    if max_tokens >= _LONGFORM_TOKEN_THRESHOLD:
        sections.append(
            "Reserve a long-form token budget for detailed reports and policy reviews.",
        )
    elif max_tokens <= _SHORT_TOKEN_THRESHOLD:
        sections.append(
            "Optimise for concise FAQ-style responses within a short token budget.",
        )
    else:
        sections.append(
            "Adopt a moderate token budget suitable for executive summaries.",
        )

    sections.append(
        "Operate at temperature "
        f"{temperature:.2f} with a budget of {max_tokens} tokens.",
    )

    return " ".join(sections)


class DefaultTrialExecutor:
    """Callable executor that evaluates trials using a golden dataset."""

    def __init__(
        self,
        *,
        dataset: GoldenDataset = _DEFAULT_GOLDEN_DATASET,
        evaluation_service: EvaluationService | None = None,
    ) -> None:
        self._dataset = dataset
        if evaluation_service is None:
            indicators: list[EvaluationIndicator] = [ClassificationIndicator()]
            if dataset.file_search:
                indicators.append(FileSearchIndicator())
            evaluation_service = EvaluationService(
                dataset,
                indicators=tuple(indicators),
            )
        self._evaluation_service = evaluation_service

    @property
    def dataset(self) -> GoldenDataset:
        """Return the dataset used by the executor."""
        return self._dataset

    @property
    def evaluation_service(self) -> EvaluationService:
        """Return the evaluation service used by the executor."""
        return self._evaluation_service

    def __call__(self, request: HPOTrialRequest) -> HPOTrialResponse:
        """Execute a trial evaluation using the configured dataset."""
        dataset = self._dataset
        evaluation_service = self._evaluation_service

        params = request.trial.hyperparameters
        temperature = float(params.get("temperature", 0.5))
        max_tokens = int(params.get("max_output_tokens", 256))
        use_tooling = bool(params.get("use_tooling", False))

        plan = build_application_plan(request.search_space, params)
        plan_metadata = plan.to_metadata()

        classification_predictions, alignment_details = _predict_outcomes(
            dataset,
            temperature=temperature,
            max_tokens=max_tokens,
            use_tooling=use_tooling,
        )

        alignment_by_id: dict[str, dict[str, float]] = {}
        for detail in alignment_details:
            identifier = detail.get("id")
            alignment = detail.get("alignment")
            threshold = detail.get("threshold")
            if not isinstance(identifier, str):
                continue
            alignment_value = (
                float(alignment) if isinstance(alignment, (int, float)) else 0.0
            )
            threshold_value = (
                float(threshold) if isinstance(threshold, (int, float)) else 1.0
            )
            alignment_by_id[identifier] = {
                "alignment": alignment_value,
                "threshold": threshold_value,
            }

        file_search_predictions: list[FileSearchPrediction] = []
        file_search_details: list[dict[str, object]] = []
        if dataset.file_search:
            file_search_predictions, file_search_details = _simulate_file_search(
                dataset,
                alignment_by_id=alignment_by_id,
                temperature=temperature,
                max_tokens=max_tokens,
                use_tooling=use_tooling,
            )

        plan_text = _build_trial_plan(temperature, max_tokens, use_tooling)

        indicator_payload: dict[str, object] = {
            ClassificationIndicator.name: classification_predictions,
        }
        file_indicator_name: str | None = None

        if file_search_predictions:
            file_indicator_name = FileSearchIndicator.name
            indicator_payload[file_indicator_name] = file_search_predictions

        summary = evaluation_service.evaluate(indicator_payload, plan=plan_text)
        summary_payload = summary.to_dict()

        evaluation_inputs: dict[str, object] = {
            "classification": {
                "labels": list(dataset.labels),
                "predictions": list(classification_predictions),
                "alignment": alignment_details,
            },
        }

        if file_indicator_name is not None:
            evaluation_inputs[file_indicator_name] = [
                {
                    "query_id": prediction.query_id,
                    "retrieved_files": list(prediction.retrieved_files),
                    "details": detail,
                }
                for prediction, detail in zip(
                    file_search_predictions,
                    file_search_details,
                    strict=False,
                )
            ]

        metadata = DEFAULT_EXECUTOR_METADATA.copy()
        metadata["application_plan"] = plan_metadata
        metadata["evaluation"] = {
            "dataset": dataset.metadata.get("name", "unknown_dataset"),
            "summary": summary_payload,
            "inputs": evaluation_inputs,
            "plan": plan_text,
        }

        notes = (
            "Composite score averages indicator metrics across classification "
            "and file search alongside the heuristic LLM judge evaluating the "
            "generated plan."
        )

        return HPOTrialResponse(
            score=summary.composite_score,
            succeeded=True,
            metadata=metadata,
            notes=notes,
        )


def default_trial_executor(
    request: HPOTrialRequest,
    *,
    dataset: GoldenDataset = _DEFAULT_GOLDEN_DATASET,
    evaluation_service: EvaluationService | None = None,
) -> HPOTrialResponse:
    """Evaluate a trial using the default executor configuration."""
    executor = DefaultTrialExecutor(
        dataset=dataset,
        evaluation_service=evaluation_service,
    )
    return executor(request)


__all__ = ["DefaultTrialExecutor", "default_trial_executor"]
