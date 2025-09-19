"""LLM-backed augmentation for reflection insights."""

from __future__ import annotations
import json
from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from clean_interfaces.llm import (
    LLMClient,
    LLMClientFactory,
    LLMConfigurationError,
    LLMGenerationRequest,
)
from clean_interfaces.hpo.schemas import (
    HyperparameterSpec,
    HyperparameterType,
    ParameterValue,
    ReflectionInsight,
    TrialObservation,
)


_REFLECTION_SYSTEM_PROMPT = (
    "You are a senior optimisation specialist assisting with LLM and RAG prompt "
    "tuning. Use metrics and qualitative judge feedback to recommend concrete "
    "hyperparameter changes."
)

_REFLECTION_RESPONSE_FORMAT: Mapping[str, object] = {
    "type": "json_schema",
    "json_schema": {
        "name": "hpo_reflection_summary",
        "schema": {
            "type": "object",
            "properties": {
                "hyperparameters": {
                    "type": "object",
                    "additionalProperties": True,
                    "default": {},
                },
                "insights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "detail": {"type": "string"},
                        },
                        "required": ["detail"],
                        "additionalProperties": False,
                    },
                    "default": [],
                },
                "actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "critique": {"type": "string"},
            },
            "required": ["hyperparameters"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class LLMReflectionContext:
    """Context supplied to the LLM when requesting augmented guidance."""

    best_trial: TrialObservation
    suggestion: Mapping[str, ParameterValue]
    search_space: Sequence[HyperparameterSpec]
    metrics: Mapping[str, object] | None
    judge: Mapping[str, object] | None
    plan: str | None


@dataclass(frozen=True)
class LLMReflectionResult:
    """Structured payload returned from the LLM strategy."""

    suggestion: dict[str, ParameterValue]
    insights: tuple[ReflectionInsight, ...]
    actions: tuple[str, ...]
    critique: str | None
    evidence: Mapping[str, object]


class LLMReflectionStrategy:
    """Call the configured LLM to gather additional reflection feedback."""

    def __init__(self, factory: LLMClientFactory | None = None) -> None:
        """Initialise the strategy with an optional client factory."""
        self._factory = factory or LLMClientFactory()

    def generate(
        self, context: LLMReflectionContext,
    ) -> LLMReflectionResult | None:
        """Request insights from the LLM if a client can be constructed."""
        try:
            client = self._factory.create()
        except LLMConfigurationError:
            return None

        payload = self._call_llm(client, context)
        if payload is None:
            return None

        suggestion = self._merge_llm_hyperparameters(
            payload.get("hyperparameters"),
            context.suggestion,
            context.search_space,
        )
        insights = tuple(self._insights_from_llm_payload(payload.get("insights")))
        actions = tuple(self._actions_from_llm_payload(payload.get("actions")))
        critique = None
        critique_candidate = payload.get("critique")
        if isinstance(critique_candidate, str) and critique_candidate.strip():
            critique = critique_candidate.strip()

        evidence = {"llm": dict(payload)}
        return LLMReflectionResult(
            suggestion=suggestion,
            insights=insights,
            actions=actions,
            critique=critique,
            evidence=evidence,
        )

    def _call_llm(
        self,
        client: LLMClient,
        context: LLMReflectionContext,
    ) -> Mapping[str, object] | None:
        llm_context = self._build_llm_context(context)

        context_payload = json.dumps(
            llm_context,
            ensure_ascii=False,
            indent=2,
        )
        user_prompt = (
            "Analyse the optimisation context and propose updated "
            "hyperparameters, insights, actions, and critique as JSON. Context:\n"
            f"{context_payload}"
        )

        request = LLMGenerationRequest(
            input=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": _REFLECTION_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ],
            response_format=_REFLECTION_RESPONSE_FORMAT,
            metadata={"best_trial": context.best_trial.trial_id},
        )

        try:
            result = client.generate(request)
        except Exception:  # pragma: no cover - defensive network path
            return None

        payload: Mapping[str, object] | None = None
        if isinstance(result.structured_output, Mapping):
            payload = result.structured_output
        elif result.function_call is not None:
            payload = result.function_call.arguments

        if payload is None:
            return None

        return dict(payload)

    def _build_llm_context(
        self, context: LLMReflectionContext,
    ) -> dict[str, object]:
        metrics_payload: Mapping[str, object] = (
            dict(context.metrics) if context.metrics is not None else {}
        )

        judge_payload: Mapping[str, object] = (
            dict(context.judge) if context.judge is not None else {}
        )

        return {
            "best_trial": {
                "id": context.best_trial.trial_id,
                "score": context.best_trial.score,
                "hyperparameters": context.best_trial.hyperparameters,
            },
            "suggestion": dict(context.suggestion),
            "metrics": metrics_payload,
            "judge": judge_payload,
            "plan": context.plan,
            "search_space": [
                self._serialise_spec(spec) for spec in context.search_space
            ],
        }

    def _serialise_spec(self, spec: HyperparameterSpec) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": spec.name,
            "type": (
                spec.param_type.value
                if hasattr(spec.param_type, "value")
                else str(spec.param_type)
            ),
        }
        if spec.lower is not None:
            payload["lower"] = spec.lower
        if spec.upper is not None:
            payload["upper"] = spec.upper
        if spec.choices:
            payload["choices"] = list(spec.choices)
        return payload

    def _merge_llm_hyperparameters(
        self,
        hyperparameters: object,
        suggestion: Mapping[str, ParameterValue],
        search_space: Sequence[HyperparameterSpec],
    ) -> dict[str, ParameterValue]:
        mapping = _coerce_mapping(hyperparameters)
        if mapping is None:
            return dict(suggestion)

        specs = {spec.name: spec for spec in search_space}
        merged = dict(suggestion)
        for name, value in mapping.items():
            spec = specs.get(name)
            if spec is None:
                continue
            coerced = self._coerce_parameter_value(value, spec)
            if coerced is not None:
                merged[spec.name] = coerced
        return merged

    def _coerce_parameter_value(
        self,
        value: object,
        spec: HyperparameterSpec,
    ) -> ParameterValue | None:
        param_type = self._normalise_param_type(spec.param_type)
        if param_type is None:
            return None

        match param_type:
            case HyperparameterType.FLOAT:
                return self._coerce_float(value, spec)
            case HyperparameterType.INT:
                return self._coerce_int(value, spec)
            case HyperparameterType.BOOL:
                return self._coerce_bool(value)
            case HyperparameterType.CATEGORICAL:
                return self._coerce_categorical(value, spec)
        return None

    def _normalise_param_type(
        self, param_type: HyperparameterType | str | object,
    ) -> HyperparameterType | None:
        if isinstance(param_type, HyperparameterType):
            return param_type
        try:
            return HyperparameterType(str(param_type))
        except ValueError:
            return None

    def _coerce_float(
        self,
        value: object,
        spec: HyperparameterSpec,
    ) -> float | None:
        if isinstance(value, (int, float)):
            numeric = float(value)
        elif isinstance(value, str):
            try:
                numeric = float(value)
            except ValueError:
                return None
        else:
            return None
        if spec.lower is not None:
            numeric = max(numeric, float(spec.lower))
        if spec.upper is not None:
            numeric = min(numeric, float(spec.upper))
        return round(numeric, 3)

    def _coerce_int(
        self,
        value: object,
        spec: HyperparameterSpec,
    ) -> int | None:
        if isinstance(value, (int, float)):
            numeric = int(float(value))
        elif isinstance(value, str):
            try:
                numeric = int(float(value))
            except ValueError:
                return None
        else:
            return None
        if spec.lower is not None:
            numeric = max(numeric, int(spec.lower))
        if spec.upper is not None:
            numeric = min(numeric, int(spec.upper))
        step = self._coerce_int_step(spec)
        if step is not None and spec.lower is not None:
            base = int(spec.lower)
            offset = max(0, (numeric - base) // step)
            numeric = base + offset * step
        return numeric

    def _coerce_int_step(self, spec: HyperparameterSpec) -> int | None:
        if spec.step is None:
            return None
        try:
            step = int(spec.step)
        except (TypeError, ValueError):
            return None
        if step <= 0:
            return None
        return step

    def _coerce_bool(self, value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "enable", "enabled"}:
                return True
            if lowered in {"false", "0", "no", "disable", "disabled"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    def _coerce_categorical(
        self,
        value: object,
        spec: HyperparameterSpec,
    ) -> str | None:
        if not spec.choices:
            return None
        candidate = str(value)
        if candidate in spec.choices:
            return candidate
        return None

    def _insights_from_llm_payload(self, data: object) -> list[ReflectionInsight]:
        insights: list[ReflectionInsight] = []
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            sequence = cast("Sequence[object]", data)
            for entry in sequence:
                mapping_entry = _coerce_mapping(entry)
                if mapping_entry is not None:
                    raw_title = str(mapping_entry.get("title", "LLM insight")).strip()
                    title = raw_title or "LLM insight"
                    detail = str(mapping_entry.get("detail", "")).strip()
                else:
                    title = "LLM insight"
                    detail = str(entry).strip()
                if detail:
                    insights.append(ReflectionInsight(title=title, detail=detail))
        return insights

    def _actions_from_llm_payload(self, data: object) -> list[str]:
        actions: list[str] = []
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            sequence = cast("Sequence[object]", data)
            for entry in sequence:
                text = str(entry).strip()
                if text:
                    actions.append(text)
        return actions


def _coerce_mapping(value: object) -> dict[str, object] | None:
    """Convert mapping-like objects into ``dict[str, object]`` payloads."""
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    items = cast("Iterable[tuple[object, object]]", value.items())
    for key_obj, entry in items:
        if not isinstance(key_obj, str):
            continue
        result[key_obj] = entry
    return result
