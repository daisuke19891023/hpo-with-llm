"""Judge and metric augmentation for reflection suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from clean_interfaces.hpo.schemas import (
    HyperparameterSpec,
    HyperparameterType,
    ParameterValue,
    ReflectionInsight,
    TrialObservation,
)


@dataclass(frozen=True)
class JudgeAugmentationResult:
    """Structured result capturing judge-driven adjustments."""

    suggestion: dict[str, ParameterValue]
    insights: tuple[ReflectionInsight, ...]
    actions: tuple[str, ...]
    critique: str | None
    evidence: dict[str, object]
    metrics: dict[str, object]
    judge: dict[str, object] | None
    plan: str | None


class JudgeAugmentationStrategy:
    """Interpret evaluation metadata to adjust reflection suggestions."""

    def augment(
        self,
        *,
        best_trial: TrialObservation,
        search_space: Sequence[HyperparameterSpec],
        suggestion: Mapping[str, ParameterValue],
    ) -> JudgeAugmentationResult:
        """Enrich the baseline suggestion using judge feedback and metrics."""
        adjusted = dict(suggestion)
        insights: list[ReflectionInsight] = []
        actions: list[str] = []
        evidence: dict[str, object] = {}

        evaluation = self._evaluation_payload(best_trial)
        metrics, judge_payload = self._extract_summary(evaluation)
        plan_text = self._extract_plan(evaluation)
        critique: str | None = None

        if plan_text is not None:
            insights.append(
                ReflectionInsight(title="Plan under review", detail=plan_text),
            )
            evidence["plan"] = plan_text

        adjusted, metric_actions = self._integrate_metrics(
            metrics=metrics,
            adjusted=adjusted,
            search_space=search_space,
            insights=insights,
            evidence=evidence,
        )
        actions.extend(metric_actions)

        (
            adjusted,
            judge_actions,
            critique,
            judge_evidence,
            judge_insights,
        ) = self._integrate_judge(
            judge_payload=judge_payload,
            adjusted=adjusted,
            search_space=search_space,
        )
        actions.extend(judge_actions)
        insights.extend(judge_insights)
        if judge_evidence:
            evidence["judge"] = judge_evidence

        if metrics:
            evidence["metrics"] = {
                key: metrics.get(key)
                for key in ("accuracy", "precision", "recall", "f1")
                if key in metrics
            }

        return JudgeAugmentationResult(
            suggestion=adjusted,
            insights=tuple(insights),
            actions=tuple(dict.fromkeys(actions)),
            critique=critique,
            evidence=evidence,
            metrics=metrics,
            judge=judge_payload,
            plan=plan_text,
        )

    def _evaluation_payload(
        self, best_trial: TrialObservation,
    ) -> dict[str, object] | None:
        metadata = best_trial.metadata
        evaluation = metadata.get("evaluation")
        return _maybe_mapping(evaluation)

    def _extract_summary(
        self, evaluation: dict[str, object] | None,
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        if evaluation is None:
            return {}, None

        summary_payload = _maybe_mapping(evaluation.get("summary"))
        if summary_payload is not None:
            metrics = self._metrics_from_summary(summary_payload)
            judge_payload = _maybe_mapping(summary_payload.get("judge"))
            return metrics, judge_payload

        metrics = _coerce_mapping(evaluation.get("metrics"))
        judge_payload = _maybe_mapping(evaluation.get("judge"))
        return metrics, judge_payload

    def _extract_plan(self, evaluation: Mapping[str, object] | None) -> str | None:
        if evaluation is None:
            return None
        plan_candidate = evaluation.get("plan")
        if isinstance(plan_candidate, str):
            plan_text = plan_candidate.strip()
            if plan_text:
                return plan_text
        return None

    def _integrate_metrics(
        self,
        *,
        metrics: Mapping[str, object],
        adjusted: Mapping[str, ParameterValue],
        search_space: Sequence[HyperparameterSpec],
        insights: list[ReflectionInsight],
        evidence: dict[str, object],
    ) -> tuple[dict[str, ParameterValue], list[str]]:
        if not metrics:
            return dict(adjusted), []

        snapshot = self._metric_snapshot(metrics)
        if snapshot:
            insights.append(
                ReflectionInsight(title="Metric snapshot", detail="; ".join(snapshot)),
            )

        updated = self._adjust_from_metrics(metrics, adjusted, search_space)
        actions: list[str] = []
        if snapshot:
            actions.append(
                "Use metric deltas to decide whether to emphasise recall or precision.",
            )
        if snapshot:
            evidence["metric_snapshot"] = snapshot
        return updated, actions

    def _integrate_judge(
        self,
        *,
        judge_payload: dict[str, object] | None,
        adjusted: Mapping[str, ParameterValue],
        search_space: Sequence[HyperparameterSpec],
    ) -> tuple[
        dict[str, ParameterValue],
        list[str],
        str | None,
        dict[str, object],
        list[ReflectionInsight],
    ]:
        if judge_payload is None:
            critique = (
                "Judge feedback unavailable; falling back to metric-driven heuristics."
            )
            return dict(adjusted), [], critique, {}, []

        missing = _coerce_str_tuple(judge_payload.get("missing_keywords"))
        matched = _coerce_str_tuple(judge_payload.get("matched_keywords"))
        rationale = None
        rationale_candidate = judge_payload.get("rationale")
        if isinstance(rationale_candidate, str):
            cleaned = rationale_candidate.strip()
            if cleaned:
                rationale = cleaned

        insights = self._judge_insights(missing, matched)
        adjusted_map = self._adjust_from_judge(missing, adjusted, search_space)
        actions = self._actions_from_judge_keywords(missing)
        critique = rationale
        evidence: dict[str, object] = {
            "missing_keywords": list(missing),
            "matched_keywords": list(matched),
            "rationale": rationale,
            "insights": [insight.detail for insight in insights],
        }
        return adjusted_map, actions, critique, evidence, insights

    def _judge_insights(
        self,
        missing: Sequence[str],
        matched: Sequence[str],
    ) -> list[ReflectionInsight]:
        insights: list[ReflectionInsight] = []
        if missing:
            details = ", ".join(missing)
            insights.append(
                ReflectionInsight(
                    title="Judge feedback",
                    detail="Missing keywords highlighted by the judge: " + details,
                ),
            )
        if matched:
            covered = ", ".join(matched)
            insights.append(
                ReflectionInsight(
                    title="Strengths",
                    detail="Plan already covers: " + covered,
                ),
            )
        return insights

    def _metric_snapshot(self, metrics: Mapping[str, object]) -> list[str]:
        snapshot: list[str] = []
        for key in ("accuracy", "precision", "recall", "f1"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                snapshot.append(f"{key}={value:.2f}")
        return snapshot

    def _metrics_from_summary(
        self, summary_payload: dict[str, object],
    ) -> dict[str, object]:
        indicators_raw = summary_payload.get("indicators")
        for indicator in _coerce_mapping_sequence(indicators_raw):
            if indicator.get("name") == "classification":
                metrics_payload = _maybe_mapping(indicator.get("metrics"))
                if metrics_payload is not None:
                    return metrics_payload
        return _coerce_mapping(summary_payload.get("metrics"))

    def _adjust_from_metrics(
        self,
        metrics: Mapping[str, object],
        suggestion: Mapping[str, ParameterValue],
        search_space: Sequence[HyperparameterSpec],
    ) -> dict[str, ParameterValue]:
        adjusted = dict(suggestion)
        token_spec = self._find_spec(search_space, "max_output_tokens")
        recall = metrics.get("recall")
        precision = metrics.get("precision")
        if (
            isinstance(recall, (int, float))
            and isinstance(precision, (int, float))
            and token_spec is not None
        ):
            if recall + 0.1 < precision:
                upper_boundary = self._numeric_boundary(token_spec, "upper")
                adjusted["max_output_tokens"] = upper_boundary
            elif precision + 0.1 < recall:
                lower_boundary = self._numeric_boundary(token_spec, "lower")
                adjusted["max_output_tokens"] = lower_boundary
        return adjusted

    def _adjust_from_judge(
        self,
        missing_keywords: Iterable[str],
        suggestion: Mapping[str, ParameterValue],
        search_space: Sequence[HyperparameterSpec],
    ) -> dict[str, ParameterValue]:
        adjusted = dict(suggestion)
        missing = {keyword.lower() for keyword in missing_keywords}
        temperature_spec = self._find_spec(search_space, "temperature")
        token_spec = self._find_spec(search_space, "max_output_tokens")

        if missing & {"vector", "tooling"} and "use_tooling" in adjusted:
            adjusted["use_tooling"] = True
        if "keyword" in missing and "use_tooling" in adjusted:
            adjusted["use_tooling"] = False

        if temperature_spec is not None:
            base_temperature = float(
                adjusted.get(
                    "temperature",
                    self._midpoint_value(temperature_spec),
                ),
            )
            if "creative" in missing:
                adjusted["temperature"] = self._nudge_numeric(
                    base_temperature,
                    temperature_spec,
                    direction="increase",
                )
                base_temperature = float(adjusted["temperature"])
            if "grounded" in missing:
                adjusted["temperature"] = self._nudge_numeric(
                    base_temperature,
                    temperature_spec,
                    direction="decrease",
                )

        if token_spec is not None:
            if missing & {"report", "reports", "long", "detailed"}:
                upper_boundary = self._numeric_boundary(token_spec, "upper")
                adjusted["max_output_tokens"] = upper_boundary
            if missing & {"concise", "summary", "summaries"}:
                lower_boundary = self._numeric_boundary(token_spec, "lower")
                adjusted["max_output_tokens"] = lower_boundary

        return adjusted

    def _actions_from_judge_keywords(self, missing: Iterable[str]) -> list[str]:
        actions: list[str] = []
        missing_set = {keyword.lower() for keyword in missing}
        if missing_set & {"vector", "tooling"}:
            actions.append(
                (
                    "Integrate retrieval tooling or vector search "
                    "to satisfy judge expectations."
                ),
            )
        if "keyword" in missing_set:
            actions.append("Deliver keyword-oriented answers without heavy tooling.")
        if "creative" in missing_set:
            actions.append("Increase temperature to encourage more creative ideation.")
        if "grounded" in missing_set:
            actions.append("Lower temperature and emphasise grounded evidence usage.")
        if missing_set & {"concise", "summary", "summaries"}:
            actions.append("Tighten token budget to deliver concise summaries.")
        if missing_set & {"report", "reports", "long", "detailed"}:
            actions.append("Extend token budget to support detailed reporting outputs.")
        return actions

    @staticmethod
    def _find_spec(
        search_space: Sequence[HyperparameterSpec],
        name: str,
    ) -> HyperparameterSpec | None:
        for spec in search_space:
            if spec.name == name:
                return spec
        return None

    def _numeric_boundary(self, spec: HyperparameterSpec, which: str) -> ParameterValue:
        if spec.lower is None or spec.upper is None:
            msg = "Numeric boundary requires both lower and upper bounds"
            raise ValueError(msg)
        if spec.param_type == HyperparameterType.FLOAT:
            return float(spec.upper if which == "upper" else spec.lower)
        if spec.param_type == HyperparameterType.INT:
            boundary = int(spec.upper if which == "upper" else spec.lower)
            if spec.step:
                step = int(spec.step)
                if which == "upper":
                    lower_bound = int(spec.lower)
                    offset = boundary - lower_bound
                    steps = offset // step
                    boundary = lower_bound + steps * step
                else:
                    boundary = int(spec.lower)
            return boundary
        msg = "Numeric boundary requested for non-numeric parameter"
        raise ValueError(msg)

    def _midpoint_value(self, spec: HyperparameterSpec) -> float:
        if spec.lower is None or spec.upper is None:
            msg = "Midpoint value requires both lower and upper bounds"
            raise ValueError(msg)
        return (float(spec.lower) + float(spec.upper)) / 2

    def _nudge_numeric(
        self,
        current: float,
        spec: HyperparameterSpec,
        *,
        direction: str,
    ) -> float:
        if spec.lower is None or spec.upper is None:
            msg = "Cannot adjust numeric value without lower and upper bounds"
            raise ValueError(msg)
        step = (float(spec.upper) - float(spec.lower)) * 0.1
        if direction == "increase":
            candidate = current + step
            return round(min(candidate, float(spec.upper)), 3)
        candidate = current - step
        return round(max(candidate, float(spec.lower)), 3)


def _maybe_mapping(value: object) -> dict[str, object] | None:
    """Convert mapping-like objects to ``dict[str, object]`` copies."""
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    items = cast("Iterable[tuple[object, object]]", value.items())
    for key_obj, entry in items:
        if not isinstance(key_obj, str):
            continue
        result[key_obj] = entry
    return result


def _coerce_mapping(value: object) -> dict[str, object]:
    """Return a mapping copy with string keys when possible."""
    mapping = _maybe_mapping(value)
    return mapping or {}


def _coerce_mapping_sequence(value: object) -> tuple[dict[str, object], ...]:
    """Extract mapping entries from a heterogeneous sequence."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    collected: list[dict[str, object]] = []
    sequence = cast("Sequence[object]", value)
    for entry in sequence:
        mapping = _maybe_mapping(entry)
        if mapping is not None:
            collected.append(mapping)
    return tuple(collected)


def _coerce_str_tuple(value: object) -> tuple[str, ...]:
    """Return a tuple of non-empty string values."""
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        collected: list[str] = []
        sequence = cast("Sequence[object]", value)
        for element in sequence:
            if isinstance(element, str):
                stripped = element.strip()
                if stripped:
                    collected.append(stripped)
        return tuple(collected)
    return ()
