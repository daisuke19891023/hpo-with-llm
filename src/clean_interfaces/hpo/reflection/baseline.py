"""Baseline heuristics for reflection recommendations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from clean_interfaces.hpo.schemas import (
    HyperparameterSpec,
    HyperparameterType,
    ParameterValue,
    ReflectionInsight,
    ReflectionMode,
    TrialObservation,
)


_MIN_HISTORY_FOR_TREND = 2


def _numeric_bounds(spec: HyperparameterSpec) -> tuple[float, float] | None:
    """Return numeric bounds when both lower and upper values are defined."""
    if spec.lower is None or spec.upper is None:
        return None
    return float(spec.lower), float(spec.upper)


@dataclass(frozen=True)
class BaselineAnalysis:
    """Result of applying baseline heuristics to the trial history."""

    best_trial: TrialObservation
    latest_trial: TrialObservation
    improving: bool
    suggestion: dict[str, ParameterValue]
    base_suggestion: dict[str, ParameterValue]
    insights: tuple[ReflectionInsight, ...]
    actions: tuple[str, ...]
    summary: str
    evidence: Mapping[str, object]


class BaselineReflectionStrategy:
    """Applies metric-driven heuristics without LLM or judge input."""

    def suggest_initial(
        self, search_space: Sequence[HyperparameterSpec],
    ) -> dict[str, ParameterValue]:
        """Return balanced defaults drawn from the search space midpoints."""
        assignment: dict[str, ParameterValue] = {}
        for spec in search_space:
            match spec.param_type:
                case HyperparameterType.FLOAT:
                    bounds = _numeric_bounds(spec)
                    if bounds is None:
                        continue
                    lower, upper = bounds
                    midpoint = (lower + upper) / 2
                    assignment[spec.name] = round(midpoint, 3)
                case HyperparameterType.INT:
                    bounds = _numeric_bounds(spec)
                    if bounds is None:
                        continue
                    lower, upper = bounds
                    midpoint = (int(lower) + int(upper)) / 2
                    assignment[spec.name] = round(midpoint)
                case HyperparameterType.CATEGORICAL:
                    if spec.choices:
                        assignment[spec.name] = spec.choices[0]
                case HyperparameterType.BOOL:
                    assignment[spec.name] = False
        return assignment

    def analyse(
        self,
        history: Sequence[TrialObservation],
        search_space: Sequence[HyperparameterSpec],
        *,
        maximize: bool,
        mode: ReflectionMode,
    ) -> BaselineAnalysis:
        """Analyse the history and return heuristic recommendations."""
        best_trial = self._select_best_trial(history, maximize)
        latest_trial = history[-1]
        improving = self._is_improving(history, maximize)

        insights: list[ReflectionInsight] = [
            ReflectionInsight(
                title="Best observed trial",
                detail=(
                    f"Trial {best_trial.trial_id} achieved a score of "
                    f"{best_trial.score:.3f}."
                ),
            ),
        ]
        if latest_trial.trial_id != best_trial.trial_id:
            insights.append(
                ReflectionInsight(
                    title="Most recent trial",
                    detail=(
                        "Latest trial "
                        f"{latest_trial.trial_id} scored {latest_trial.score:.3f}; "
                        "use it to judge short-term trends."
                    ),
                ),
            )

        base_suggestion = dict(best_trial.hyperparameters)
        suggestion, extra_insights, actions = self._baseline_strategy(
            history,
            search_space,
            base_suggestion,
            maximize,
        )

        insights.extend(extra_insights)
        summary = self._compose_summary(best_trial, latest_trial, improving, mode)
        evidence = {
            "best_trial_id": best_trial.trial_id,
            "best_score": best_trial.score,
            "history_count": len(history),
        }

        return BaselineAnalysis(
            best_trial=best_trial,
            latest_trial=latest_trial,
            improving=improving,
            suggestion=suggestion,
            base_suggestion=base_suggestion,
            insights=tuple(insights),
            actions=actions,
            summary=summary,
            evidence=evidence,
        )

    def _select_best_trial(
        self, history: Sequence[TrialObservation], maximize: bool,
    ) -> TrialObservation:
        best = history[0]
        best_score = self._comparable_score(best.score, maximize)
        for observation in history[1:]:
            candidate_score = self._comparable_score(observation.score, maximize)
            if (maximize and candidate_score > best_score) or (
                not maximize and candidate_score < best_score
            ):
                best = observation
                best_score = candidate_score
        return best

    @staticmethod
    def _comparable_score(score: float, maximize: bool) -> float:
        if math.isnan(score):
            return -math.inf if maximize else math.inf
        return score

    def _is_improving(
        self, history: Sequence[TrialObservation], maximize: bool,
    ) -> bool:
        if len(history) < _MIN_HISTORY_FOR_TREND:
            return True
        prev = history[-2].score
        latest = history[-1].score
        if math.isnan(prev) or math.isnan(latest):
            return True
        return latest >= prev if maximize else latest <= prev

    def _baseline_strategy(
        self,
        history: Sequence[TrialObservation],
        search_space: Sequence[HyperparameterSpec],
        suggestion: dict[str, ParameterValue],
        maximize: bool,
    ) -> tuple[
        dict[str, ParameterValue], tuple[ReflectionInsight, ...], tuple[str, ...],
    ]:
        improving = self._is_improving(history, maximize)
        insights: list[ReflectionInsight] = []
        actions: list[str] = []
        adjusted = dict(suggestion)

        if not improving:
            insights.append(
                ReflectionInsight(
                    title="Recent regression",
                    detail=(
                        "Latest trial underperformed the previous one; pull parameters "
                        "towards safer midpoint values."
                    ),
                ),
            )
            adjusted = self._nudge_towards_midpoint(adjusted, search_space)
            actions.append(
                "Reset around midpoint values before exploring bolder deviations.",
            )
        else:
            actions.append(
                "Probe local variations around the leading trial to confirm stability.",
            )

        return adjusted, tuple(insights), tuple(actions)

    def _nudge_towards_midpoint(
        self,
        suggestion: dict[str, ParameterValue],
        search_space: Sequence[HyperparameterSpec],
    ) -> dict[str, ParameterValue]:
        adjusted = dict(suggestion)
        for spec in search_space:
            if spec.param_type == HyperparameterType.FLOAT:
                bounds = _numeric_bounds(spec)
                if bounds is None:
                    continue
                lower, upper = bounds
                midpoint = (lower + upper) / 2
                current = float(adjusted.get(spec.name, midpoint))
                adjusted[spec.name] = round((current + midpoint) / 2, 3)
            elif spec.param_type == HyperparameterType.INT:
                bounds = _numeric_bounds(spec)
                if bounds is None:
                    continue
                lower, upper = bounds
                midpoint = (int(lower) + int(upper)) / 2
                current = int(adjusted.get(spec.name, midpoint))
                adjusted[spec.name] = round((current + midpoint) / 2)
        return adjusted

    @staticmethod
    def _compose_summary(
        best: TrialObservation,
        latest: TrialObservation,
        improving: bool,
        mode: ReflectionMode,
    ) -> str:
        trend = "improving" if improving else "regressing"
        summary = (
            f"Best observed score {best.score:.3f} from trial {best.trial_id}. "
            f"Recent trend appears {trend}."
        )
        if latest.trial_id != best.trial_id:
            summary += f" Latest trial {latest.trial_id} scored {latest.score:.3f}."
        summary += f" Reflection mode: {mode.value}."
        return summary
