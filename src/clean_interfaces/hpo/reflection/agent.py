"""Reflection agent composed of modular strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from clean_interfaces.base import BaseComponent
from clean_interfaces.hpo.schemas import (
    HPOReflectionRequest,
    HPOReflectionResponse,
    HyperparameterSpec,
    ParameterValue,
    ReflectionInsight,
    ReflectionMode,
    direction_to_bool,
)
from .baseline import BaselineAnalysis, BaselineReflectionStrategy
from .judge import JudgeAugmentationResult, JudgeAugmentationStrategy
from .llm import LLMReflectionContext, LLMReflectionResult, LLMReflectionStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from clean_interfaces.llm import LLMClientFactory


class ReflectionAgent(BaseComponent):
    """Agent capable of reasoning over past trials to suggest new strategies."""

    def __init__(
        self,
        *,
        baseline_strategy: BaselineReflectionStrategy | None = None,
        judge_strategy: JudgeAugmentationStrategy | None = None,
        llm_strategy: LLMReflectionStrategy | None = None,
        llm_factory: LLMClientFactory | None = None,
    ) -> None:
        """Initialise the agent with optional strategy overrides."""
        super().__init__()
        self._baseline = baseline_strategy or BaselineReflectionStrategy()
        self._judge = judge_strategy or JudgeAugmentationStrategy()
        if llm_strategy is None:
            self._llm = LLMReflectionStrategy(factory=llm_factory)
        else:
            self._llm = llm_strategy

    def reflect(self, request: HPOReflectionRequest) -> HPOReflectionResponse:
        """Generate a reflection summary for the supplied HPO history."""
        history = list(request.history)
        if not history:
            suggestion = self._baseline.suggest_initial(request.search_space)
            insight = ReflectionInsight(
                title="Initial exploration",
                detail=(
                    "No trials have been executed; start with balanced defaults drawn "
                    "from the search space midpoints."
                ),
            )
            summary = (
                "No prior trials observed. Recommend seeding the search with balanced "
                "defaults before collecting evaluation signals."
            )
            return HPOReflectionResponse(
                mode=request.mode,
                summary=summary,
                suggested_hyperparameters=suggestion,
                insights=(insight,),
                next_actions=("Run an initial trial to gather baseline metrics.",),
                critique=(
                    "Switch to LLM-augmented reflections once judge feedback is "
                    "available."
                ),
                evidence={"history_count": 0},
            )

        baseline = self._run_baseline_strategy(request)
        suggestion = dict(baseline.suggestion)
        insights = list(baseline.insights)
        actions = list(baseline.actions)
        evidence = dict(baseline.evidence)
        critique: str | None = None

        if request.mode == ReflectionMode.LLM_AUGMENTED:
            judge_result = self._judge.augment(
                best_trial=baseline.best_trial,
                search_space=request.search_space,
                suggestion=suggestion,
            )
            suggestion = dict(judge_result.suggestion)
            insights.extend(judge_result.insights)
            actions.extend(judge_result.actions)
            evidence.update(judge_result.evidence)
            critique = judge_result.critique

            llm_result = self._invoke_llm(
                suggestion,
                request.search_space,
                baseline=baseline,
                judge_result=judge_result,
            )
            if llm_result is not None:
                suggestion = dict(llm_result.suggestion)
                insights.extend(llm_result.insights)
                actions.extend(llm_result.actions)
                evidence.update(llm_result.evidence)
                if llm_result.critique:
                    critique = llm_result.critique
        else:
            critique = (
                "Remain in metric-driven mode until richer judge signals or "
                "qualitative feedback are available."
            )

        deduped_actions = tuple(dict.fromkeys(actions))

        return HPOReflectionResponse(
            mode=request.mode,
            summary=baseline.summary,
            suggested_hyperparameters=suggestion,
            insights=tuple(insights),
            next_actions=deduped_actions,
            critique=critique,
            evidence=evidence,
        )

    def _run_baseline_strategy(self, request: HPOReflectionRequest) -> BaselineAnalysis:
        maximize = direction_to_bool(request.config.direction)
        return self._baseline.analyse(
            request.history,
            request.search_space,
            maximize=maximize,
            mode=request.mode,
        )

    def _invoke_llm(
        self,
        suggestion: dict[str, ParameterValue],
        search_space: Sequence[HyperparameterSpec],
        *,
        baseline: BaselineAnalysis,
        judge_result: JudgeAugmentationResult,
    ) -> LLMReflectionResult | None:
        context = LLMReflectionContext(
            best_trial=baseline.best_trial,
            suggestion=suggestion,
            search_space=search_space,
            metrics=judge_result.metrics,
            judge=judge_result.judge,
            plan=judge_result.plan,
        )
        return self._llm.generate(context)


__all__ = ["ReflectionAgent"]
