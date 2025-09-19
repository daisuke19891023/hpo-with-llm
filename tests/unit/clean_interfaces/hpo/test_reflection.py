"""Unit tests for the reflection agent."""

from typing import cast

import pytest

from clean_interfaces.llm.base import LLMClient, LLMGenerationResult
from clean_interfaces.llm.factory import LLMClientFactory
from clean_interfaces.hpo.reflection import ReflectionAgent
from clean_interfaces.hpo.schemas import (
    CodingTask,
    HPOReflectionRequest,
    HPORunConfig,
    HyperparameterSpec,
    HyperparameterType,
    ReflectionMode,
    TrialObservation,
)
from clean_interfaces.utils.settings import LLMProvider


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
            upper=512,
            step=64,
        ),
        HyperparameterSpec(
            name="use_tooling",
            param_type=HyperparameterType.BOOL,
        ),
    )


def test_reflection_agent_returns_defaults_without_history() -> None:
    """The agent should fall back to balanced defaults when history is empty."""
    agent = ReflectionAgent()
    request = HPOReflectionRequest(
        task=CodingTask(task_id="demo", description="Tune RAG prompts"),
        config=HPORunConfig(max_trials=3),
        search_space=_search_space(),
        history=(),
        mode=ReflectionMode.BASELINE,
    )

    response = agent.reflect(request)

    assert response.mode is ReflectionMode.BASELINE
    assert "No prior trials observed" in response.summary
    assert response.suggested_hyperparameters["temperature"] == pytest.approx(
        0.5, abs=1e-3,
    )
    assert response.next_actions


def test_reflection_agent_prefers_best_trial_when_improving() -> None:
    """Baseline mode should reuse the best trial hyperparameters when trend improves."""
    history = (
        TrialObservation(
            trial_id="0",
            hyperparameters={
                "temperature": 0.3,
                "max_output_tokens": 192,
                "use_tooling": False,
            },
            score=0.42,
            succeeded=True,
        ),
        TrialObservation(
            trial_id="1",
            hyperparameters={
                "temperature": 0.6,
                "max_output_tokens": 320,
                "use_tooling": True,
            },
            score=0.71,
            succeeded=True,
        ),
    )

    agent = ReflectionAgent()
    request = HPOReflectionRequest(
        task=CodingTask(task_id="demo", description="Tune RAG prompts"),
        config=HPORunConfig(max_trials=4),
        search_space=_search_space(),
        history=history,
        mode=ReflectionMode.BASELINE,
    )

    response = agent.reflect(request)

    assert response.mode is ReflectionMode.BASELINE
    assert response.suggested_hyperparameters["temperature"] == pytest.approx(
        0.6, abs=1e-6,
    )
    assert response.suggested_hyperparameters["max_output_tokens"] == 320
    assert response.suggested_hyperparameters["use_tooling"] is True


def test_reflection_agent_llm_mode_applies_judge_feedback() -> None:
    """LLM mode should leverage judge metadata to adjust recommendations."""
    history = (
        TrialObservation(
            trial_id="best",
            hyperparameters={
                "temperature": 0.4,
                "max_output_tokens": 256,
                "use_tooling": False,
            },
            score=0.68,
            succeeded=True,
            metadata={
                "evaluation": {
                    "summary": {
                        "indicators": [
                            {
                                "name": "classification",
                                "metrics": {
                                    "accuracy": 0.72,
                                    "precision": 0.81,
                                    "recall": 0.55,
                                    "f1": 0.65,
                                },
                                "score_components": [0.72, 0.81, 0.55, 0.65],
                            },
                        ],
                        "judge": {
                            "rationale": (
                                "Introduce vector tooling to cover missing keywords."
                            ),
                            "missing_keywords": ["vector"],
                            "matched_keywords": ["grounded"],
                            "score": 0.7,
                        },
                        "composite_score": 0.68,
                    },
                    "plan": "Operate conservatively while grounding responses.",
                },
            },
        ),
        TrialObservation(
            trial_id="recent",
            hyperparameters={
                "temperature": 0.55,
                "max_output_tokens": 192,
                "use_tooling": False,
            },
            score=0.4,
            succeeded=True,
        ),
    )

    agent = ReflectionAgent()
    request = HPOReflectionRequest(
        task=CodingTask(task_id="demo", description="Tune RAG prompts"),
        config=HPORunConfig(max_trials=4),
        search_space=_search_space(),
        history=history,
        mode=ReflectionMode.LLM_AUGMENTED,
    )

    response = agent.reflect(request)

    assert response.mode is ReflectionMode.LLM_AUGMENTED
    assert response.suggested_hyperparameters["use_tooling"] is True
    assert response.suggested_hyperparameters["max_output_tokens"] == 512
    assert response.critique == "Introduce vector tooling to cover missing keywords."
    assert any("Missing keywords" in insight.detail for insight in response.insights)


def test_reflection_agent_llm_mode_uses_llm_client() -> None:
    """LLM augmented mode should incorporate structured output from the client."""
    history = (
        TrialObservation(
            trial_id="best",
            hyperparameters={
                "temperature": 0.4,
                "max_output_tokens": 256,
                "use_tooling": False,
            },
            score=0.7,
            succeeded=True,
        ),
    )

    class StubClient:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def generate(self, request: object) -> LLMGenerationResult:
            self.requests.append(request)
            return LLMGenerationResult(
                response_id="resp",
                model="model",
                text=None,
                structured_output={
                    "hyperparameters": {
                        "use_tooling": True,
                        "temperature": 0.65,
                    },
                    "insights": [
                        {
                            "title": "LLM",
                            "detail": "Vector search improves grounding.",
                        },
                    ],
                    "actions": ["Enable vector search pipeline"],
                    "critique": "Favour retrieval coverage.",
                },
                function_call=None,
                usage={},
                raw={},
            )

    stub_client = StubClient()

    class StubFactory(LLMClientFactory):
        def __init__(self) -> None:
            super().__init__(settings=None)

        def create(
            self,
            provider: LLMProvider | str | None = None,
        ) -> LLMClient:
            _ = provider
            return cast("LLMClient", stub_client)

    agent = ReflectionAgent(llm_factory=StubFactory())
    request = HPOReflectionRequest(
        task=CodingTask(task_id="demo", description="Tune RAG prompts"),
        config=HPORunConfig(max_trials=3),
        search_space=_search_space(),
        history=history,
        mode=ReflectionMode.LLM_AUGMENTED,
    )

    response = agent.reflect(request)

    assert stub_client.requests
    assert response.suggested_hyperparameters["use_tooling"] is True
    assert response.suggested_hyperparameters["temperature"] == pytest.approx(
        0.65, abs=1e-6,
    )
    assert any("Vector search" in insight.detail for insight in response.insights)
    assert "retrieval" in (response.critique or "").lower()
