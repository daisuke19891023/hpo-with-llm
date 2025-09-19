"""Tests for the HPO orchestrator."""

from collections.abc import Sequence

import pytest

from clean_interfaces.hpo.backends import InMemorySearchBackend
from clean_interfaces.hpo.orchestrator import HPOOrchestrator
from clean_interfaces.hpo.schemas import (
    CodingTask,
    HPOOptimizationResult,
    HPOTrialRequest,
    HPOTrialResponse,
    HPORunConfig,
    HyperparameterSpec,
    HyperparameterType,
    TrialObservation,
)


def _default_search_space() -> Sequence[HyperparameterSpec]:
    """Return a canonical search space used across tests."""
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
            upper=256,
            step=64,
        ),
        HyperparameterSpec(
            name="use_tooling",
            param_type=HyperparameterType.BOOL,
        ),
    )


class TestHPOOrchestrator:
    """Unit tests covering orchestration behaviour."""

    def test_runs_trials_and_returns_best(self) -> None:
        """The orchestrator should evaluate the expected number of trials."""
        backend = InMemorySearchBackend()
        orchestrator = HPOOrchestrator(
            search_backend=backend,
            trial_executor=_scoring_executor,
        )
        task = CodingTask(task_id="unit", description="Tune prompt settings")
        config = HPORunConfig(max_trials=4, random_seed=11)

        result = orchestrator.optimize(task, _default_search_space(), config)

        assert isinstance(result, HPOOptimizationResult)
        assert len(result.trials) == config.max_trials
        assert result.best_trial is not None
        assert result.best_trial.succeeded is True
        assert "temperature" in result.best_trial.hyperparameters

    def test_history_is_provided_to_executor(self) -> None:
        """Each trial request should include previous outcomes in history."""
        history_lengths: list[int] = []

        def executor(request: HPOTrialRequest) -> HPOTrialResponse:
            history_lengths.append(len(request.history))
            return HPOTrialResponse(score=1.0)

        backend = InMemorySearchBackend()
        orchestrator = HPOOrchestrator(
            search_backend=backend,
            trial_executor=executor,
        )
        task = CodingTask(task_id="history", description="Validate history propagation")

        orchestrator.optimize(
            task,
            _default_search_space(),
            HPORunConfig(max_trials=3, random_seed=3),
        )

        assert history_lengths == [0, 1, 2]

    def test_direction_controls_best_selection(self) -> None:
        """Best trial selection should respect optimization direction."""
        scores = [0.5, 0.2, 0.7]

        def executor(request: HPOTrialRequest) -> HPOTrialResponse:
            idx = len(request.history)
            return HPOTrialResponse(score=scores[idx])

        task = CodingTask(task_id="direction", description="Check direction handling")

        minimize_backend = InMemorySearchBackend()
        minimizer = HPOOrchestrator(
            search_backend=minimize_backend,
            trial_executor=executor,
        )
        minimize_result = minimizer.optimize(
            task,
            _default_search_space(),
            HPORunConfig(max_trials=3, random_seed=5, direction="minimize"),
        )

        assert minimize_result.best_trial is not None
        assert minimize_result.best_trial.score == pytest.approx(0.2)

        maximize_backend = InMemorySearchBackend()
        maximizer = HPOOrchestrator(
            search_backend=maximize_backend,
            trial_executor=executor,
        )
        maximize_result = maximizer.optimize(
            task,
            _default_search_space(),
            HPORunConfig(max_trials=3, random_seed=5, direction="maximize"),
        )

        assert maximize_result.best_trial is not None
        assert maximize_result.best_trial.score == pytest.approx(0.7)

    def test_trial_logger_is_invoked(self) -> None:
        """Configured trial loggers should receive each completed trial."""

        class RecordingLogger:
            def __init__(self) -> None:
                self.records: list[str] = []

            def record(
                self,
                *,
                request: HPOTrialRequest,
                response: HPOTrialResponse,
                observation: TrialObservation,
            ) -> None:
                assert request
                assert response
                self.records.append(observation.trial_id)

        backend = InMemorySearchBackend()
        logger = RecordingLogger()
        orchestrator = HPOOrchestrator(
            search_backend=backend,
            trial_executor=_scoring_executor,
            trial_logger=logger,
        )

        task = CodingTask(task_id="logged", description="Ensure logging")
        config = HPORunConfig(max_trials=2, random_seed=7)

        orchestrator.optimize(task, _default_search_space(), config)

        assert len(logger.records) == config.max_trials


def _scoring_executor(request: HPOTrialRequest) -> HPOTrialResponse:
    """Compute a deterministic score favouring mid temperatures and more tokens."""
    params = request.trial.hyperparameters
    temperature = float(params.get("temperature", 0.5))
    max_tokens = int(params.get("max_output_tokens", 256))

    score = 1.0 - abs(temperature - 0.3)
    score += max_tokens / 1024.0

    return HPOTrialResponse(score=score)
