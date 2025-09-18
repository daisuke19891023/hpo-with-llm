"""High-level orchestration for hyperparameter optimization workflows."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Protocol

from clean_interfaces.base import BaseComponent

from clean_interfaces.hpo.schemas import (
    CodingTask,
    HPOOptimizationResult,
    HPOTrialRequest,
    HPOTrialResponse,
    HPORunConfig,
    HPOSuggestedTrial,
    HyperparameterSpec,
    TrialObservation,
    direction_to_bool,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .backends import HPOSearchBackend
    from .logging import HPOTrialLogger


class TrialExecutorProtocol(Protocol):
    """Callable protocol executed for each HPO trial."""

    def __call__(self, request: HPOTrialRequest) -> HPOTrialResponse:
        """Execute an HPO trial and return the evaluation response."""
        ...


class HPOOrchestrator(BaseComponent):
    """Coordinates the interaction between search backends and executors."""

    def __init__(
        self,
        *,
        search_backend: HPOSearchBackend,
        trial_executor: TrialExecutorProtocol,
        trial_logger: HPOTrialLogger | None = None,
    ) -> None:
        """Initialize the orchestrator with backend and executor."""
        super().__init__()
        self._backend = search_backend
        self._trial_executor = trial_executor
        self._trial_logger = trial_logger

    def optimize(
        self,
        task: CodingTask,
        search_space: Sequence[HyperparameterSpec],
        config: HPORunConfig | None = None,
    ) -> HPOOptimizationResult:
        """Run an HPO experiment and return the aggregated result."""
        active_config = config or HPORunConfig()
        self._backend.initialize(search_space, active_config)
        history: list[TrialObservation] = []
        maximize = direction_to_bool(active_config.direction)

        self.logger.info(
            "Starting HPO run",
            task_id=task.task_id,
            max_trials=active_config.max_trials,
            direction=active_config.direction,
        )

        for _ in range(active_config.max_trials):
            suggestion = self._backend.ask()
            request = self._build_request(
                task=task,
                suggestion=suggestion,
                search_space=search_space,
                history=history,
            )

            try:
                response = self._trial_executor(request)
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.logger.error(
                    "Trial executor raised exception",
                    trial_id=suggestion.trial_id,
                    error=str(exc),
                )
                response = self._failure_response(maximize=maximize, error=str(exc))

            observation = self._backend.tell(suggestion.trial_id, response)
            history.append(observation)

            self.logger.info(
                "Completed HPO trial",
                trial_id=observation.trial_id,
                score=observation.score,
                succeeded=observation.succeeded,
            )

            if self._trial_logger is not None:
                try:
                    self._call_trial_logger(
                        request=request,
                        response=response,
                        observation=observation,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging path
                    self.logger.warning(
                        "Failed to log trial",
                        trial_id=observation.trial_id,
                        error=str(exc),
                    )

        best = self._backend.best_trial()
        self.logger.info(
            "Finished HPO run",
            trials=len(history),
            best_trial=best.trial_id if best else None,
        )
        return HPOOptimizationResult(task=task, trials=history, best_trial=best)

    def _call_trial_logger(
        self,
        *,
        request: HPOTrialRequest,
        response: HPOTrialResponse,
        observation: TrialObservation,
    ) -> None:
        """Invoke the configured trial logger with compatible keyword arguments."""
        logger = self._trial_logger
        if logger is None:  # pragma: no cover - guarded by caller
            message = "Trial logger is not configured"
            raise RuntimeError(message)

        parameters = inspect.signature(logger.record).parameters
        dispatch_map: dict[str, object] = {
            "request": request,
            "_request": request,
            "response": response,
            "_response": response,
            "observation": observation,
        }
        kwargs = {
            name: value for name, value in dispatch_map.items() if name in parameters
        }
        if "observation" not in kwargs:
            kwargs["observation"] = observation
        logger.record(**kwargs)  # type: ignore[arg-type]

    def _build_request(
        self,
        *,
        task: CodingTask,
        suggestion: HPOSuggestedTrial,
        search_space: Sequence[HyperparameterSpec],
        history: Sequence[TrialObservation],
    ) -> HPOTrialRequest:
        """Construct the structured request for the trial executor."""
        return HPOTrialRequest(
            task=task,
            trial=suggestion,
            search_space=tuple(search_space),
            history=tuple(history),
        )

    def _failure_response(self, *, maximize: bool, error: str) -> HPOTrialResponse:
        """Create a sentinel response when execution fails unexpectedly."""
        score = float("-inf") if maximize else float("inf")
        return HPOTrialResponse(
            score=score,
            succeeded=False,
            metadata={"error": error},
            notes="Trial failed before completion",
        )


__all__ = ["HPOOrchestrator", "TrialExecutorProtocol"]
