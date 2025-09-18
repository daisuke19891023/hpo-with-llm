"""Search backend implementations for hyperparameter optimization."""

from __future__ import annotations

import math
import random
import typing
from typing import TYPE_CHECKING, Protocol

from clean_interfaces.base import BaseComponent

from .schemas import (
    BackendTrialState,
    HPORunConfig,
    HPOSuggestedTrial,
    HPOTrialResponse,
    HyperparameterSpec,
    HyperparameterType,
    TrialObservation,
    direction_to_bool,
)

try:  # pragma: no cover - optional dependency
    import optuna  # type: ignore[import-not-found]
    from optuna.trial import TrialState  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled in runtime logic
    optuna = typing.cast(typing.Any, None)
    TrialState = typing.cast(typing.Any, None)

if TYPE_CHECKING:
    from optuna import trial as optuna_trial
    from optuna.study import Study as OptunaStudy
else:  # pragma: no cover - typing fallback when optuna is unavailable
    optuna_trial = typing.cast(typing.Any, None)
    OptunaStudy = typing.cast(typing.Any, None)


def _comparable_score(score: float, maximize: bool) -> float:
    """Convert raw scores into comparable values handling NaNs."""
    if math.isnan(score):
        return -math.inf if maximize else math.inf
    return score


def _require_numeric_bounds(spec: HyperparameterSpec) -> tuple[float, float]:
    """Validate that both lower and upper bounds are defined for a spec."""
    if spec.lower is None or spec.upper is None:
        msg = f"Hyperparameter '{spec.name}' requires lower and upper bounds"
        raise ValueError(msg)
    return spec.lower, spec.upper


def _require_choices(
    spec: HyperparameterSpec,
) -> typing.Sequence[str | bool | int | float]:
    """Validate that categorical specifications define at least one choice."""
    if not spec.choices:
        msg = f"Hyperparameter '{spec.name}' requires at least one choice"
        raise ValueError(msg)
    return spec.choices


class HPOSearchBackend(Protocol):
    """Protocol implemented by optimization backends."""

    def initialize(
        self,
        search_space: typing.Sequence[HyperparameterSpec],
        config: HPORunConfig,
    ) -> None:
        """Prepare the backend for a new optimization run."""
        ...

    def ask(self) -> HPOSuggestedTrial:
        """Return a candidate trial to evaluate."""
        ...

    def tell(self, trial_id: str, outcome: HPOTrialResponse) -> TrialObservation:
        """Record the outcome of an evaluated trial."""
        ...

    def best_trial(self) -> TrialObservation | None:
        """Return the best completed trial if available."""
        ...


class InMemorySearchBackend(BaseComponent):
    """Simple stochastic search backend for testing and defaults."""

    def __init__(self) -> None:
        """Initialize backend state."""
        super().__init__()
        self._search_space: typing.Sequence[HyperparameterSpec] | None = None
        self._config: HPORunConfig | None = None
        self._rng: random.Random | None = None
        self._next_id = 0
        self._active_trials: dict[str, BackendTrialState] = {}
        self._history: list[TrialObservation] = []
        self._maximize = True
        self._best: TrialObservation | None = None

    def initialize(
        self,
        search_space: typing.Sequence[HyperparameterSpec],
        config: HPORunConfig,
    ) -> None:
        """Prepare backend with new search space and configuration."""
        self._search_space = tuple(search_space)
        self._config = config
        # Pseudo-random sampling is sufficient for non-cryptographic HPO search.
        self._rng = random.Random(config.random_seed)  # noqa: S311
        self._next_id = 0
        self._active_trials.clear()
        self._history.clear()
        self._best = None
        self._maximize = direction_to_bool(config.direction)
        self.logger.info(
            "Initialized in-memory HPO backend",
            dimensions=len(self._search_space),
            maximize=self._maximize,
        )

    def ask(self) -> HPOSuggestedTrial:
        """Randomly sample a new configuration from the search space."""
        if self._search_space is None or self._rng is None:
            msg = "Backend must be initialized before calling ask()"
            raise RuntimeError(msg)

        params: dict[str, float | int | str | bool] = {}
        for spec in self._search_space:
            match spec.param_type:
                case HyperparameterType.FLOAT:
                    lower, upper = _require_numeric_bounds(spec)
                    params[spec.name] = self._rng.uniform(lower, upper)
                case HyperparameterType.INT:
                    lower, upper = _require_numeric_bounds(spec)
                    step = int(spec.step) if spec.step is not None else 1
                    lower_int = int(lower)
                    upper_int = int(upper)
                    choices = list(range(lower_int, upper_int + 1, step))
                    if not choices:
                        msg = (
                            "Integer hyperparameter must produce at least one candidate"
                        )
                        raise ValueError(msg)
                    params[spec.name] = self._rng.choice(choices)
                case HyperparameterType.CATEGORICAL:
                    choices = tuple(_require_choices(spec))
                    params[spec.name] = self._rng.choice(choices)
                case HyperparameterType.BOOL:
                    params[spec.name] = bool(self._rng.getrandbits(1))

        trial_id = str(self._next_id)
        self._next_id += 1

        self._active_trials[trial_id] = BackendTrialState(params=params, metadata={})
        return HPOSuggestedTrial(trial_id=trial_id, hyperparameters=params)

    def tell(self, trial_id: str, outcome: HPOTrialResponse) -> TrialObservation:
        """Record trial outcome and update best trial tracking."""
        state = self._active_trials.pop(trial_id, None)
        if state is None:
            msg = f"Unknown trial_id: {trial_id}"
            raise KeyError(msg)

        metadata = state.metadata.copy()
        metadata.update(outcome.metadata)
        observation = TrialObservation(
            trial_id=trial_id,
            hyperparameters=state.params,
            score=outcome.score,
            succeeded=outcome.succeeded,
            metadata=metadata,
            notes=outcome.notes,
        )
        self._history.append(observation)

        if len(self._history) == 1 or self._best is None:
            self._best = observation
        else:
            best_score = _comparable_score(self._best.score, self._maximize)
            candidate_score = _comparable_score(observation.score, self._maximize)
            if (self._maximize and candidate_score > best_score) or (
                not self._maximize and candidate_score < best_score
            ):
                self._best = observation

        return observation

    def best_trial(self) -> TrialObservation | None:
        """Return the best trial observed so far."""
        return getattr(self, "_best", None)


class OptunaSearchBackend(BaseComponent):
    """Search backend backed by Optuna."""

    def __init__(self, study: OptunaStudy | None = None) -> None:
        """Initialize the Optuna backend."""
        super().__init__()
        if optuna is None:  # pragma: no cover - executed when dependency missing
            msg = "Optuna is not installed. Install 'optuna' to use this backend."
            raise RuntimeError(msg)

        self._study = study
        self._search_space: typing.Sequence[HyperparameterSpec] | None = None
        self._config: HPORunConfig | None = None
        self._active_trials: dict[
            str, tuple[optuna_trial.Trial, BackendTrialState],
        ] = {}
        self._history: list[TrialObservation] = []
        self._maximize = True

    def initialize(
        self,
        search_space: typing.Sequence[HyperparameterSpec],
        config: HPORunConfig,
    ) -> None:
        """Prepare the Optuna study for a fresh run."""
        if optuna is None:  # pragma: no cover - handled above
            msg = "Optuna backend cannot be initialized without optuna installed"
            raise RuntimeError(msg)

        self._search_space = tuple(search_space)
        self._config = config
        self._history.clear()
        self._active_trials.clear()
        self._maximize = direction_to_bool(config.direction)

        sampler = (
            config.backend_options.get("sampler")
            if config.backend_options
            else None
        )
        if sampler is None and config.random_seed is not None:
            sampler = optuna.samplers.TPESampler(seed=config.random_seed)

        study_kwargs = config.backend_options.copy()
        if sampler is not None:
            study_kwargs["sampler"] = sampler

        if self._study is None:
            self._study = optuna.create_study(
                direction="maximize" if self._maximize else "minimize",
                study_name=config.study_name,
                **study_kwargs,
            )
        self.logger.info(
            "Initialized Optuna backend",
            study_name=self._study.study_name if self._study else None,
            maximize=self._maximize,
            dimensions=len(self._search_space),
        )

    def ask(self) -> HPOSuggestedTrial:
        """Sample a configuration using Optuna's suggest API."""
        if self._study is None or self._search_space is None:
            msg = "Backend must be initialized before calling ask()"
            raise RuntimeError(msg)

        trial = self._study.ask()
        params: dict[str, float | int | str | bool] = {}
        for spec in self._search_space:
            match spec.param_type:
                case HyperparameterType.FLOAT:
                    lower, upper = _require_numeric_bounds(spec)
                    params[spec.name] = trial.suggest_float(
                        spec.name,
                        lower,
                        upper,
                        log=spec.log,
                        step=spec.step,
                    )
                case HyperparameterType.INT:
                    lower, upper = _require_numeric_bounds(spec)
                    params[spec.name] = trial.suggest_int(
                        spec.name,
                        int(lower),
                        int(upper),
                        step=int(spec.step) if spec.step is not None else 1,
                        log=spec.log,
                    )
                case HyperparameterType.CATEGORICAL:
                    choices = tuple(_require_choices(spec))
                    params[spec.name] = trial.suggest_categorical(
                        spec.name,
                        choices,
                    )
                case HyperparameterType.BOOL:
                    params[spec.name] = trial.suggest_categorical(
                        spec.name,
                        (True, False),
                    )

        trial_id = str(trial.number)
        self._active_trials[trial_id] = (
            trial,
            BackendTrialState(params=params, metadata={}),
        )
        return HPOSuggestedTrial(trial_id=trial_id, hyperparameters=params)

    def tell(self, trial_id: str, outcome: HPOTrialResponse) -> TrialObservation:
        """Finalize a trial within the Optuna study."""
        entry = self._active_trials.pop(trial_id, None)
        if entry is None:
            msg = f"Unknown trial_id: {trial_id}"
            raise KeyError(msg)
        trial, state = entry

        if outcome.metadata:
            for key, value in outcome.metadata.items():
                trial.set_user_attr(key, value)

        study = self._study
        if study is None:
            msg = "Backend must be initialized before calling tell()"
            raise RuntimeError(msg)

        if outcome.succeeded:
            study.tell(trial, outcome.score)
        else:
            study.tell(trial, state=TrialState.FAIL)

        observation = TrialObservation(
            trial_id=trial_id,
            hyperparameters=state.params,
            score=outcome.score,
            succeeded=outcome.succeeded,
            metadata=state.metadata | outcome.metadata,
            notes=outcome.notes,
        )
        self._history.append(observation)
        return observation

    def best_trial(self) -> TrialObservation | None:
        """Return the best observed trial according to the configured direction."""
        if not self._history:
            return None

        best = self._history[0]
        best_score = _comparable_score(best.score, self._maximize)
        for observation in self._history[1:]:
            candidate_score = _comparable_score(observation.score, self._maximize)
            if (self._maximize and candidate_score > best_score) or (
                not self._maximize and candidate_score < best_score
            ):
                best = observation
                best_score = candidate_score
        return best


__all__ = [
    "HPOSearchBackend",
    "InMemorySearchBackend",
    "OptunaSearchBackend",
]
