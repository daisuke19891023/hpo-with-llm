"""Pydantic schemas for hyperparameter optimization orchestration."""

from __future__ import annotations

import pathlib
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

if TYPE_CHECKING:
    from pathlib import Path
else:
    Path = pathlib.Path

ParameterValue = bool | int | float | str


class ParameterLocationKind(str, Enum):
    """Supported mechanisms for applying hyperparameters during trials."""

    ENVIRONMENT = "environment"
    CLI_ARGUMENT = "cli_argument"
    FILE_CONTENT = "file_content"
    CONFIG_YAML = "config_yaml"


class _BaseParameterLocation(BaseModel):
    """Shared configuration for parameter location models."""

    description: str | None = Field(
        default=None,
        description="Optional human readable hint for how the value is applied.",
    )

    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class EnvironmentParameterLocation(_BaseParameterLocation):
    """Apply a hyperparameter by exporting an environment variable."""

    kind: Literal["environment"] = Field(
        default=ParameterLocationKind.ENVIRONMENT.value,
    )
    variable: str = Field(
        description="Name of the environment variable that will receive the value.",
    )


class CLIArgumentParameterLocation(_BaseParameterLocation):
    """Apply a hyperparameter via command line arguments."""

    kind: Literal["cli_argument"] = Field(
        default=ParameterLocationKind.CLI_ARGUMENT.value,
    )
    flag: str | None = Field(
        default=None,
        description=(
            "Optional flag or option name to prefix the value with when building "
            "command line arguments."
        ),
    )
    value_template: str | None = Field(
        default=None,
        description=(
            "Optional format string used to render the value. Use '{value}' as the "
            "placeholder. When omitted the raw value is stringified."
        ),
    )


class FileContentParameterLocation(_BaseParameterLocation):
    """Apply a hyperparameter by writing it to a file."""

    kind: Literal["file_content"] = Field(
        default=ParameterLocationKind.FILE_CONTENT.value,
    )
    path: Path = Field(
        description="Path to the file that should contain the value.",
    )
    encoding: str = Field(
        default="utf-8",
        description="Text encoding to use when writing the file.",
    )


class YAMLConfigParameterLocation(_BaseParameterLocation):
    """Apply a hyperparameter by updating a YAML configuration document."""

    kind: Literal["config_yaml"] = Field(
        default=ParameterLocationKind.CONFIG_YAML.value,
    )
    path: Path = Field(
        description="Path to the YAML configuration file that should be updated.",
    )
    key_path: str = Field(
        description=(
            "Dot separated path to the configuration field that should be "
            "overwritten with the tuned value."
        ),
    )
    encoding: str = Field(
        default="utf-8",
        description="Text encoding used when persisting the YAML file.",
    )


ParameterLocation = Annotated[
    EnvironmentParameterLocation
    | CLIArgumentParameterLocation
    | FileContentParameterLocation
    | YAMLConfigParameterLocation,
    Field(discriminator="kind"),
]


class HyperparameterType(str, Enum):
    """Supported hyperparameter data types."""

    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    BOOL = "bool"


class HyperparameterSpec(BaseModel):
    """Definition of an individual hyperparameter search dimension."""

    name: str = Field(description="Unique identifier for the hyperparameter")
    param_type: HyperparameterType = Field(
        description="Type of hyperparameter controlling available options",
    )
    description: str | None = Field(
        default=None,
        description="Optional human-readable summary of the parameter.",
    )
    lower: float | None = Field(
        default=None,
        description="Lower bound for numeric parameters",
    )
    upper: float | None = Field(
        default=None,
        description="Upper bound for numeric parameters",
    )
    step: float | None = Field(
        default=None,
        description="Step size for integer or discretized parameters",
    )
    log: bool = Field(
        default=False,
        description="Whether to sample the parameter on a log scale",
    )
    choices: list[ParameterValue] | None = Field(
        default=None,
        description="Enumerated options for categorical parameters",
    )
    location: ParameterLocation | None = Field(
        default=None,
        description=(
            "Instructions describing how this parameter is applied during trial "
            "execution."
        ),
    )

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("lower")
    @classmethod
    def _validate_lower(
        cls,
        value: float | None,
        info: ValidationInfo,
    ) -> float | None:
        """Ensure numeric parameters include a lower bound."""
        numeric_types = {HyperparameterType.FLOAT, HyperparameterType.INT}
        param_type = info.data.get("param_type")
        if param_type in numeric_types and value is None:
            msg = "Numeric parameters require a lower bound"
            raise ValueError(msg)
        return value

    @field_validator("upper")
    @classmethod
    def _validate_upper(
        cls,
        value: float | None,
        info: ValidationInfo,
    ) -> float | None:
        """Ensure numeric parameters include an upper bound."""
        numeric_types = {HyperparameterType.FLOAT, HyperparameterType.INT}
        param_type = info.data.get("param_type")
        if param_type in numeric_types:
            if value is None:
                msg = "Numeric parameters require an upper bound"
                raise ValueError(msg)
            lower = info.data.get("lower")
            if lower is not None and value <= lower:
                msg = "Upper bound must be greater than lower bound"
                raise ValueError(msg)
        return value

    @field_validator("choices")
    @classmethod
    def _validate_choices(
        cls,
        value: list[ParameterValue] | None,
        info: ValidationInfo,
    ) -> list[ParameterValue] | None:
        """Ensure categorical parameters declare their options."""
        if (
            info.data.get("param_type") == HyperparameterType.CATEGORICAL
            and not value
        ):
            msg = "Categorical parameters require at least one choice"
            raise ValueError(msg)
        return value

    @field_validator("step")
    @classmethod
    def _validate_step(
        cls,
        value: float | None,
        info: ValidationInfo,
    ) -> float | None:
        """Ensure integer parameters have an integer step size."""
        is_integer_param = info.data.get("param_type") == HyperparameterType.INT
        if is_integer_param and value is not None and not float(value).is_integer():
            msg = "Integer parameters must use integer step sizes"
            raise ValueError(msg)
        return value


class CodingTask(BaseModel):
    """Describes the target coding task being optimized."""

    task_id: str = Field(description="Unique identifier for the task")
    description: str = Field(description="Plain language description of the task")
    acceptance_criteria: str | None = Field(
        default=None,
        description="Optional acceptance criteria for success",
    )
    additional_context: dict[str, str] = Field(
        default_factory=dict,
        description="Supplementary context or metadata for the task",
    )


class HPORunConfig(BaseModel):
    """Configuration controlling an HPO experiment."""

    max_trials: int = Field(
        default=5,
        gt=0,
        description="Maximum number of trials to execute",
    )
    direction: str = Field(
        default="maximize",
        pattern="^(minimize|maximize)$",
        description="Whether scores should be minimized or maximized",
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Optional timeout for the optimization run",
    )
    random_seed: int | None = Field(
        default=None,
        description="Optional random seed passed to stochastic search backends",
    )
    study_name: str | None = Field(
        default=None,
        description="Optional study identifier for Optuna-based backends",
    )
    backend_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific keyword arguments",
    )


class TrialObservation(BaseModel):
    """Record of a completed trial."""

    trial_id: str = Field(description="Backend-specific trial identifier")
    hyperparameters: dict[str, ParameterValue] = Field(
        description="Evaluated hyperparameter values",
    )
    score: float = Field(description="Observed objective value for the trial")
    succeeded: bool = Field(
        default=True,
        description="Indicates whether the trial completed successfully",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Supplementary metadata recorded for the trial",
    )
    notes: str | None = Field(
        default=None,
        description="Optional human-readable notes about the outcome",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HPOSuggestedTrial(BaseModel):
    """Candidate trial proposed by the search backend."""

    trial_id: str = Field(description="Backend-specific identifier for the trial")
    hyperparameters: dict[str, ParameterValue] = Field(
        description="Hyperparameter assignments for evaluation",
    )


class HPOTrialRequest(BaseModel):
    """Structured payload sent to an LLM/tooling interface for evaluation."""

    task: CodingTask = Field(description="Description of the coding task to evaluate")
    trial: HPOSuggestedTrial = Field(
        description="Candidate hyperparameters to evaluate",
    )
    search_space: typing.Sequence[HyperparameterSpec] = Field(
        description="Hyperparameter search space definition",
    )
    history: tuple[TrialObservation, ...] = Field(
        default_factory=tuple,
        description="Prior trial results shared with the executor",
    )


class HPOTrialResponse(BaseModel):
    """Structured response returned by the executor after evaluation."""

    score: float = Field(description="Measured objective value for the candidate")
    succeeded: bool = Field(
        default=True,
        description="Indicates whether the evaluation completed successfully",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from the executor",
    )
    notes: str | None = Field(
        default=None,
        description="Optional human-readable summary of the evaluation",
    )


class HPOOptimizationResult(BaseModel):
    """Aggregate result of an HPO experiment."""

    task: CodingTask = Field(description="Task description associated with the run")
    trials: list[TrialObservation] = Field(
        description="Chronological record of observed trials",
    )
    best_trial: TrialObservation | None = Field(
        default=None,
        description="Best-performing trial if available",
    )


class HPOExecutionRequest(BaseModel):
    """Request payload for API or CLI triggered HPO runs."""

    task: CodingTask = Field(description="Task to optimize")
    search_space: list[HyperparameterSpec] = Field(
        description="Hyperparameter search space to explore",
    )
    config: HPORunConfig = Field(
        default_factory=HPORunConfig,
        description="Optimization configuration",
    )


class HPOExecutionResponse(HPOOptimizationResult):
    """Response payload for API-driven HPO runs."""



class ReflectionMode(str, Enum):
    """Operating modes supported by the reflection agent."""

    BASELINE = "baseline"
    LLM_AUGMENTED = "llm_augmented"


class ReflectionInsight(BaseModel):
    """Structured insight surfaced by the reflection agent."""

    title: str = Field(description="Short label describing the insight")
    detail: str = Field(description="Explanation of the observation or takeaway")


class HPOReflectionRequest(BaseModel):
    """Request payload for prompting the reflection agent."""

    task: CodingTask = Field(description="Task the HPO run targets")
    config: HPORunConfig = Field(description="Optimization configuration in effect")
    search_space: typing.Sequence[HyperparameterSpec] = Field(
        description="Hyperparameter definitions explored during optimization",
    )
    history: typing.Sequence[TrialObservation] = Field(
        default_factory=tuple,
        description="Chronological record of observed trials",
    )
    mode: ReflectionMode = Field(
        default=ReflectionMode.BASELINE,
        description="Reflection strategy controlling reasoning depth",
    )


class HPOReflectionResponse(BaseModel):
    """Response payload returned by the reflection agent."""

    mode: ReflectionMode = Field(
        description="Reflection strategy that produced the output",
    )
    summary: str = Field(description="High-level summary of the recommendation")
    suggested_hyperparameters: dict[str, ParameterValue] = Field(
        description="Candidate hyperparameters proposed for the next trial",
    )
    insights: tuple[ReflectionInsight, ...] = Field(
        default_factory=tuple,
        description="Collection of notable observations surfaced by the agent",
    )
    next_actions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Actionable follow-ups recommended by the agent",
    )
    critique: str | None = Field(
        default=None,
        description="Optional critique or caution raised by the agent",
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured evidence supporting the reflection output",
    )


def direction_to_bool(direction: str) -> bool:
    """Return True when optimization direction is maximize."""
    return direction == "maximize"


@dataclass(slots=True)
class BackendTrialState:
    """Internal data tracked for active backend trials."""

    params: dict[str, ParameterValue]
    metadata: dict[str, Any]
