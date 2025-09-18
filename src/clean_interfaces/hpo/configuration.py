"""Utilities for loading and applying hyperparameter tuning configurations."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from collections.abc import Mapping, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .schemas import (
    CLIArgumentParameterLocation,
    EnvironmentParameterLocation,
    FileContentParameterLocation,
    HyperparameterSpec,
    HyperparameterType,
    ParameterLocation,
    ParameterValue,
    YAMLConfigParameterLocation,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class TuningParameterDefinition(BaseModel):
    """Definition of a tunable parameter loaded from YAML configuration."""

    name: str
    param_type: HyperparameterType = Field(alias="type")
    lower: float | None = None
    upper: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[ParameterValue] | None = None
    description: str | None = None
    location: ParameterLocation | None = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_search_fields(self) -> TuningParameterDefinition:
        """Reuse the HyperparameterSpec validators for consistency."""
        # Instantiating HyperparameterSpec will trigger validation
        HyperparameterSpec(
            name=self.name,
            param_type=self.param_type,
            description=self.description,
            lower=self.lower,
            upper=self.upper,
            step=self.step,
            log=self.log,
            choices=self.choices,
            location=self.location,
        )
        return self

    def to_spec(self) -> HyperparameterSpec:
        """Convert the definition into a HyperparameterSpec instance."""
        return HyperparameterSpec(
            name=self.name,
            param_type=self.param_type,
            description=self.description,
            lower=self.lower,
            upper=self.upper,
            step=self.step,
            log=self.log,
            choices=self.choices,
            location=self.location,
        )


def _empty_parameter_definitions() -> list[TuningParameterDefinition]:
    """Return an empty parameter definition list with precise typing."""
    return []


class TuningConfig(BaseModel):
    """Container for a collection of tunable parameter definitions."""

    parameters: list[TuningParameterDefinition] = Field(
        default_factory=_empty_parameter_definitions,
    )

    model_config = ConfigDict(extra="forbid")

    def to_search_space(self) -> list[HyperparameterSpec]:
        """Convert the configuration into a HyperparameterSpec sequence."""
        return [parameter.to_spec() for parameter in self.parameters]

    def describe(self) -> list[dict[str, Any]]:
        """Return a user-friendly description of the tuning parameters."""
        descriptions: list[dict[str, Any]] = []
        for parameter in self.parameters:
            spec = parameter.to_spec()
            type_value = str(spec.param_type)
            descriptions.append(
                {
                    "name": spec.name,
                    "type": type_value,
                    "description": spec.description,
                    "location": spec.location.model_dump() if spec.location else None,
                },
            )
        return descriptions


def load_tuning_config(path: str | Path) -> TuningConfig:
    """Load tuning configuration definitions from a YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        msg = f"Tuning configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)

    content: dict[str, Any]
    if isinstance(loaded, Mapping):
        mapping = cast("Mapping[str, Any]", loaded)
        content = dict(mapping)
    else:
        content = {}

    try:
        return TuningConfig.model_validate(content)
    except ValidationError as exc:  # pragma: no cover - exercised via unit tests
        msg = f"Invalid tuning configuration: {exc}"  # pragma: no cover
        raise ValueError(msg) from exc  # pragma: no cover


def default_tuning_config() -> TuningConfig:
    """Provide a default configuration for quick experimentation."""
    return TuningConfig(
        parameters=[
            TuningParameterDefinition(
                name="temperature",
                type=HyperparameterType.FLOAT,
                lower=0.0,
                upper=1.0,
                description="Sampling temperature controlling creativity",
                location=EnvironmentParameterLocation(variable="LLM_TEMPERATURE"),
            ),
            TuningParameterDefinition(
                name="max_output_tokens",
                type=HyperparameterType.INT,
                lower=64,
                upper=512,
                step=64,
                description="Maximum number of tokens produced by the model",
                location=CLIArgumentParameterLocation(
                    flag="--max-output-tokens",
                    value_template="{value}",
                ),
            ),
            TuningParameterDefinition(
                name="retrieval_strategy",
                type=HyperparameterType.CATEGORICAL,
                choices=["keyword", "vector", "hybrid"],
                description="Retrieval approach used by the RAG pipeline",
                location=YAMLConfigParameterLocation(
                    path=Path("rag_pipeline.yaml"),
                    key_path="retrieval.strategy",
                ),
            ),
        ],
    )


@dataclass(slots=True)
class YAMLKeyUpdate:
    """Instruction describing how to update a YAML document."""

    key_path: tuple[str, ...]
    value: ParameterValue


def _empty_yaml_key_updates() -> list[YAMLKeyUpdate]:
    """Return an empty list for YAML key updates."""
    return []


@dataclass(slots=True)
class YAMLUpdateInstruction:
    """Collection of YAML key updates for a single file."""

    path: Path
    encoding: str
    updates: list[YAMLKeyUpdate] = field(default_factory=_empty_yaml_key_updates)


@dataclass(slots=True)
class FileWriteInstruction:
    """Instruction describing a file write required for a trial."""

    path: Path
    content: str
    encoding: str


@dataclass(slots=True)
class ParameterApplicationPlan:
    """Operations required to apply hyperparameter assignments."""

    parameter_values: dict[str, ParameterValue]
    environment: dict[str, str]
    cli_arguments: list[str]
    file_writes: list[FileWriteInstruction]
    yaml_updates: list[YAMLUpdateInstruction]

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the plan into metadata suitable for logging."""
        file_writes = [
            {"path": str(instruction.path), "encoding": instruction.encoding}
            for instruction in self.file_writes
        ]
        return {
            "parameter_values": self.parameter_values,
            "environment": self.environment,
            "cli_arguments": self.cli_arguments,
            "file_writes": file_writes,
            "yaml_updates": self._describe_yaml_updates(),
        }

    def apply(
        self,
        *,
        commit: bool = False,
        base_environment: Mapping[str, str] | None = None,
        base_cli_arguments: Sequence[str] | None = None,
    ) -> ParameterApplicationResult:
        """Apply the plan to produce a runnable trial configuration."""
        environment = dict(base_environment or {})
        environment.update(self.environment)

        cli_arguments = list(base_cli_arguments or [])
        cli_arguments.extend(self.cli_arguments)

        file_contents: dict[Path, str] = {}
        for instruction in self.file_writes:
            file_contents[instruction.path] = instruction.content
            if commit:
                instruction.path.parent.mkdir(parents=True, exist_ok=True)
                instruction.path.write_text(
                    instruction.content,
                    encoding=instruction.encoding,
                )

        yaml_documents: dict[Path, dict[str, Any]] = {}
        for instruction in self.yaml_updates:
            document: dict[str, Any] = {}
            if instruction.path.exists():
                existing_text = instruction.path.read_text(instruction.encoding)
                loaded_obj = yaml.safe_load(existing_text)
                if isinstance(loaded_obj, Mapping):
                    loaded_mapping = cast("Mapping[str, Any]", loaded_obj)
                    document = copy.deepcopy(dict(loaded_mapping))

            for update in instruction.updates:
                _apply_yaml_update(document, update.key_path, update.value)

            yaml_documents[instruction.path] = document

            if commit:
                instruction.path.parent.mkdir(parents=True, exist_ok=True)
                with instruction.path.open(
                    "w",
                    encoding=instruction.encoding,
                ) as handle:
                    yaml.safe_dump(document, handle, sort_keys=False)

        if commit:
            os.environ.update(self.environment)

        return ParameterApplicationResult(
            environment=environment,
            cli_arguments=cli_arguments,
            file_writes=file_contents,
            yaml_documents=yaml_documents,
        )

    def _describe_yaml_updates(self) -> list[dict[str, object]]:
        """Summarise YAML update instructions for serialisation."""
        updates: list[dict[str, object]] = []
        for instruction in self.yaml_updates:
            key_labels = [".".join(update.key_path) for update in instruction.updates]
            updates.append(
                {
                    "path": str(instruction.path),
                    "encoding": instruction.encoding,
                    "keys": key_labels,
                },
            )
        return updates


@dataclass(slots=True)
class ParameterApplicationResult:
    """Result of applying a plan against a base execution context."""

    environment: dict[str, str]
    cli_arguments: list[str]
    file_writes: dict[Path, str]
    yaml_documents: dict[Path, dict[str, Any]]


def build_application_plan(
    search_space: Sequence[HyperparameterSpec],
    assignments: Mapping[str, ParameterValue],
) -> ParameterApplicationPlan:
    """Build an application plan for the provided hyperparameter assignments."""
    environment: dict[str, str] = {}
    cli_arguments: list[str] = []
    file_writes_map: dict[Path, FileWriteInstruction] = {}
    yaml_updates_map: dict[Path, YAMLUpdateInstruction] = {}

    for spec in search_space:
        if spec.location is None:
            continue

        if spec.name not in assignments:
            continue

        value = assignments[spec.name]

        location = spec.location

        if isinstance(location, EnvironmentParameterLocation):
            environment[location.variable] = str(value)
        elif isinstance(location, CLIArgumentParameterLocation):
            rendered_value = (
                location.value_template.format(value=value)
                if location.value_template is not None
                else str(value)
            )
            if location.flag:
                cli_arguments.extend([location.flag, rendered_value])
            else:
                cli_arguments.append(rendered_value)
        elif isinstance(location, FileContentParameterLocation):
            file_writes_map[location.path] = FileWriteInstruction(
                path=location.path,
                content=str(value),
                encoding=location.encoding,
            )
        else:
            if not isinstance(location, YAMLConfigParameterLocation):
                msg = (
                    "Unsupported parameter location encountered during plan "
                    "construction"
                )
                raise TypeError(msg)
            key_path = _split_key_path(location.key_path)
            updates = yaml_updates_map.setdefault(
                location.path,
                YAMLUpdateInstruction(
                    path=location.path,
                    encoding=location.encoding,
                ),
            )
            updates.updates.append(YAMLKeyUpdate(key_path=key_path, value=value))

    return ParameterApplicationPlan(
        parameter_values=dict(assignments),
        environment=environment,
        cli_arguments=cli_arguments,
        file_writes=list(file_writes_map.values()),
        yaml_updates=list(yaml_updates_map.values()),
    )


def _split_key_path(key_path: str) -> tuple[str, ...]:
    """Split a dot separated key path into tuple form."""
    parts = [segment.strip() for segment in key_path.split(".") if segment.strip()]
    if not parts:
        msg = "YAML key_path must contain at least one segment"
        raise ValueError(msg)
    return tuple(parts)


def _apply_yaml_update(
    document: dict[str, Any],
    key_path: Iterable[str],
    value: ParameterValue,
) -> None:
    """Apply a value update to a nested dictionary following the key path."""
    cursor: dict[str, Any] = document
    key_list = list(key_path)
    for segment in key_list[:-1]:
        if segment not in cursor or not isinstance(cursor[segment], dict):
            cursor[segment] = {}
        cursor = cursor[segment]

    cursor[key_list[-1]] = value


__all__ = [
    "FileWriteInstruction",
    "ParameterApplicationPlan",
    "ParameterApplicationResult",
    "TuningConfig",
    "TuningParameterDefinition",
    "YAMLKeyUpdate",
    "YAMLUpdateInstruction",
    "build_application_plan",
    "default_tuning_config",
    "load_tuning_config",
]

