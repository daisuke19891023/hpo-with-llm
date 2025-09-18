"""Golden dataset loading helpers for evaluation workflows."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, cast
from collections.abc import Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ClassificationRecord(BaseModel):
    """Individual labelled example used for computing metrics."""

    record_id: str = Field(alias="id", description="Unique identifier for the record")
    label: bool = Field(description="Ground truth label for binary classification")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata describing the scenario",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class FileSearchRecord(BaseModel):
    """Ground truth definition for file search evaluation."""

    query_id: str = Field(alias="id", description="Unique identifier for the query")
    relevant_files: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Ordered collection of files considered relevant",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context describing the query",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class GoldenDataset(BaseModel):
    """Container for ground truth examples and reference plans."""

    classification: tuple[ClassificationRecord, ...] = Field(
        default_factory=tuple,
        description="Labelled examples for computing accuracy/precision/recall",
    )
    file_search: tuple[FileSearchRecord, ...] = Field(
        default_factory=tuple,
        description="File search scenarios for retrieval-based evaluation",
    )
    reference_plans: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Canonical natural language plans used for LLM judging",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata describing the dataset",
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def size(self) -> int:
        """Return the number of examples relevant for indicator evaluation."""
        if self.classification:
            return len(self.classification)
        if self.file_search:
            return len(self.file_search)
        return 0

    @property
    def labels(self) -> tuple[bool, ...]:
        """Return the ordered ground-truth labels."""
        return tuple(record.label for record in self.classification)

    @model_validator(mode="after")
    def _validate_indicator_content(self) -> GoldenDataset:
        """Ensure at least one indicator dataset is present."""
        if not self.classification and not self.file_search:
            message = "Golden dataset must define classification or file search records"
            raise ValueError(message)
        return self

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GoldenDataset:
        """Validate and construct a dataset from a mapping."""
        try:
            dataset = cls.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - exercised in tests
            errors = exc.errors()
            if errors:
                detail = errors[0].get("msg")
                if detail:
                    raise ValueError(detail) from exc
            message = "Invalid golden dataset payload"
            raise ValueError(message) from exc

        return dataset

    @classmethod
    def from_path(cls, path: Path | str) -> GoldenDataset:
        """Load a dataset definition from a JSON or YAML file."""
        resolved = Path(path)
        if not resolved.exists():
            msg = f"Golden dataset file not found: {resolved}"
            raise FileNotFoundError(msg)

        raw = resolved.read_text(encoding="utf-8")
        loaded: Any
        if resolved.suffix.lower() in {".yaml", ".yml"}:
            loaded = yaml.safe_load(raw) or {}
        elif resolved.suffix.lower() == ".json":
            loaded = json.loads(raw or "{}")
        else:
            msg = f"Unsupported golden dataset format: {resolved.suffix}"
            raise ValueError(msg)

        if not isinstance(loaded, Mapping):
            msg = "Golden dataset file must contain a mapping at the top level"
            raise TypeError(msg)

        mapping = cast(Mapping[str, Any], loaded)
        return cls.from_mapping(mapping)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataset into a standard dictionary."""
        return self.model_dump(mode="python")


@lru_cache(maxsize=1)
def load_default_golden_dataset() -> GoldenDataset:
    """Load the packaged default golden dataset definition."""
    package = "clean_interfaces.evaluation.data"
    resource_name = "default_golden_dataset.yaml"

    try:
        resource = resources.files(package).joinpath(resource_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - importlib contract
        msg = "Default golden dataset package not available"
        raise FileNotFoundError(msg) from exc

    with resource.open("r", encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle) or {}

    if not isinstance(payload, Mapping):
        msg = "Default golden dataset asset must contain a mapping"
        raise TypeError(msg)

    mapping = cast(Mapping[str, Any], payload)
    return GoldenDataset.from_mapping(mapping)


def load_golden_dataset(path: Path | str) -> GoldenDataset:
    """Load a golden dataset definition from the provided path."""
    try:
        return GoldenDataset.from_path(path)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


__all__ = [
    "ClassificationRecord",
    "FileSearchRecord",
    "GoldenDataset",
    "load_default_golden_dataset",
    "load_golden_dataset",
]
