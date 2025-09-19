"""Tests for loading and applying HPO tuning configurations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import yaml

from clean_interfaces.hpo.configuration import (
    ParameterApplicationPlan,
    build_application_plan,
    default_tuning_config,
    load_tuning_config,
)

if TYPE_CHECKING:
    from pathlib import Path
    import pytest


def _write_config(tmp_path: Path, *, file_path: Path, yaml_path: Path) -> Path:
    """Create a sample tuning configuration on disk and return its path."""
    config = {
        "parameters": [
            {
                "name": "temperature",
                "type": "float",
                "lower": 0.0,
                "upper": 1.0,
                "description": "Sampling temperature",
                "location": {
                    "kind": "environment",
                    "variable": "PROMPT_TEMPERATURE",
                },
            },
            {
                "name": "max_tokens",
                "type": "int",
                "lower": 32,
                "upper": 256,
                "step": 32,
                "location": {
                    "kind": "cli_argument",
                    "flag": "--max-tokens",
                },
            },
            {
                "name": "prompt_template",
                "type": "categorical",
                "choices": ["simple", "extended"],
                "location": {
                    "kind": "file_content",
                    "path": str(file_path),
                },
            },
            {
                "name": "retrieval_mode",
                "type": "categorical",
                "choices": ["keyword", "vector"],
                "location": {
                    "kind": "config_yaml",
                    "path": str(yaml_path),
                    "key_path": "retrieval.mode",
                },
            },
        ],
    }

    config_path = tmp_path / "tuning.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_load_configuration_and_build_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YAML configuration should be parsed into application plans."""
    file_path = tmp_path / "prompt.txt"
    yaml_path = tmp_path / "rag.yaml"
    config_path = _write_config(tmp_path, file_path=file_path, yaml_path=yaml_path)

    config = load_tuning_config(config_path)
    search_space = config.to_search_space()

    assignments = {
        "temperature": 0.35,
        "max_tokens": 128,
        "prompt_template": "extended",
        "retrieval_mode": "vector",
    }

    plan = build_application_plan(search_space, assignments)

    assert isinstance(plan, ParameterApplicationPlan)
    assert plan.environment["PROMPT_TEMPERATURE"] == "0.35"
    assert plan.cli_arguments == ["--max-tokens", "128"]

    file_instruction = {
        instruction.path: instruction.content for instruction in plan.file_writes
    }
    assert file_instruction[file_path] == "extended"

    yaml_instruction = {
        instruction.path: instruction.updates for instruction in plan.yaml_updates
    }
    assert yaml_instruction[yaml_path][0].key_path == ("retrieval", "mode")
    assert yaml_instruction[yaml_path][0].value == "vector"

    monkeypatch.setenv("PROMPT_TEMPERATURE", "0.1")
    result = plan.apply(commit=False, base_environment=os.environ)

    assert result.environment["PROMPT_TEMPERATURE"] == "0.35"
    assert result.cli_arguments == ["--max-tokens", "128"]
    assert yaml_path in result.yaml_documents
    assert result.yaml_documents[yaml_path]["retrieval"]["mode"] == "vector"

    plan.apply(commit=True)

    assert os.environ["PROMPT_TEMPERATURE"] == "0.35"
    assert file_path.read_text(encoding="utf-8") == "extended"
    stored_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert stored_yaml["retrieval"]["mode"] == "vector"


def test_default_configuration_provides_locations() -> None:
    """The default configuration should expose location metadata."""
    config = default_tuning_config()
    specs = config.to_search_space()
    assert any(spec.location is not None for spec in specs)
    descriptions = config.describe()
    assert descriptions[0]["name"] == "temperature"

