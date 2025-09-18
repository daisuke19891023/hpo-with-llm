"""Tests for golden dataset loading utilities."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from clean_interfaces.evaluation.datasets import (
    GoldenDataset,
    load_default_golden_dataset,
    load_golden_dataset,
)


if TYPE_CHECKING:
    from pathlib import Path


def _sample_dataset() -> dict[str, object]:
    return {
        "classification": [
            {"id": "a", "label": True},
            {"id": "b", "label": False},
        ],
        "reference_plans": [
            "Enable vector tooling for compliance reviews.",
            "Use keyword lookup for FAQ responses.",
        ],
        "metadata": {"name": "unit_test_dataset"},
    }


class TestGoldenDatasetLoading:
    """Validate dataset loading from disk."""

    def test_load_from_json_file(self, tmp_path: Path) -> None:
        """Datasets should load correctly from JSON files."""
        payload = _sample_dataset()
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        dataset = load_golden_dataset(path)
        assert dataset.size == 2
        assert dataset.labels == (True, False)
        assert dataset.metadata["name"] == "unit_test_dataset"

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        """Datasets should load correctly from YAML files."""
        path = tmp_path / "dataset.yaml"
        path.write_text(
            """
classification:
  - id: example-1
    label: true
reference_plans:
  - Example plan
            """.strip(),
            encoding="utf-8",
        )

        dataset = load_golden_dataset(path)
        assert dataset.size == 1
        assert dataset.labels == (True,)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Missing files should raise ``FileNotFoundError``."""
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_golden_dataset(missing)

    def test_invalid_payload_raises(self, tmp_path: Path) -> None:
        """Invalid payloads should raise ``ValueError``."""
        path = tmp_path / "invalid.yaml"
        path.write_text("- not: a-mapping", encoding="utf-8")

        with pytest.raises(ValueError, match="mapping"):
            load_golden_dataset(path)

    def test_dataset_requires_indicator_data(self) -> None:
        """Datasets must contain at least one supported indicator payload."""
        with pytest.raises(ValueError, match="must define classification"):
            GoldenDataset.from_mapping({"classification": [], "file_search": []})

    def test_file_search_only_dataset_loads(self) -> None:
        """Datasets containing only file search records should load successfully."""
        dataset = GoldenDataset.from_mapping(
            {
                "file_search": [
                    {
                        "id": "search",
                        "relevant_files": ["docs/example.txt"],
                    },
                ],
            },
        )

        assert dataset.size == 1
        assert dataset.file_search[0].query_id == "search"

    def test_default_dataset_asset_loads(self) -> None:
        """The packaged default dataset should load from disk."""
        dataset = load_default_golden_dataset()

        assert dataset.metadata["name"] == "synthetic_rag_benchmark"
        assert len(dataset.classification) == 5
        assert len(dataset.file_search) == 5
