"""Pytest configuration for test_project tests."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def temp_dir():  # type: ignore[no-untyped-def]  # noqa: ANN201
    """Create a temporary directory for file operations testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_json_data() -> dict[str, Any]:
    """Sample JSON data for testing."""
    return {
        "name": "Test Project",
        "version": "1.0.0",
        "features": ["file_handling", "logging", "testing"],
        "config": {
            "debug": True,
            "max_retries": 3,
            "encoding": "utf-8",
        },
        "unicode_test": "テスト文字列",  # Japanese text for encoding tests
    }


@pytest.fixture
def sample_yaml_data() -> dict[str, Any]:
    """Sample YAML data for testing."""
    return {
        "application": {
            "name": "TestApp",
            "environment": "development",
            "settings": {
                "port": 8080,
                "host": "localhost",
                "ssl_enabled": False,
            },
        },
        "database": {
            "driver": "postgresql",
            "host": "localhost",
            "port": 5432,
            "name": "testdb",
        },
        "unicode_content": "テストYAML",  # Japanese text for encoding tests
    }


@pytest.fixture
def sample_text_utf8() -> str:
    """Sample UTF-8 text for testing."""
    return """This is a test file with UTF-8 encoding.
It contains multiple lines and special characters: €, ñ, ü
日本語のテキストも含まれています。
End of file."""


@pytest.fixture
def sample_text_cp932() -> str:
    """Sample text that can be encoded in CP932."""
    return """This is a test file for CP932 encoding.
日本語の文字列テスト
シフトJISエンコーディング
End of file."""


@pytest.fixture
def sample_files(
    temp_dir: Path,
    sample_json_data: dict[str, Any],
    sample_yaml_data: dict[str, Any],
    sample_text_utf8: str,
    sample_text_cp932: str,
) -> dict[str, Path]:
    """Create sample files for testing and return their paths."""
    files: dict[str, Path] = {}

    # Create JSON file
    json_path = temp_dir / "test_data.json"
    json_path.write_text(
        json.dumps(sample_json_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    files["json"] = json_path

    # Create YAML file
    yaml_path = temp_dir / "test_config.yaml"
    yaml_path.write_text(
        yaml.dump(sample_yaml_data, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    files["yaml"] = yaml_path

    # Create UTF-8 text file
    utf8_path = temp_dir / "test_utf8.txt"
    utf8_path.write_text(sample_text_utf8, encoding="utf-8")
    files["utf8_text"] = utf8_path

    # Create CP932 text file
    cp932_path = temp_dir / "test_cp932.txt"
    cp932_path.write_text(sample_text_cp932, encoding="cp932")
    files["cp932_text"] = cp932_path

    # Create nested directory structure for testing parent creation
    nested_dir = temp_dir / "nested" / "deep" / "structure"
    files["nested_dir"] = nested_dir

    return files


@pytest.fixture
def invalid_json_file(temp_dir: Path) -> Path:
    """Create an invalid JSON file for error testing."""
    invalid_path = temp_dir / "invalid.json"
    # Invalid JSON with undefined value
    invalid_path.write_text(
        '{"name": "test", "value": undefined}',
        encoding="utf-8",
    )
    return invalid_path


@pytest.fixture
def invalid_yaml_file(temp_dir: Path) -> Path:
    """Create an invalid YAML file for error testing."""
    invalid_path = temp_dir / "invalid.yaml"
    # Invalid YAML with bad indentation
    invalid_path.write_text(
        "key: value\n  invalid: indentation without parent",
        encoding="utf-8",
    )
    return invalid_path


@pytest.fixture
def non_existent_file(temp_dir: Path) -> Path:
    """Return path to a non-existent file for error testing."""
    return temp_dir / "does_not_exist.txt"
