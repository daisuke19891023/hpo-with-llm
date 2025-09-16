"""End-to-end integration tests for file_handler with logger module."""

import logging
from pathlib import Path
from typing import Any

import pytest
import structlog
import yaml

from clean_interfaces.utils.file_handler import FileHandler
from clean_interfaces.utils.logger import configure_logging, get_logger


class TestFileHandlerLoggingIntegration:
    """Test file handler operations with logging integration."""

    @pytest.fixture(autouse=True)
    def setup_logging_capture(self, caplog: pytest.LogCaptureFixture) -> None:
        """Set up logging capture for tests."""
        # Configure structlog for testing
        configure_logging(
            log_level="DEBUG",
            log_format="plain",
            log_file=None,
        )
        caplog.set_level(logging.DEBUG)

    def test_complete_json_workflow_with_logging(
        self,
        temp_dir: Path,
        sample_json_data: dict[str, Any],
    ) -> None:
        """Test complete JSON file workflow with proper logging."""
        logger = get_logger("test_json_workflow")
        handler = FileHandler()

        # Test file path
        json_file = temp_dir / "workflow_test.json"

        # Log workflow start
        logger.info("Starting JSON workflow test", file_path=str(json_file))

        # Write JSON file
        handler.write_json(json_file, sample_json_data)
        logger.info("JSON file written", data_keys=list(sample_json_data.keys()))

        # Read JSON file back
        read_data = handler.read_json(json_file)
        logger.info("JSON file read", data_keys=list(read_data.keys()))

        # Verify data integrity
        assert read_data == sample_json_data
        logger.info("JSON workflow completed successfully")

    def test_yaml_operations_with_error_handling(
        self,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test YAML operations with error handling and logging."""
        handler = FileHandler()
        logger = get_logger("test_yaml_errors")

        # Create an invalid YAML file
        invalid_yaml_path = temp_dir / "invalid.yaml"
        invalid_content = "key: value\n  bad: indentation"
        invalid_yaml_path.write_text(invalid_content)

        # Attempt to read invalid YAML
        logger.info("Attempting to read invalid YAML file")

        with pytest.raises(yaml.YAMLError):
            handler.read_yaml(invalid_yaml_path)

        # Check that error was logged
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_logs) > 0
        assert "Invalid YAML format" in error_logs[-1].message

    def test_encoding_conversion_workflow(
        self,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test file encoding conversion workflow with logging."""
        logger = get_logger("test_encoding")

        # Original content with Japanese text
        japanese_content = """ファイルハンドラーテスト
これは日本語のテキストです。
エンコーディング変換のテストを行います。"""

        utf8_file = temp_dir / "japanese_utf8.txt"
        cp932_file = temp_dir / "japanese_cp932.txt"

        # Write as UTF-8
        logger.info("Writing Japanese text as UTF-8")
        handler_utf8 = FileHandler(encoding="utf-8")
        handler_utf8.write_text(utf8_file, japanese_content)

        # Read UTF-8 and write as CP932
        logger.info("Converting UTF-8 to CP932")
        content = handler_utf8.read_text(utf8_file)

        handler_cp932 = FileHandler(encoding="cp932")
        handler_cp932.write_text(cp932_file, content)

        # Verify conversion
        logger.info("Verifying encoding conversion")
        cp932_content = handler_cp932.read_text(cp932_file)
        assert cp932_content == japanese_content

        # Check logs contain encoding information
        log_messages = [r.message for r in caplog.records]
        assert any("utf-8" in msg.lower() for msg in log_messages)
        assert any("cp932" in msg.lower() for msg in log_messages)

    def test_batch_file_processing(
        self,
        temp_dir: Path,
        sample_json_data: dict[str, Any],
        sample_yaml_data: dict[str, Any],
    ) -> None:
        """Test batch processing multiple files with structured logging."""
        logger = get_logger("batch_processor")
        handler = FileHandler()

        # Create test files
        files_to_process = {
            "config.json": sample_json_data,
            "settings.yaml": sample_yaml_data,
            "data.json": {"items": [1, 2, 3], "total": 6},
        }

        logger.info("Starting batch file processing", file_count=len(files_to_process))

        # Write all files
        for filename, data in files_to_process.items():
            file_path = temp_dir / filename

            with structlog.contextvars.bound_contextvars(
                filename=filename,
                operation="write",
            ):
                if filename.endswith(".json"):
                    handler.write_json(file_path, data)
                elif filename.endswith(".yaml"):
                    handler.write_yaml(file_path, data)

                logger.info("File processed")

        # Read and verify all files
        for filename in files_to_process:
            file_path = temp_dir / filename

            with structlog.contextvars.bound_contextvars(
                filename=filename,
                operation="read",
            ):
                if filename.endswith(".json"):
                    data = handler.read_json(file_path)
                    assert data is not None
                elif filename.endswith(".yaml"):
                    data = handler.read_yaml(file_path)
                    assert data is not None
                logger.info("File verified")

        logger.info(
            "Batch processing completed",
            processed_files=list(files_to_process.keys()),
        )

    def test_error_recovery_workflow(
        self,
        non_existent_file: Path,
    ) -> None:
        """Test error handling and recovery workflow."""
        logger = get_logger("error_recovery")
        handler = FileHandler()

        # Try to read non-existent file
        logger.info("Testing error recovery workflow")

        try:
            handler.read_text(non_existent_file)
        except FileNotFoundError:
            logger.warning("File not found, creating default content")

            # Create default content
            default_content = {"default": True, "message": "Auto-generated"}
            handler.write_json(non_existent_file, default_content)
            logger.info("Default file created")

        # Verify recovery
        recovered_data = handler.read_json(non_existent_file)
        assert recovered_data["default"] is True
        logger.info("Error recovery successful")

    def test_concurrent_file_operations(self, temp_dir: Path) -> None:
        """Test multiple file operations with contextual logging."""
        logger = get_logger("concurrent_ops")

        # Simulate processing multiple file types concurrently
        operations = [
            ("user_data.json", {"users": ["alice", "bob"]}, "json"),
            ("config.yaml", {"server": {"port": 8080}}, "yaml"),
            ("notes.txt", "Important notes here", "text"),
        ]

        for filename, content, file_type in operations:
            file_path = temp_dir / filename

            # Use context manager for structured logging
            with structlog.contextvars.bound_contextvars(
                file=filename,
                type=file_type,
                thread="simulated",
            ):
                handler = FileHandler()
                logger.info("Processing file")

                if file_type == "json":
                    handler.write_json(file_path, content)  # type: ignore[arg-type]
                    result = handler.read_json(file_path)
                elif file_type == "yaml":
                    handler.write_yaml(file_path, content)  # type: ignore[arg-type]
                    result = handler.read_yaml(file_path)
                else:
                    handler.write_text(file_path, content)  # type: ignore[arg-type]
                    result = handler.read_text(file_path)

                logger.info("File processed successfully")
                assert result is not None

    def test_large_file_handling_with_progress(self, temp_dir: Path) -> None:
        """Test handling large files with progress logging."""
        logger = get_logger("large_file_handler")
        handler = FileHandler()

        # Create a large data structure
        large_data = {
            f"section_{i}": {
                "data": list(range(100)),
                "metadata": {"index": i, "size": 100},
            }
            for i in range(50)  # 50 sections
        }

        large_file = temp_dir / "large_data.json"

        # Write large file with progress logging
        logger.info("Writing large file", sections=len(large_data))
        handler.write_json(
            large_file,
            large_data,
        )  # No indent for smaller file

        file_size = large_file.stat().st_size
        logger.info("Large file written", size_bytes=file_size)

        # Read large file
        logger.info("Reading large file")
        read_data = handler.read_json(large_file)

        assert len(read_data) == 50
        logger.info("Large file processed successfully", sections_read=len(read_data))

    def test_file_monitoring_simulation(
        self,
        temp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Simulate file monitoring with detailed logging."""
        logger = get_logger("file_monitor")
        handler = FileHandler()

        watched_file = temp_dir / "monitored.json"

        # Initial file creation
        initial_data = {"version": 1, "status": "active"}
        handler.write_json(watched_file, initial_data)
        logger.info("File monitoring started", file=str(watched_file))

        # Simulate file updates
        for i in range(3):
            # Read current state
            current = handler.read_json(watched_file)
            logger.debug("File state read", version=current.get("version"))

            # Update file
            current["version"] = i + 2
            current[f"update_{i}"] = f"change_{i}"
            handler.write_json(watched_file, current)

            logger.info(
                "File updated",
                version=current["version"],
                changes=f"update_{i}",
            )

        # Verify all updates were logged
        info_logs = [
            r
            for r in caplog.records
            if r.levelname == "INFO" and "File updated" in r.message
        ]
        assert len(info_logs) == 3
