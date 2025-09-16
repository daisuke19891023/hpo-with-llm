"""Unit tests for logger functionality."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import structlog

from clean_interfaces.utils.logger import configure_logging, get_logger, log_performance

# Test constants
EXPECTED_DURATION_150 = 150
EXPECTED_RETRY_COUNT_3 = 3
EXPECTED_INT_42 = 42
EXPECTED_FLOAT_314 = 3.14
EXPECTED_LOG_LEVELS_COUNT = 5
TEST_EXCEPTION_MESSAGE = "Test exception"


class TestLoggerConfiguration:
    """Unit tests for logger configuration."""

    def test_configure_logging_with_json_format(self) -> None:
        """Test logger configuration with JSON format."""
        configure_logging(log_level="INFO", log_format="json")

        # Verify structlog is configured
        logger = structlog.get_logger("test")
        assert logger is not None

        # Test that logger can be called without errors
        logger.info("Test message", key="value")

    def test_configure_logging_with_console_format(self) -> None:
        """Test logger configuration with console format."""
        configure_logging(log_level="DEBUG", log_format="console")

        logger = structlog.get_logger("test")
        assert logger is not None
        logger.debug("Debug message", debug_key="debug_value")

    def test_configure_logging_with_file_output(self) -> None:
        """Test logger configuration with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            configure_logging(
                log_level="WARNING",
                log_format="json",
                log_file=str(log_file),
            )

            logger = structlog.get_logger("file_test")
            logger.warning("Warning message", warning_key="warning_value")
            logger.info("Info message")  # Should not appear due to WARNING level

            # Verify log file exists and contains expected content
            assert log_file.exists()
            content = log_file.read_text()
            assert "Warning message" in content
            assert "Info message" not in content

    def test_get_logger_returns_bound_logger(self) -> None:
        """Test that get_logger returns a properly bound logger."""
        configure_logging(log_level="INFO", log_format="json")

        logger = get_logger("test_module")
        assert logger is not None

        # Test that logger has expected methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "critical")

    def test_logger_with_context_binding(self) -> None:
        """Test logger context binding functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "context_test.log"

            configure_logging(
                log_level="INFO",
                log_format="json",
                log_file=str(log_file),
            )

            logger = get_logger("context_test")

            # Bind context to logger
            bound_logger = logger.bind(
                user_id="user123",
                session_id="session456",
                request_id="req789",
            )

            bound_logger.info("Action performed", action="login")

            # Verify context is included in log output
            content = log_file.read_text()
            log_entry: dict[str, Any] = json.loads(content.strip())

            assert log_entry["user_id"] == "user123"
            assert log_entry["session_id"] == "session456"
            assert log_entry["request_id"] == "req789"
            assert log_entry["action"] == "login"
            assert log_entry["event"] == "Action performed"


class TestStructuredOutput:
    """Unit tests for structured log output validation."""

    def test_json_output_structure(self) -> None:
        """Test that JSON output has the expected structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "structure_test.log"

            configure_logging(
                log_level="INFO",
                log_format="json",
                log_file=str(log_file),
            )

            logger = get_logger("structure_test")
            logger.info(
                "Test message",
                string_field="value",
                int_field=EXPECTED_INT_42,
                float_field=EXPECTED_FLOAT_314,
                bool_field=True,
                list_field=["a", "b", "c"],
                dict_field={"nested": "value"},
            )

            content = log_file.read_text()
            log_entry: dict[str, Any] = json.loads(content.strip())

            # Verify required fields
            assert "timestamp" in log_entry
            assert "level" in log_entry
            assert "logger" in log_entry
            assert "event" in log_entry

            # Verify custom fields
            assert log_entry["string_field"] == "value"
            assert log_entry["int_field"] == EXPECTED_INT_42
            assert log_entry["float_field"] == EXPECTED_FLOAT_314
            assert log_entry["bool_field"] is True
            assert log_entry["list_field"] == ["a", "b", "c"]
            assert log_entry["dict_field"] == {"nested": "value"}

    def test_log_levels(self) -> None:
        """Test different log levels are properly handled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "levels_test.log"

            configure_logging(
                log_level="DEBUG",
                log_format="json",
                log_file=str(log_file),
            )

            logger = get_logger("levels_test")

            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")

            content = log_file.read_text()
            log_lines = [line for line in content.strip().split("\n") if line]

            assert len(log_lines) == EXPECTED_LOG_LEVELS_COUNT

            levels: list[str] = []
            for line in log_lines:
                entry: dict[str, Any] = json.loads(line)
                levels.append(entry["level"])

            assert levels == ["debug", "info", "warning", "error", "critical"]


class TestOpenTelemetryCompatibility:
    """Unit tests for OpenTelemetry integration compatibility."""

    def test_logger_without_opentelemetry(self) -> None:
        """Test that logger works without OpenTelemetry installed."""
        # Should work fine without OpenTelemetry
        configure_logging(log_level="INFO", log_format="json")
        logger = get_logger("otel_compat_test")
        logger.info("Test message without OpenTelemetry")

    def test_logger_structure_supports_trace_fields(self) -> None:
        """Test that log structure can accommodate OpenTelemetry trace fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "trace_test.log"

            configure_logging(
                log_level="INFO",
                log_format="json",
                log_file=str(log_file),
            )

            logger = get_logger("trace_test")

            # Manually add trace-like fields to verify structure supports them
            logger.info(
                "Test message with trace info",
                trace_id="abc123def456",
                span_id="789xyz",
                trace_flags="01",
            )

            content = log_file.read_text()
            log_entry: dict[str, Any] = json.loads(content.strip())

            assert log_entry["trace_id"] == "abc123def456"
            assert log_entry["span_id"] == "789xyz"
            assert log_entry["trace_flags"] == "01"


class TestPerformanceLogging:
    """Unit tests for performance logging decorator."""

    def test_log_performance_decorator(self) -> None:
        """Test performance logging decorator functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "perf_test.log"

            configure_logging(
                log_level="INFO",
                log_format="json",
                log_file=str(log_file),
            )

            logger = get_logger("perf_test")

            @log_performance(logger)
            def test_function(value: int) -> int:
                return value * 2

            result = test_function(21)
            assert result == EXPECTED_INT_42

            # Verify performance log was created
            content = log_file.read_text()
            log_entry: dict[str, Any] = json.loads(content.strip())

            assert "duration_ms" in log_entry
            assert "function_name" in log_entry
            assert log_entry["function_name"] == "test_function"
            assert isinstance(log_entry["duration_ms"], int | float)
            assert log_entry["duration_ms"] >= 0

    def test_log_performance_with_exception(self) -> None:
        """Test performance logging when function raises exception."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "perf_exception_test.log"

            configure_logging(
                log_level="INFO",
                log_format="json",
                log_file=str(log_file),
            )

            logger = get_logger("perf_exception_test")

            @log_performance(logger)
            def failing_function() -> None:
                exception_msg = TEST_EXCEPTION_MESSAGE
                raise ValueError(exception_msg)

            with pytest.raises(ValueError, match=TEST_EXCEPTION_MESSAGE):
                failing_function()

            # Verify performance log was created even with exception
            content = log_file.read_text()
            log_entry: dict[str, Any] = json.loads(content.strip())

            assert "duration_ms" in log_entry
            assert "function_name" in log_entry
            assert log_entry["function_name"] == "failing_function"
            assert "exception" in log_entry
            assert log_entry["exception"] == f"ValueError: {TEST_EXCEPTION_MESSAGE}"
