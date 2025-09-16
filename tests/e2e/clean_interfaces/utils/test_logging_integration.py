"""E2E tests for logging integration."""

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from clean_interfaces.utils.logger import configure_logging, get_logger, log_performance

# Test constants
EXPECTED_DURATION_150 = 150
EXPECTED_RETRY_COUNT_3 = 3
MIN_LOG_LINES_COUNT = 2


class TestLoggingIntegration:
    """Complete workflow tests for logging functionality."""

    def test_complete_logging_workflow(self) -> None:
        """Complete logging workflow from setup to structured output."""
        # This test will be implemented after logger.py is created
        # It should test:
        # 1. Logger initialization
        # 2. Structured log output
        # 3. Different log levels
        # 4. Context binding
        # 5. OpenTelemetry compatibility

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Configure logging with JSON output
            configure_logging(
                log_level="INFO",
                log_format="json",
                log_file=str(log_file),
            )

            logger = get_logger("test_module")

            # Test structured logging
            logger.info(
                "User action completed",
                user_id="123",
                action="login",
                duration_ms=EXPECTED_DURATION_150,
            )

            logger.error(
                "Database connection failed",
                database="primary",
                error_code="CONNECTION_TIMEOUT",
                retry_count=EXPECTED_RETRY_COUNT_3,
            )

            # Verify log file contains structured JSON
            assert log_file.exists()
            log_content = log_file.read_text()
            log_lines = [line for line in log_content.strip().split("\n") if line]

            assert len(log_lines) >= MIN_LOG_LINES_COUNT

            # Parse and verify first log entry
            first_log: dict[str, Any] = json.loads(log_lines[0])
            assert first_log["event"] == "User action completed"
            assert first_log["user_id"] == "123"
            assert first_log["action"] == "login"
            assert first_log["duration_ms"] == EXPECTED_DURATION_150
            assert "timestamp" in first_log
            assert first_log["level"] == "info"

            # Parse and verify second log entry
            second_log: dict[str, Any] = json.loads(log_lines[1])
            assert second_log["event"] == "Database connection failed"
            assert second_log["database"] == "primary"
            assert second_log["error_code"] == "CONNECTION_TIMEOUT"
            assert second_log["retry_count"] == EXPECTED_RETRY_COUNT_3
            assert second_log["level"] == "error"

    def test_opentelemetry_trace_integration(self) -> None:
        """Test OpenTelemetry trace context integration."""
        # This test will verify that logging integrates with OpenTelemetry traces
        # when available, including trace_id and span_id in log records

        configure_logging(log_level="DEBUG", log_format="json")
        logger = get_logger("otel_test")

        # Test logging without OpenTelemetry context (should work normally)
        logger.info("Test message without trace context")

        # Future: Test with OpenTelemetry context when dependency is added
        # This would include trace_id and span_id in the log output

    def test_performance_logging_workflow(self) -> None:
        """Test performance logging decorator workflow."""
        configure_logging(log_level="DEBUG", log_format="json")
        logger = get_logger("perf_test")

        @log_performance(logger)
        def slow_operation(duration: float = 0.001) -> str:
            time.sleep(duration)
            return "completed"

        result = slow_operation(0.001)
        assert result == "completed"

        # Performance metrics should be logged automatically
