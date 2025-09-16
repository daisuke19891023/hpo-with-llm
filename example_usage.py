"""Example usage of the OpenTelemetry-enabled logger."""
# ruff: noqa: T201, D401, S108

import os

from test_project.utils.logger import setup_application_logging, shutdown_logging
from test_project.utils.settings import get_settings, reset_settings


def example_basic_usage() -> None:
    """Basic usage example."""
    print("=== Basic Usage Example ===")

    # Setup logging for development
    logger = setup_application_logging("example_app", environment="development")

    # Log some messages
    logger.info("Application started", version="1.0.0")
    logger.debug("Debug information", debug_data={"key": "value"})
    logger.warning("This is a warning", warning_code="W001")
    logger.error("An error occurred", error_code="E001", details="Something went wrong")

    # Bind context
    user_logger = logger.bind(user_id="user123", session_id="sess456")
    user_logger.info("User logged in", action="login")
    user_logger.info("User performed action", action="update_profile")

    print("✓ Basic logging completed")


def example_with_file_output() -> None:
    """Example with file output."""
    print("\n=== File Output Example ===")

    # Set environment variables
    os.environ["LOG_FILE_PATH"] = "/tmp/app.log"
    os.environ["OTEL_LOGS_EXPORT_MODE"] = "file"

    # Reset settings to pick up new environment
    reset_settings()

    # Setup logging
    logger = setup_application_logging("file_example", environment="production")

    # Log messages
    logger.info("Application initialized", mode="file")
    logger.error("Sample error for file logging", error_type="FileError")

    print(f"✓ Logs written to {os.environ['LOG_FILE_PATH']}")

    # Clean up
    del os.environ["LOG_FILE_PATH"]
    del os.environ["OTEL_LOGS_EXPORT_MODE"]


def example_with_settings() -> None:
    """Example showing settings usage."""
    print("\n=== Settings Example ===")

    # Set various environment variables
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_FORMAT"] = "json"
    os.environ["OTEL_LOGS_EXPORT_MODE"] = "both"
    os.environ["OTEL_ENDPOINT"] = "http://localhost:4317"
    os.environ["OTEL_SERVICE_NAME"] = "my-service"
    os.environ["LOG_FILE_PATH"] = "/tmp/both_mode.log"

    # Reset and get settings
    reset_settings()
    settings = get_settings()

    print("Current settings:")
    print(f"  - Log Level: {settings.log_level}")
    print(f"  - Log Format: {settings.log_format}")
    print(f"  - Export Mode: {settings.otel_logs_export_mode}")
    print(f"  - OTEL Enabled: {settings.otel_export_enabled}")
    print(f"  - Service Name: {settings.otel_service_name}")

    # Use the logger
    logger = setup_application_logging("settings_example")
    logger.info(
        "Using custom settings",
        settings_mode=settings.otel_logs_export_mode.value,
    )

    # Clean up
    for key in [
        "LOG_LEVEL",
        "LOG_FORMAT",
        "OTEL_LOGS_EXPORT_MODE",
        "OTEL_ENDPOINT",
        "OTEL_SERVICE_NAME",
        "LOG_FILE_PATH",
    ]:
        os.environ.pop(key, None)

    print("✓ Settings example completed")


def main() -> None:
    """Run all examples."""
    try:
        example_basic_usage()
        example_with_file_output()
        example_with_settings()
    finally:
        # Always shutdown logging to clean up resources
        shutdown_logging()
        print("\n✓ All examples completed successfully!")


if __name__ == "__main__":
    main()
