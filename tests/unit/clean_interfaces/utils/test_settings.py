"""Unit tests for settings module."""

import os
from typing import Any

import pytest
from pydantic import ValidationError

from clean_interfaces.utils.settings import (
    LoggingSettings,
    OTelExportMode,
)


class TestLoggingSettings:
    """Unit tests for LoggingSettings configuration."""

    def test_default_settings(self) -> None:
        """Test LoggingSettings with default values."""
        settings = LoggingSettings()

        assert settings.log_level == "INFO"
        assert settings.log_format == "json"
        assert settings.log_file_path is None
        assert settings.otel_logs_export_mode == OTelExportMode.FILE
        assert settings.otel_endpoint == "http://localhost:4317"
        assert settings.otel_service_name == "python-app"
        assert settings.otel_export_timeout == 30000
        assert settings.otel_export_enabled is False

    def test_settings_from_environment(self) -> None:
        """Test LoggingSettings loading from environment variables."""
        # Set environment variables
        env_vars = {
            "LOG_LEVEL": "DEBUG",
            "LOG_FORMAT": "console",
            "LOG_FILE_PATH": "/var/log/app.log",
            "OTEL_LOGS_EXPORT_MODE": "otlp",
            "OTEL_ENDPOINT": "http://otel-collector:4317",
            "OTEL_SERVICE_NAME": "my-service",
            "OTEL_EXPORT_TIMEOUT": "60000",
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        try:
            settings = LoggingSettings()

            assert settings.log_level == "DEBUG"
            assert settings.log_format == "console"
            assert settings.log_file_path == "/var/log/app.log"
            assert settings.otel_logs_export_mode == OTelExportMode.OTLP
            assert settings.otel_endpoint == "http://otel-collector:4317"
            assert settings.otel_service_name == "my-service"
            assert settings.otel_export_timeout == 60000
            assert settings.otel_export_enabled is True

        finally:
            # Clean up environment variables
            for key in env_vars:
                os.environ.pop(key, None)

    def test_export_mode_validation(self) -> None:
        """Test OTelExportMode enum validation."""
        # Valid modes
        for mode in ["file", "otlp", "both"]:
            os.environ["OTEL_LOGS_EXPORT_MODE"] = mode
            try:
                settings = LoggingSettings()
                assert settings.otel_logs_export_mode.value == mode
            finally:
                os.environ.pop("OTEL_LOGS_EXPORT_MODE", None)

        # Invalid mode should raise validation error
        os.environ["OTEL_LOGS_EXPORT_MODE"] = "invalid"
        try:
            with pytest.raises(ValidationError):
                LoggingSettings()
        finally:
            os.environ.pop("OTEL_LOGS_EXPORT_MODE", None)

    def test_log_level_validation(self) -> None:
        """Test log level validation."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            os.environ["LOG_LEVEL"] = level
            try:
                settings = LoggingSettings()
                assert settings.log_level == level
            finally:
                os.environ.pop("LOG_LEVEL", None)

        # Invalid level should raise validation error
        os.environ["LOG_LEVEL"] = "INVALID"
        try:
            with pytest.raises(ValidationError):
                LoggingSettings()
        finally:
            os.environ.pop("LOG_LEVEL", None)

    def test_log_format_validation(self) -> None:
        """Test log format validation."""
        valid_formats = ["json", "console", "plain"]

        for fmt in valid_formats:
            os.environ["LOG_FORMAT"] = fmt
            try:
                settings = LoggingSettings()
                assert settings.log_format == fmt
            finally:
                os.environ.pop("LOG_FORMAT", None)

        # Invalid format should raise validation error
        os.environ["LOG_FORMAT"] = "xml"
        try:
            with pytest.raises(ValidationError):
                LoggingSettings()
        finally:
            os.environ.pop("LOG_FORMAT", None)

    def test_otel_export_enabled_property(self) -> None:
        """Test otel_export_enabled property logic."""
        # File mode - export disabled
        os.environ["OTEL_LOGS_EXPORT_MODE"] = "file"
        try:
            settings = LoggingSettings()
            assert settings.otel_export_enabled is False
        finally:
            os.environ.pop("OTEL_LOGS_EXPORT_MODE", None)

        # OTLP mode - export enabled
        os.environ["OTEL_LOGS_EXPORT_MODE"] = "otlp"
        try:
            settings = LoggingSettings()
            assert settings.otel_export_enabled is True
        finally:
            os.environ.pop("OTEL_LOGS_EXPORT_MODE", None)

        # Both mode - export enabled
        os.environ["OTEL_LOGS_EXPORT_MODE"] = "both"
        try:
            settings = LoggingSettings()
            assert settings.otel_export_enabled is True
        finally:
            os.environ.pop("OTEL_LOGS_EXPORT_MODE", None)

    def test_timeout_validation(self) -> None:
        """Test timeout value validation."""
        # Valid timeout
        os.environ["OTEL_EXPORT_TIMEOUT"] = "5000"
        try:
            settings = LoggingSettings()
            assert settings.otel_export_timeout == 5000
        finally:
            os.environ.pop("OTEL_EXPORT_TIMEOUT", None)

        # Invalid timeout (negative)
        os.environ["OTEL_EXPORT_TIMEOUT"] = "-1000"
        try:
            with pytest.raises(ValidationError):
                LoggingSettings()
        finally:
            os.environ.pop("OTEL_EXPORT_TIMEOUT", None)

        # Invalid timeout (non-numeric)
        os.environ["OTEL_EXPORT_TIMEOUT"] = "abc"
        try:
            with pytest.raises(ValidationError):
                LoggingSettings()
        finally:
            os.environ.pop("OTEL_EXPORT_TIMEOUT", None)

    def test_settings_model_dump(self) -> None:
        """Test settings serialization."""
        os.environ["LOG_LEVEL"] = "WARNING"
        os.environ["OTEL_LOGS_EXPORT_MODE"] = "both"

        try:
            settings = LoggingSettings()
            data: dict[str, Any] = settings.model_dump()

            assert data["log_level"] == "WARNING"
            assert data["otel_logs_export_mode"] == "both"
            assert "otel_export_enabled" in data
            assert data["otel_export_enabled"] is True

        finally:
            os.environ.pop("LOG_LEVEL", None)
            os.environ.pop("OTEL_LOGS_EXPORT_MODE", None)


class TestInterfaceSettings:
    """Test interface settings functionality."""

    def test_default_interface_type(self) -> None:
        """Test default interface type is CLI."""
        from clean_interfaces.types import InterfaceType
        from clean_interfaces.utils.settings import InterfaceSettings

        settings = InterfaceSettings()
        assert settings.interface_type == "cli"
        assert settings.interface_type_enum == InterfaceType.CLI

    def test_interface_type_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test interface type can be set from environment variable."""
        from clean_interfaces.types import InterfaceType
        from clean_interfaces.utils.settings import InterfaceSettings

        monkeypatch.setenv("INTERFACE_TYPE", "cli")
        settings = InterfaceSettings()
        assert settings.interface_type == "cli"
        assert settings.interface_type_enum == InterfaceType.CLI

    def test_invalid_interface_type_raises_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test invalid interface type raises validation error."""
        from clean_interfaces.utils.settings import InterfaceSettings

        monkeypatch.setenv("INTERFACE_TYPE", "invalid")
        with pytest.raises(ValueError, match="Invalid interface type"):
            InterfaceSettings()

    def test_get_interface_settings_singleton(self) -> None:
        """Test get_interface_settings returns singleton instance."""
        from clean_interfaces.utils.settings import (
            get_interface_settings,
            reset_interface_settings,
        )

        # Reset first to ensure clean state
        reset_interface_settings()

        settings1 = get_interface_settings()
        settings2 = get_interface_settings()
        assert settings1 is settings2

        # Clean up
        reset_interface_settings()

    def test_interface_settings_model_dump(self) -> None:
        """Test interface settings can be dumped to dict."""
        from clean_interfaces.utils.settings import InterfaceSettings

        settings = InterfaceSettings()
        data = settings.model_dump()
        assert "interface_type" in data
        assert data["interface_type"] == "cli"
