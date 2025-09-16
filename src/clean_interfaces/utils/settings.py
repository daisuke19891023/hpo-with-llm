"""Application settings management using Pydantic Settings.

This module provides centralized configuration management for the application,
with support for environment variables and validation.
"""

from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OTelExportMode(str, Enum):
    """OpenTelemetry log export modes."""

    FILE = "file"
    OTLP = "otlp"
    BOTH = "both"


class LoggingSettings(BaseSettings):
    """Logging configuration settings.

    All settings can be configured via environment variables.
    """

    instance: ClassVar[Any] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Basic logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    log_format: Literal["json", "console", "plain"] = Field(
        default="json",
        description="Log output format",
    )

    log_file_path: str | None = Field(
        default=None,
        description="Path to log file for local file logging",
    )

    # OpenTelemetry settings
    otel_logs_export_mode: OTelExportMode = Field(
        default=OTelExportMode.FILE,
        description="OpenTelemetry logs export mode: file, otlp, or both",
    )

    otel_endpoint: str = Field(
        default="http://localhost:4317",
        description="OpenTelemetry collector endpoint",
    )

    otel_service_name: str = Field(
        default="python-app",
        description="Service name for OpenTelemetry",
    )

    otel_export_timeout: int = Field(
        default=30000,
        description="OpenTelemetry export timeout in milliseconds",
        ge=1,
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level value."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            msg = f"Invalid log level: {v}. Must be one of {valid_levels}"
            raise ValueError(msg)
        return v.upper()

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format value."""
        valid_formats = {"json", "console", "plain"}
        if v.lower() not in valid_formats:
            msg = f"Invalid log format: {v}. Must be one of {valid_formats}"
            raise ValueError(msg)
        return v.lower()

    @field_validator("otel_export_timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v <= 0:
            msg = "Timeout must be a positive integer"
            raise ValueError(msg)
        return v

    @property
    def otel_export_enabled(self) -> bool:
        """Check if OpenTelemetry export is enabled."""
        return self.otel_logs_export_mode in (OTelExportMode.OTLP, OTelExportMode.BOTH)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Dump model including computed properties."""
        data = super().model_dump(**kwargs)
        data["otel_export_enabled"] = self.otel_export_enabled
        return data


class InterfaceSettings(BaseSettings):
    """Interface configuration settings."""

    instance: ClassVar[Any] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    interface_type: str = Field(
        default="cli",
        description="Type of interface to use (cli, restapi)",
    )

    @field_validator("interface_type")
    @classmethod
    def validate_interface_type(cls, v: str) -> str:
        """Validate interface type value."""
        from clean_interfaces.types import InterfaceType

        try:
            # Validate that it's a valid interface type
            InterfaceType(v.lower())
            return v.lower()
        except ValueError:
            valid_types = [t.value for t in InterfaceType]
            msg = f"Invalid interface type: {v}. Must be one of {valid_types}"
            raise ValueError(msg) from None

    @property
    def interface_type_enum(self) -> Any:
        """Get interface type as enum."""
        from clean_interfaces.types import InterfaceType

        return InterfaceType(self.interface_type)


def get_settings() -> LoggingSettings:
    """Get the global settings instance.

    Returns:
        LoggingSettings: The settings instance

    """
    if LoggingSettings.instance is None:
        LoggingSettings.instance = LoggingSettings()
    return LoggingSettings.instance


def reset_settings() -> None:
    """Reset the global settings instance.

    This is mainly useful for testing.
    """
    LoggingSettings.instance = None


def get_interface_settings() -> InterfaceSettings:
    """Get the global interface settings instance.

    Returns:
        InterfaceSettings: The interface settings instance

    """
    if InterfaceSettings.instance is None:
        InterfaceSettings.instance = InterfaceSettings()
    return InterfaceSettings.instance


def reset_interface_settings() -> None:
    """Reset the global interface settings instance.

    This is mainly useful for testing.
    """
    InterfaceSettings.instance = None
