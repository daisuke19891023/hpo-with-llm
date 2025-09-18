"""Application settings management using Pydantic Settings.

This module provides centralized configuration management for the application,
with support for environment variables and validation.
"""

from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
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


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"


class LLMSettings(BaseSettings):
    """Configuration for LLM access retrieved from environment variables."""

    instance: ClassVar[Any] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        validation_alias=AliasChoices("LLM_PROVIDER", "provider"),
        description="Which LLM provider to use for requests.",
    )
    model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_MODEL", "model"),
        description="Default model or deployment name to target.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        validation_alias=AliasChoices("LLM_TEMPERATURE", "temperature"),
        description="Default sampling temperature used for generations.",
    )
    max_output_tokens: int = Field(
        default=512,
        ge=1,
        validation_alias=AliasChoices(
            "LLM_MAX_OUTPUT_TOKENS",
            "LLM_MAX_TOKENS",
            "max_output_tokens",
        ),
        description="Default maximum number of output tokens to request.",
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "LLM_OPENAI_API_KEY",
            "openai_api_key",
        ),
        description="API key used for the OpenAI provider.",
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_BASE_URL",
            "LLM_OPENAI_BASE_URL",
            "openai_base_url",
        ),
        description="Optional override for the OpenAI base URL.",
    )
    azure_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_OPENAI_API_KEY",
            "LLM_AZURE_OPENAI_API_KEY",
            "azure_api_key",
        ),
        description="API key used for Azure OpenAI deployments.",
    )
    azure_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_OPENAI_ENDPOINT",
            "LLM_AZURE_OPENAI_ENDPOINT",
            "azure_endpoint",
        ),
        description="Azure OpenAI endpoint URL.",
    )
    azure_deployment: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_OPENAI_DEPLOYMENT",
            "LLM_AZURE_OPENAI_DEPLOYMENT",
            "azure_deployment",
        ),
        description="Azure OpenAI deployment name to target.",
    )
    azure_api_version: str = Field(
        default="2024-08-01-preview",
        validation_alias=AliasChoices(
            "AZURE_OPENAI_API_VERSION",
            "LLM_AZURE_OPENAI_API_VERSION",
            "azure_api_version",
        ),
        description="Azure OpenAI API version.",
    )

    @model_validator(mode="after")
    def _normalise_provider(self) -> "LLMSettings":
        """Ensure provider aliases normalise to canonical enum values."""
        if isinstance(self.provider, LLMProvider):
            return self

        normalised = str(self.provider).strip().lower()
        try:
            self.provider = LLMProvider(normalised)
        except ValueError as exc:  # pragma: no cover - defensive branch
            message = f"Unsupported LLM provider: {self.provider!r}"
            raise ValueError(message) from exc
        return self


def get_llm_settings() -> LLMSettings:
    """Return cached LLM settings resolved from the environment."""
    if LLMSettings.instance is None:
        LLMSettings.instance = LLMSettings()
    return LLMSettings.instance


def reset_llm_settings() -> None:
    """Clear cached LLM settings to force a reload on next access."""
    LLMSettings.instance = None
