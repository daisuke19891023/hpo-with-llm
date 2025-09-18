"""Provider specific responses API client implementations."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - optional dependency
    from openai import AzureOpenAI, OpenAI
except ImportError:  # pragma: no cover - handled in runtime logic
    AzureOpenAI = None  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment]

from clean_interfaces.llm.base import LLMClient, LLMConfigurationError

if TYPE_CHECKING:
    from clean_interfaces.utils.settings import LLMSettings


class OpenAIClient(LLMClient):
    """Client targeting the public OpenAI responses API."""

    def __init__(self, settings: LLMSettings) -> None:
        """Initialise the OpenAI client with provided credentials."""
        if not settings.openai_api_key:
            msg = "OPENAI_API_KEY environment variable is required for OpenAI provider"
            raise LLMConfigurationError(msg)

        if OpenAI is None:
            msg = "openai package is required to use the OpenAI provider"
            raise LLMConfigurationError(msg)

        client_kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._client: Any = OpenAI(**client_kwargs)  # type: ignore[operator]
        super().__init__(settings=settings, model=settings.model)

    def _invoke(self, payload: typing.Mapping[str, object]) -> Any:
        return self._client.responses.create(**payload)


class AzureOpenAIClient(LLMClient):
    """Client targeting Azure OpenAI responses deployments."""

    def __init__(self, settings: LLMSettings) -> None:
        """Initialise the Azure OpenAI client and validate required settings."""
        missing: list[str] = []
        if not settings.azure_api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not settings.azure_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not settings.azure_deployment:
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        if missing:
            msg = (
                "Azure OpenAI provider selected but required settings are missing: "
                + ", ".join(missing)
            )
            raise LLMConfigurationError(msg)

        if AzureOpenAI is None:
            msg = "openai package is required to use the Azure OpenAI provider"
            raise LLMConfigurationError(msg)

        assert settings.azure_endpoint is not None
        assert settings.azure_deployment is not None

        self._client: Any = AzureOpenAI(  # type: ignore[operator]
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_endpoint,
        )
        self._deployment = settings.azure_deployment
        super().__init__(settings=settings, model=settings.azure_deployment)

    def _invoke(self, payload: typing.Mapping[str, object]) -> Any:
        request_payload = dict(payload)
        request_payload.setdefault("model", self._deployment)
        return self._client.responses.create(**request_payload)


__all__ = ["AzureOpenAIClient", "OpenAIClient"]
