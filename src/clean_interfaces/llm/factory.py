"""Factory helpers to construct provider specific LLM clients."""

from __future__ import annotations

from clean_interfaces.llm.base import LLMClient, LLMConfigurationError
from clean_interfaces.llm.clients import AzureOpenAIClient, OpenAIClient
from clean_interfaces.utils.settings import LLMProvider, LLMSettings, get_llm_settings


class LLMClientFactory:
    """Factory creating LLM clients based on configured provider."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        """Store an optional settings override for later client creation."""
        self._settings = settings

    def create(self, provider: LLMProvider | str | None = None) -> LLMClient:
        """Instantiate a client for the requested provider."""
        settings = self._settings or get_llm_settings()
        selected = provider or settings.provider
        if not isinstance(selected, LLMProvider):
            selected = LLMProvider(str(selected).lower())

        if selected is LLMProvider.OPENAI:
            return OpenAIClient(settings)
        if selected is LLMProvider.AZURE_OPENAI:
            return AzureOpenAIClient(settings)

        msg = f"Unsupported LLM provider requested: {selected!r}"
        raise LLMConfigurationError(msg)


__all__ = ["LLMClientFactory"]
