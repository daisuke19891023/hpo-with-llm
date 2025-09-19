"""Tests for responses API client implementations and factory."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import clean_interfaces.llm.clients as clients_module
import clean_interfaces.llm.factory as factory_module
from clean_interfaces.llm.base import (
    LLMConfigurationError,
    LLMGenerationRequest,
)
from clean_interfaces.llm.clients import AzureOpenAIClient, OpenAIClient
from clean_interfaces.llm.factory import LLMClientFactory
from clean_interfaces.utils.settings import LLMProvider, LLMSettings


class _StubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


def _build_payload() -> dict[str, Any]:
    return {
        "id": "resp_123",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "parsed": {
                                "score": 0.85,
                                "rationale": "Great coverage",
                                "matched_keywords": ["vector"],
                                "missing_keywords": ["report"],
                            },
                        },
                    },
                    {
                        "type": "tool_call",
                        "tool_call": {
                            "type": "function",
                            "function": {
                                "name": "select_tool",
                                "arguments": '{"tool": "search"}',
                            },
                        },
                    },
                ],
            },
        ],
        "usage": {"input_tokens": 120, "output_tokens": 50, "total_tokens": 170},
    }


def test_openai_client_parses_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI client should invoke the responses API and normalise output."""
    payload = _build_payload()
    created_clients: list[Any] = []

    class StubOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.calls: list[dict[str, Any]] = []
            self.responses = SimpleNamespace(create=self._create)
            created_clients.append(self)

        def _create(self, **kwargs: Any) -> _StubResponse:
            self.calls.append(kwargs)
            return _StubResponse(payload)

    monkeypatch.setattr(clients_module, "OpenAI", StubOpenAI)

    settings = LLMSettings(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        temperature=0.3,
        max_output_tokens=256,
        openai_api_key="sk-test",
    )
    client = OpenAIClient(settings)

    request = LLMGenerationRequest(
        input=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        response_format={"type": "json_schema"},
        tools=[{"type": "function", "function": {"name": "select_tool"}}],
        tool_choice={"type": "function", "function": {"name": "select_tool"}},
        metadata={"trial": "demo"},
        extra_options={"reasoning": {"effort": "medium"}},
    )

    result = client.generate(request)

    stub = created_clients[0]
    assert stub.kwargs["api_key"] == "sk-test"
    call = stub.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == pytest.approx(0.3)
    assert call["max_output_tokens"] == 256
    assert call["tools"]
    assert call["tool_choice"]
    assert call["metadata"] == {"trial": "demo"}
    assert call["reasoning"] == {"effort": "medium"}

    assert result.text == "Hello world"
    assert result.structured_output == {
        "score": 0.85,
        "rationale": "Great coverage",
        "matched_keywords": ["vector"],
        "missing_keywords": ["report"],
    }
    assert result.function_call is not None
    assert result.function_call.name == "select_tool"
    assert result.function_call.arguments == {"tool": "search"}
    assert result.usage["total_tokens"] == 170


def test_azure_client_requires_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Azure client should validate required environment configuration."""
    payload = _build_payload()
    created_clients: list[Any] = []

    class StubAzure:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.calls: list[dict[str, Any]] = []
            self.responses = SimpleNamespace(create=self._create)
            created_clients.append(self)

        def _create(self, **kwargs: Any) -> _StubResponse:
            self.calls.append(kwargs)
            return _StubResponse(payload)

    monkeypatch.setattr(clients_module, "AzureOpenAI", StubAzure)

    settings = LLMSettings(
        provider=LLMProvider.AZURE_OPENAI,
        model="unused",
        azure_api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        azure_api_version="2024-08-01-preview",
    )
    client = AzureOpenAIClient(settings)
    result = client.generate(
        LLMGenerationRequest(
            input=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        ),
    )

    stub = created_clients[0]
    assert stub.kwargs["api_key"] == "azure-key"
    assert stub.kwargs["azure_endpoint"] == "https://example.openai.azure.com"
    call = stub.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert result.text == "Hello world"

    invalid_settings = LLMSettings(
        provider=LLMProvider.AZURE_OPENAI,
        model="unused",
    )
    with pytest.raises(LLMConfigurationError):
        AzureOpenAIClient(invalid_settings)


def test_factory_selects_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory should create clients based on provider configuration."""

    class StubOpenAIClient:
        def __init__(self, settings: LLMSettings) -> None:
            self.settings = settings

    class StubAzureClient:
        def __init__(self, settings: LLMSettings) -> None:
            self.settings = settings

    monkeypatch.setattr(
        factory_module,
        "OpenAIClient",
        StubOpenAIClient,
    )
    monkeypatch.setattr(
        factory_module,
        "AzureOpenAIClient",
        StubAzureClient,
    )

    openai_settings = LLMSettings(
        provider=LLMProvider.OPENAI,
        openai_api_key="sk",
    )
    factory = LLMClientFactory(settings=openai_settings)
    client = factory.create()
    assert isinstance(client, StubOpenAIClient)

    azure_settings = LLMSettings(
        provider=LLMProvider.OPENAI,
        openai_api_key="sk",
    )
    factory = LLMClientFactory(settings=azure_settings)
    client = factory.create("azure_openai")
    assert isinstance(client, StubAzureClient)
