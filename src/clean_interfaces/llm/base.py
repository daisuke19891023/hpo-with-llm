"""Shared dataclasses and base client for responses API integrations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, cast
from collections.abc import Mapping, Sequence

from clean_interfaces.base import BaseComponent

if TYPE_CHECKING:
    from clean_interfaces.utils.settings import LLMSettings


class LLMConfigurationError(RuntimeError):
    """Raised when an LLM client cannot be instantiated due to configuration."""


@dataclass(slots=True, frozen=True)
class LLMFunctionCall:
    """Structured representation of a function call issued by the model."""

    name: str
    arguments: Mapping[str, object]


@dataclass(slots=True)
class LLMGenerationRequest:
    """Structured payload describing an LLM generation request."""

    input: Sequence[Mapping[str, object]]
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    response_format: Mapping[str, object] | None = None
    tools: Sequence[Mapping[str, object]] | None = None
    tool_choice: Mapping[str, object] | None = None
    metadata: Mapping[str, object] | None = None
    extra_options: Mapping[str, object] | None = None

    def to_payload(
        self,
        *,
        default_model: str,
        default_temperature: float,
        default_max_output_tokens: int,
    ) -> dict[str, object]:
        """Convert the request into keyword arguments for the responses API."""
        if not self.input:
            msg = "LLMGenerationRequest requires at least one message in input"
            raise ValueError(msg)

        payload: dict[str, object] = {
            "model": self.model or default_model,
            "input": [_copy_str_mapping(message) for message in self.input],
            "temperature": (
                self.temperature
                if self.temperature is not None
                else default_temperature
            ),
            "max_output_tokens": (
                self.max_output_tokens
                if self.max_output_tokens is not None
                else default_max_output_tokens
            ),
        }

        if self.response_format is not None:
            payload["response_format"] = _copy_str_mapping(self.response_format)
        if self.tools is not None:
            payload["tools"] = [_copy_str_mapping(tool) for tool in self.tools]
        if self.tool_choice is not None:
            payload["tool_choice"] = _copy_str_mapping(self.tool_choice)
        if self.metadata is not None:
            payload["metadata"] = _copy_str_mapping(self.metadata)
        if self.extra_options is not None:
            payload.update(_copy_str_mapping(self.extra_options))

        return payload


@dataclass(slots=True, frozen=True)
class LLMGenerationResult:
    """Normalised representation of a responses API generation."""

    response_id: str
    model: str | None
    text: str | None
    structured_output: Mapping[str, object] | None
    function_call: LLMFunctionCall | None
    usage: Mapping[str, object]
    raw: Mapping[str, object] = field(repr=False)


class LLMClient(BaseComponent, ABC):
    """Base client wrapping provider specific responses API integrations."""

    def __init__(
        self,
        *,
        settings: LLMSettings,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        """Store settings and default overrides for concrete clients."""
        super().__init__()
        self._settings = settings
        self._model = model or settings.model
        self._default_temperature = temperature or settings.temperature
        self._default_max_output_tokens = (
            max_output_tokens or settings.max_output_tokens
        )

    def generate(self, request: LLMGenerationRequest) -> LLMGenerationResult:
        """Execute a generation request and normalise the result."""
        payload = request.to_payload(
            default_model=self._model,
            default_temperature=self._default_temperature,
            default_max_output_tokens=self._default_max_output_tokens,
        )
        response = self._invoke(payload)
        return self._parse_response(response)

    @abstractmethod
    def _invoke(self, payload: Mapping[str, object]) -> Any:
        """Execute a responses API call using provider specific transport."""

    def _parse_response(self, response: Any) -> LLMGenerationResult:
        """Normalise responses API payloads into :class:`LLMGenerationResult`."""
        if hasattr(response, "model_dump"):
            data = cast(Mapping[str, object], response.model_dump())
        elif isinstance(response, Mapping):
            data = cast(Mapping[str, object], response)
        else:  # pragma: no cover - defensive path for unexpected SDK payloads
            msg = f"Unsupported response payload type: {type(response)!r}"
            raise TypeError(msg)
        return self._parse_response_data(data)

    @staticmethod
    def _parse_response_data(data: Mapping[str, object]) -> LLMGenerationResult:
        """Extract text, structured output, and function calls from payloads."""
        parser = _ResponseContentParser()
        parser.consume_outputs(data.get("output"))

        text = parser.final_text()
        usage = _coerce_mapping(data.get("usage")) or {}

        model_value = data.get("model")
        model = model_value if isinstance(model_value, str) else None

        return LLMGenerationResult(
            response_id=str(data.get("id", "")),
            model=model,
            text=text,
            structured_output=parser.structured_output,
            function_call=parser.function_call,
            usage=usage,
            raw=dict(data),
        )


class _ResponseContentParser:
    """Helper that extracts structured pieces from responses payloads."""

    def __init__(self) -> None:
        self._text_parts: list[str] = []
        self.structured_output: Mapping[str, object] | None = None
        self.function_call: LLMFunctionCall | None = None

    def consume_outputs(self, outputs: Any) -> None:
        """Consume the list of outputs emitted by the responses API."""
        if not isinstance(outputs, Sequence):
            return
        sequence = cast(Sequence[object], outputs)
        for entry in sequence:
            if isinstance(entry, Mapping):
                mapping_item = _coerce_mapping(entry)
                if mapping_item is not None:
                    self._consume_output_item(mapping_item)

    def final_text(self) -> str | None:
        """Return the normalised textual content collected so far."""
        combined = "\n".join(part for part in self._text_parts if part).strip()
        return combined or None

    def _consume_output_item(self, item: Mapping[str, object]) -> None:
        contents = item.get("content")
        if not isinstance(contents, Sequence):
            return
        content_sequence = cast(Sequence[object], contents)
        for entry in content_sequence:
            if isinstance(entry, Mapping):
                mapping_content = _coerce_mapping(entry)
                if mapping_content is not None:
                    self._consume_content(mapping_content)

    def _consume_content(self, content: Mapping[str, object]) -> None:
        content_type = str(content.get("type", ""))
        handlers: dict[str, Callable[[Mapping[str, object]], None]] = {
            "text": self._handle_text,
            "output_text": self._handle_text,
            "json_schema": self._handle_json_schema,
            "tool_call": self._handle_tool_call,
            "function_call": self._handle_tool_call,
        }
        handler = handlers.get(content_type)
        if handler is None:
            return
        handler(content)

    def _handle_text(self, content: Mapping[str, object]) -> None:
        text = content.get("text")
        if isinstance(text, str):
            self._text_parts.append(text)

    def _handle_json_schema(self, content: Mapping[str, object]) -> None:
        schema_payload = _coerce_mapping(content.get("json_schema"))
        if schema_payload is None:
            return
        parsed = self._extract_schema_payload(schema_payload)
        if isinstance(parsed, Mapping):
            self.structured_output = dict(parsed)

    def _handle_tool_call(self, content: Mapping[str, object]) -> None:
        tool_payload = _coerce_mapping(content.get("tool_call")) or content
        function_payload = _coerce_mapping(tool_payload.get("function"))
        if function_payload is None:
            return
        name = function_payload.get("name")
        if not isinstance(name, str):
            return
        arguments = self._normalise_arguments(function_payload.get("arguments"))
        self.function_call = LLMFunctionCall(name=name, arguments=arguments)

    @staticmethod
    def _extract_schema_payload(
        schema_payload: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        parsed: object | None = schema_payload.get("parsed")
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if parsed is None and "output" in schema_payload:
            parsed = schema_payload.get("output")
        if parsed is None and "json" in schema_payload:
            maybe_json = schema_payload.get("json")
            if isinstance(maybe_json, str):
                try:
                    parsed = json.loads(maybe_json)
                except json.JSONDecodeError:
                    return None
        return _coerce_mapping(parsed)

    @staticmethod
    def _normalise_arguments(argument_payload: Any) -> Mapping[str, object]:
        if isinstance(argument_payload, Mapping):
            coerced = _coerce_mapping(argument_payload)
            return coerced or {}
        if isinstance(argument_payload, str):
            try:
                decoded = json.loads(argument_payload)
            except json.JSONDecodeError:
                return {"raw": argument_payload}
            if isinstance(decoded, Mapping):
                coerced = _coerce_mapping(decoded)
                return coerced or {}
            return {"raw": decoded}
        return {}


def _copy_str_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    """Return a shallow copy of a mapping with string keys."""

    return {key: mapping[key] for key in mapping}


def _coerce_mapping(value: object) -> dict[str, object] | None:
    """Convert a mapping-like object to a ``dict[str, object]`` when possible."""

    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for key in value:
        if not isinstance(key, str):
            return None
        result[key] = value[key]
    return result

