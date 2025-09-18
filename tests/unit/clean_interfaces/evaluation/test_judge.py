"""Tests for evaluation judges backed by LLM clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from collections.abc import Mapping

import pytest

import clean_interfaces.evaluation.judge as judge_module
from clean_interfaces.evaluation.judge import (
    HeuristicLLMJudge,
    LLMJudgeResult,
    ResponsesAPIJudge,
)
from clean_interfaces.llm.base import (
    LLMFunctionCall,
    LLMGenerationRequest,
    LLMGenerationResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class _RecordedRequest:
    request: LLMGenerationRequest


class StubClient:
    """Stub LLM client returning a predefined structured response."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        """Store the payload to return during generation."""
        self.payload: dict[str, object] = dict(payload)
        self.requests: list[_RecordedRequest] = []

    def generate(self, request: LLMGenerationRequest) -> LLMGenerationResult:
        """Record the request and return a deterministic result."""
        self.requests.append(_RecordedRequest(request=request))
        return LLMGenerationResult(
            response_id="resp",
            model="test-model",
            text="",
            structured_output=self.payload,
            function_call=None,
            usage={},
            raw={"output": []},
        )


class StubFactory:
    """Factory returning a predefined stub client."""

    def __init__(self, client: StubClient) -> None:
        """Persist the stub client used for each create call."""
        self.client = client

    def create(self) -> StubClient:
        """Return the stored stub client instance."""
        return self.client


class RecordingJudge(HeuristicLLMJudge):
    """Fallback judge that records when it is used."""

    def __init__(self) -> None:
        """Initialise the recording fallback judge."""
        super().__init__()
        self.calls: list[str] = []

    def score_plan(self, plan: str, references: Sequence[str]) -> LLMJudgeResult:
        """Record the plan before delegating to the parent implementation."""
        self.calls.append(plan)
        return super().score_plan(plan, references)


def test_responses_judge_parses_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Responses judge should parse structured JSON output from the LLM."""
    payload = {
        "score": 0.92,
        "rationale": "Enable vector tooling",
        "matched_keywords": ["vector"],
        "missing_keywords": [],
        "actions": ["Enable retrieval"],
    }
    client = StubClient(payload)
    factory = StubFactory(client)
    monkeypatch.setattr(judge_module, "LLMClientFactory", lambda: factory)

    judge = ResponsesAPIJudge()
    result = judge.score_plan(
        "Use conservative temperature",
        ["Enable vector search", "Lower temperature"],
    )

    assert result.score == pytest.approx(0.92)
    assert result.rationale == "Enable vector tooling"
    assert result.matched_keywords == ("vector",)
    assert result.missing_keywords == ()
    assert client.requests
    request = client.requests[0].request
    assert request.response_format is not None
    response_format = request.response_format
    assert response_format["type"] == "json_schema"
    assert len(request.input) >= 2
    user_message = request.input[1]
    content_payload_obj = user_message.get("content")
    assert isinstance(content_payload_obj, list)
    assert content_payload_obj
    first_item = cast(dict[str, object], content_payload_obj[0])
    user_content = first_item.get("text")
    assert isinstance(user_content, str)
    assert "Use conservative temperature" in user_content


def test_responses_judge_falls_back_without_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no structured output is returned the fallback judge should run."""

    class EmptyClient(StubClient):
        def __init__(self) -> None:
            super().__init__({})

        def generate(self, request: LLMGenerationRequest) -> LLMGenerationResult:
            self.requests.append(_RecordedRequest(request=request))
            return LLMGenerationResult(
                response_id="resp",
                model="test-model",
                text="No structure",
                structured_output=None,
                function_call=LLMFunctionCall(name="noop", arguments={}),
                usage={},
                raw={"output": []},
            )

    client = EmptyClient()
    factory = StubFactory(client)
    monkeypatch.setattr(judge_module, "LLMClientFactory", lambda: factory)

    fallback = RecordingJudge()
    judge = ResponsesAPIJudge(fallback=fallback)
    result = judge.score_plan("Plan text", ["reference one", "reference two"])

    assert fallback.calls  # fallback used
    assert 0.0 <= result.score <= 1.0
