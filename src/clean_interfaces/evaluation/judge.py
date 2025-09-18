"""Heuristic LLM judge used to score natural language plans."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol, cast
from collections.abc import Mapping, Sequence

from clean_interfaces.base import BaseComponent
from clean_interfaces.llm import (
    LLMClient,
    LLMClientFactory,
    LLMConfigurationError,
    LLMGenerationRequest,
)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_KEYWORD_FREQUENCY_THRESHOLD = 2
_MANUAL_KEYWORDS = {
    "vector",
    "search",
    "toolchain",
    "tooling",
    "compliance",
    "faq",
    "keyword",
    "brainstorming",
    "reports",
    "report",
    "summary",
    "summaries",
    "executive",
    "grounded",
    "creative",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lower-case keywords with stop words removed."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in _STOPWORDS]


@dataclass(frozen=True, slots=True)
class LLMJudgeResult:
    """Summary of the heuristic LLM judge evaluation."""

    score: float
    rationale: str
    matched_keywords: tuple[str, ...]
    missing_keywords: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable representation of the result."""
        return {
            "score": self.score,
            "rationale": self.rationale,
            "matched_keywords": list(self.matched_keywords),
            "missing_keywords": list(self.missing_keywords),
        }


class LLMJudgeProtocol(Protocol):
    """Protocol implemented by judge backends."""

    def score_plan(
        self,
        plan: str,
        references: Sequence[str],
    ) -> LLMJudgeResult:
        """Evaluate a plan against reference plans and return a score."""

        ...


class HeuristicLLMJudge(BaseComponent):
    """Lightweight keyword-based judge approximating LLM feedback."""

    def score_plan(
        self,
        plan: str,
        references: Sequence[str],
    ) -> LLMJudgeResult:
        """Score a plan using keyword coverage and diversity heuristics."""
        plan_tokens = _tokenize(plan)
        plan_counter = Counter(plan_tokens)
        reference_counter: Counter[str] = Counter()
        for reference in references:
            reference_counter.update(_tokenize(reference))

        keywords = {
            token
            for token, count in reference_counter.items()
            if count >= _KEYWORD_FREQUENCY_THRESHOLD or token in _MANUAL_KEYWORDS
        }

        if not keywords:
            # Fallback to plan richness when no reference keywords are available
            diversity = min(1.0, len(plan_counter) / 10.0)
            rationale = (
                "No reference keywords provided; "
                "scored using plan diversity only."
            )
            return LLMJudgeResult(
                score=diversity,
                rationale=rationale,
                matched_keywords=(),
                missing_keywords=(),
            )

        matched = sorted(token for token in keywords if token in plan_counter)
        missing = sorted(token for token in keywords if token not in plan_counter)

        coverage = len(matched) / len(keywords)
        diversity = min(1.0, len(plan_counter) / (len(keywords) + 5))
        score = max(0.0, min(1.0, 0.7 * coverage + 0.3 * diversity))

        rationale = (
            f"Matched {len(matched)} of {len(keywords)} reference keywords. "
            f"Missing: {', '.join(missing) if missing else 'none'}."
        )

        self.logger.debug(
            "LLM judge scoring",
            plan_length=len(plan_tokens),
            coverage=coverage,
            diversity=diversity,
            score=score,
        )

        return LLMJudgeResult(
            score=score,
            rationale=rationale,
            matched_keywords=tuple(matched),
            missing_keywords=tuple(missing),
        )


_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator analysing plans for retrieval-augmented LLM "
    "systems. Provide concise, evidence-based feedback."
)

_JUDGE_RESPONSE_FORMAT: Mapping[str, object] = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_plan_evaluation",
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "rationale": {"type": "string"},
                "matched_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "missing_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            "required": [
                "score",
                "rationale",
                "matched_keywords",
                "missing_keywords",
            ],
            "additionalProperties": False,
        },
    },
}


class ResponsesAPIJudge(BaseComponent):
    """Judge implementation backed by an actual LLM via the responses API."""

    def __init__(
        self,
        *,
        factory: LLMClientFactory | None = None,
        fallback: LLMJudgeProtocol | None = None,
    ) -> None:
        """Initialise the judge with an optional factory and fallback."""
        super().__init__()
        self._factory = factory or LLMClientFactory()
        self._fallback = fallback or HeuristicLLMJudge()
        self._client: LLMClient | None = None

    def _ensure_client(self) -> LLMClient:
        if self._client is None:
            self._client = self._factory.create()
        return self._client

    def score_plan(
        self,
        plan: str,
        references: Sequence[str],
    ) -> LLMJudgeResult:
        """Evaluate the plan using the configured LLM or the heuristic fallback."""
        if not references:
            return self._fallback.score_plan(plan, references)

        try:
            client = self._ensure_client()
        except LLMConfigurationError as exc:
            self.logger.info(
                "LLM judge unavailable; falling back to heuristic scorer",
                error=str(exc),
            )
            return self._fallback.score_plan(plan, references)

        references_block = "\n".join(f"- {reference}" for reference in references)
        user_prompt = (
            "Evaluate the following plan against provided reference strategies "
            "and return JSON with score, rationale, matched_keywords, and "
            "missing_keywords.\n"
            f"Plan: {plan}\n"
            "Reference plans:\n"
            f"{references_block}"
        )

        request = LLMGenerationRequest(
            input=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": _JUDGE_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                },
            ],
            response_format=_JUDGE_RESPONSE_FORMAT,
            metadata={"judge": "rag_plan"},
        )

        try:
            result = client.generate(request)
        except Exception as exc:  # pragma: no cover - defensive network path
            self.logger.warning(
                "LLM judge invocation failed; falling back to heuristic scorer",
                error=str(exc),
            )
            return self._fallback.score_plan(plan, references)

        payload: Mapping[str, object] | None = None
        if result.structured_output is not None:
            payload = dict(result.structured_output)
        elif result.function_call is not None:
            payload = dict(result.function_call.arguments)

        if payload is None or not payload:
            self.logger.warning(
                "LLM judge did not return structured output; using heuristic result",
            )
            return self._fallback.score_plan(plan, references)

        return _payload_to_result(payload)


def _payload_to_result(payload: Mapping[str, object]) -> LLMJudgeResult:
    """Convert structured output from the LLM into :class:`LLMJudgeResult`."""
    score_raw: Any = payload.get("score", 0.0)
    try:
        score = float(score_raw)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        score = 0.0
    score = max(0.0, min(1.0, score))

    default_rationale = "Model returned no rationale."
    rationale = str(payload.get("rationale", "")).strip() or default_rationale

    def _normalise_keywords(value: object) -> tuple[str, ...]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = cast(Sequence[object], value)
            return tuple(
                str(item)
                for item in items
                if isinstance(item, (str, bytes))
            )
        return ()

    matched = _normalise_keywords(payload.get("matched_keywords"))
    missing = _normalise_keywords(payload.get("missing_keywords"))

    return LLMJudgeResult(
        score=score,
        rationale=rationale,
        matched_keywords=matched,
        missing_keywords=missing,
    )


__all__ = [
    "HeuristicLLMJudge",
    "LLMJudgeProtocol",
    "LLMJudgeResult",
    "ResponsesAPIJudge",
]
