"""Text-only adapter for TypeSafe Jev end-of-turn judgments."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from .controller import JevConfigurationError, JevJudgment, JevProviderError, TurnInput
from .prompts import (
    CONVERSATION_SETTING,
    TRANSCRIPT_NOTES,
    judge_questions,
    scores_from_answers,
    state_from_turn,
)

__all__ = [
    "CONVERSATION_SETTING",
    "TRANSCRIPT_NOTES",
    "TypeSafeJevJudge",
    "state_from_turn",
]


class TypeSafeJevJudge:
    """Calls Jev with text/context only and returns four typed probabilities.

    The SDK is imported only when a call is made, so controlled tests and the
    baseline mode do not require TypeSafe credentials or the SDK package.
    """

    def __init__(self, *, model: str = "jev-latest", timeout_ms: int = 300) -> None:
        self.model = model
        self.timeout_ms = timeout_ms

    async def __call__(self, turn: TurnInput) -> JevJudgment:
        if not os.environ.get("TYPESAFE_API_KEY", "").strip():
            raise JevConfigurationError("TYPESAFE_API_KEY is not configured")
        include_taking_floor = bool((turn.assistant_current_speech or "").strip())
        try:
            from typesafe_sdk import AsyncTypeSafeClient, Noul, NoulCriteria, RetryPolicy
        except ImportError as exc:
            raise JevConfigurationError("typesafe-sdk is not installed") from exc

        timeout_seconds = self.timeout_ms / 1000
        retry = RetryPolicy(max_retries=0)

        def _noul(name: str, questions: Mapping[str, Any]) -> Noul:
            entry = questions[name]
            return Noul(
                instructions=entry["instructions"],
                criteria=NoulCriteria(true=entry["criteria"]["true"], false=entry["criteria"]["false"]),
            )

        shared = judge_questions(include_taking_floor=include_taking_floor)
        questions = {name: _noul(name, shared) for name in shared}
        state = state_from_turn(turn)

        started = time.perf_counter()
        try:
            async with AsyncTypeSafeClient(
                model=self.model,
                timeout=timeout_seconds,
                retry=retry,
            ) as client:
                response = await client.system_one(
                    state=state,
                    questions=questions,
                    timeout=timeout_seconds,
                    retry=retry,
                )
        except Exception as exc:
            # Never propagate raw provider text; it can include request data.
            raise JevProviderError(type(exc).__name__) from None

        answers = getattr(response, "answers", None)
        if not isinstance(answers, Mapping):
            raise JevProviderError("invalid_response")
        try:
            scores = scores_from_answers(answers)
        except (KeyError, AttributeError, TypeError, ValueError):
            raise JevProviderError("invalid_response") from None
        if any(not 0.0 <= value <= 1.0 for value in scores.values()):
            raise JevProviderError("invalid_response")

        usage = getattr(response, "usage", None)
        return JevJudgment(
            **scores,
            model=getattr(response, "model", self.model),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
