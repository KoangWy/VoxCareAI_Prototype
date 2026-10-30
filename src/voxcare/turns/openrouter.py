"""Text-only adapter for OpenRouter Decisions API turn judgments.

OpenRouter serves System One decision models (for example Inception's
Mercury Decide) behind `POST /api/alpha/decisions` with the same state +
typed-questions schema TypeSafe Jev uses. This adapter mirrors
`TypeSafeJevJudge`: same shared state, same four Noul questions, same error
discipline (configuration vs provider, raw details never logged).
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from ..config import DEFAULT_OPENROUTER_MODEL
from .controller import JevConfigurationError, JevJudgment, JevProviderError, TurnInput
from .prompts import judge_questions, scores_from_payload, state_from_turn

DEFAULT_OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

logger = logging.getLogger("voxcare.turns.openrouter")


class OpenRouterDecisionsJudge:
    """Calls an OpenRouter decisions model with text/context only.

    Raises `JevConfigurationError` without touching the network when
    `OPENROUTER_API_KEY` is absent, and `JevProviderError` (type name or
    status only — never response text, which can echo request data) for
    every transport or response-shape failure, matching `TypeSafeJevJudge`.

    `429` and `5xx` responses map to `retryable_provider_error` so the host
    can distinguish a congested route from a malformed one; timeouts map to
    `jev_timeout`. Latency is logged per call (fields only, never payloads);
    the shared client is closed with `aclose()` at application shutdown.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENROUTER_MODEL,
        timeout_ms: int = 2500,
        endpoint: str = DEFAULT_OPENROUTER_DECISIONS_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.timeout_ms = timeout_ms
        self.endpoint = endpoint
        self._transport = transport
        self._http: httpx.AsyncClient | None = None

    async def __call__(self, turn: TurnInput) -> JevJudgment:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise JevConfigurationError("OPENROUTER_API_KEY is not configured")
        include_taking_floor = bool((turn.assistant_current_speech or "").strip())
        started = time.perf_counter()
        try:
            # One client per judge instance: reuses the TLS connection instead of
            # paying a fresh handshake (~0.2-0.3 s) on every turn judgment.
            if self._http is None:
                self._http = httpx.AsyncClient(timeout=self.timeout_ms / 1000, transport=self._transport)
            response = await self._http.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.model,
                    "state": state_from_turn(turn),
                    "questions": judge_questions(include_taking_floor=include_taking_floor),
                },
            )
        except TimeoutError:
            raise JevProviderError("jev_timeout") from None
        except Exception as exc:
            if exc.__class__.__name__ in {"TimeoutException", "ConnectTimeout", "ReadTimeout", "PoolTimeout"}:
                raise JevProviderError("jev_timeout") from None
            raise JevProviderError(type(exc).__name__) from None
        latency_ms = round((time.perf_counter() - started) * 1000)
        logger.info(
            "openrouter judgment answered", extra={"status_code": response.status_code, "latency_ms": latency_ms}
        )
        if response.status_code == 429 or 500 <= response.status_code <= 599:
            raise JevProviderError("retryable_provider_error")
        if response.status_code != 200:
            raise JevProviderError(f"http_{response.status_code}")
        try:
            body = response.json()
            scores = scores_from_payload(body)
        except Exception:
            raise JevProviderError("invalid_response") from None
        if any(not 0.0 <= value <= 1.0 for value in scores.values()):
            raise JevProviderError("invalid_response")
        usage = body.get("usage") or {}
        return JevJudgment(
            **scores,
            model=str(body.get("model") or self.model),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        """Close the shared HTTP client; safe to call more than once."""
        client, self._http = self._http, None
        if client is not None:
            await client.aclose()
