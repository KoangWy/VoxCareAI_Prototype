"""Conversation adapters: OpenAI Responses, chat-completions gateways, Anthropic fallback, and the cascade.

Every adapter implements the same ``plan_turn`` contract: return every tool
call the model emitted, in order (empty list = no tool), capped at
``MAX_TOOL_CALLS_PER_TURN``. The models never write the spoken answer; the
session boundary renders it from tool results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from anthropic import AsyncAnthropic
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from ..contracts import MAX_TOOL_CALLS_PER_TURN, PlannedToolCall
from .base import SYSTEM_PROMPT, ProviderUnavailable, chat_completions_tools, chat_messages, responses_tools

logger = logging.getLogger("voxcare.providers")

_REQUEST_TIMEOUT_S = 12.0
_RESPONSES_MAX_OUTPUT_TOKENS = 250


def _is_unsupported_parameter(exc: Exception, parameter: str) -> bool:
    """True when a gateway rejected an optional request parameter (unknown/extra argument)."""
    if isinstance(exc, TypeError) and parameter in str(exc):
        return True
    if isinstance(exc, APIStatusError):
        try:
            detail = str(exc.response.json())
        except Exception:
            detail = str(exc)
        lowered = detail.lower()
        return parameter.lower() in lowered and any(
            marker in lowered for marker in ("unknown", "unrecognized", "unsupported", "unexpected", "extra", "invalid")
        )
    return False


def _parse_arguments(raw: object) -> dict[str, Any]:
    try:
        arguments = json.loads(raw or "{}")  # type: ignore[arg-type]
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid function arguments") from exc
    if not isinstance(arguments, dict):
        raise TypeError("invalid function arguments")
    return arguments


def _capped(calls: list[PlannedToolCall]) -> list[PlannedToolCall]:
    if len(calls) > MAX_TOOL_CALLS_PER_TURN:
        logger.info("parallel tool calls truncated", extra={"emitted": len(calls), "cap": MAX_TOOL_CALLS_PER_TURN})
    return calls[:MAX_TOOL_CALLS_PER_TURN]


class OpenAIResponsesAdapter:
    """Design transport: OpenAI Responses API with strict allowlisted functions."""

    label = "OpenAI Responses"

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=_REQUEST_TIMEOUT_S, max_retries=0)

    async def plan_turn(
        self, *, user_text: str, history: Sequence[Mapping[str, str]], available_tools: Sequence[Mapping[str, object]]
    ) -> list[PlannedToolCall]:
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": chat_messages(user_text, history),
            "tools": responses_tools(available_tools),
            "tool_choice": "auto",
            "reasoning": {"effort": "none"},
            "max_output_tokens": _RESPONSES_MAX_OUTPUT_TOKENS,
            "stream": True,
            "parallel_tool_calls": False,
        }
        try:
            stream = await self.client.responses.create(**payload)
        except (APIStatusError, TypeError) as exc:
            if not _is_unsupported_parameter(exc, "parallel_tool_calls"):
                raise
            payload.pop("parallel_tool_calls")
            stream = await self.client.responses.create(**payload)
        calls: list[PlannedToolCall] = []
        async for event in stream:
            if getattr(event, "type", None) != "response.output_item.done":
                continue
            item = getattr(event, "item", None)
            if getattr(item, "type", None) != "function_call":
                continue
            calls.append(PlannedToolCall(item.name, _parse_arguments(item.arguments), getattr(item, "call_id", "")))
        return _capped(calls)


class OpenAIChatCompletionsAdapter:
    """Chat-completions transport for OpenAI-compatible gateways (demo deviation from the Responses design).

    One stable ``x-opencode-session`` id per instance (one VoxCare session)
    lets the gateway reuse prompt caching; history is still sent in full.
    """

    label = "OpenAI Chat Completions"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        session_header: str | None = None,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        headers = {"x-opencode-session": session_header or str(uuid.uuid4())}
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=_REQUEST_TIMEOUT_S, max_retries=0, default_headers=headers
        )

    async def plan_turn(
        self, *, user_text: str, history: Sequence[Mapping[str, str]], available_tools: Sequence[Mapping[str, object]]
    ) -> list[PlannedToolCall]:
        """Sends ``parallel_tool_calls=False``; a gateway that rejects it gets the same request without it."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *chat_messages(user_text, history)],
            "tools": chat_completions_tools(available_tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if self.reasoning_effort:
            payload["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        try:
            stream = await self.client.chat.completions.create(**payload)
        except (APIStatusError, TypeError) as exc:
            if not _is_unsupported_parameter(exc, "parallel_tool_calls"):
                raise
            payload.pop("parallel_tool_calls")
            stream = await self.client.chat.completions.create(**payload)
        pending: dict[int, dict[str, str]] = {}
        async for event in stream:
            choices = getattr(event, "choices", None) or []
            delta = getattr(choices[0], "delta", None) if choices else None
            for piece in getattr(delta, "tool_calls", None) or []:
                entry = pending.setdefault(
                    int(getattr(piece, "index", 0) or 0), {"id": "", "name": "", "arguments": ""}
                )
                if getattr(piece, "id", None):
                    entry["id"] = str(piece.id)
                function = getattr(piece, "function", None)
                if function is not None:
                    entry["name"] += str(getattr(function, "name", None) or "")
                    entry["arguments"] += str(getattr(function, "arguments", None) or "")
        calls = [
            PlannedToolCall(entry["name"], _parse_arguments(entry["arguments"]), entry["id"])
            for _, entry in sorted(pending.items())
        ]
        return _capped(calls)


class AnthropicMessagesAdapter:
    """Sticky fallback transport: Anthropic Messages with one tool call per turn."""

    label = "Anthropic Messages"

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url, timeout=_REQUEST_TIMEOUT_S, max_retries=0)

    async def plan_turn(
        self, *, user_text: str, history: Sequence[Mapping[str, str]], available_tools: Sequence[Mapping[str, object]]
    ) -> list[PlannedToolCall]:
        tools = [
            {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}
            for tool in available_tools
        ]
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=_RESPONSES_MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=chat_messages(user_text, history),
            tools=tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        ) as stream:
            response = await stream.get_final_message()
        calls = [
            PlannedToolCall(block.name, block.input, block.id)
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]
        return _capped(calls)


class CascadeDialogueProvider:
    """Primary model with a sticky fallback after two consecutive retryable failures."""

    def __init__(
        self,
        primary: OpenAIResponsesAdapter | OpenAIChatCompletionsAdapter,
        fallback: AnthropicMessagesAdapter | None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self._fallback_active = False
        self._consecutive_retryable_failures = 0
        self._events: list[dict[str, str]] = []

    def drain_events(self) -> tuple[dict[str, str], ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    async def plan_turn(
        self, *, user_text: str, history: Sequence[Mapping[str, str]], available_tools: Sequence[Mapping[str, object]]
    ) -> list[PlannedToolCall]:
        call = {"user_text": user_text, "history": history, "available_tools": available_tools}
        if self._fallback_active:
            if self.fallback is None:
                raise ProviderUnavailable("fallback provider unavailable")
            return await self.fallback.plan_turn(**call)
        try:
            result = await self.primary.plan_turn(**call)
        except Exception as exc:
            if not self._is_retryable(exc):
                raise
            self._consecutive_retryable_failures += 1
            if self._consecutive_retryable_failures < 2 or self.fallback is None:
                raise
            self._fallback_active = True
            event = {
                "event": "LLM_PROVIDER_FAILOVER",
                "from_provider": getattr(self.primary, "label", "primary"),
                "to_provider": getattr(self.fallback, "label", "fallback"),
                "failure_count": str(self._consecutive_retryable_failures),
            }
            self._events.append(event)
            logger.warning(
                "LLM_PROVIDER_FAILOVER", extra={key: value for key, value in event.items() if key != "event"}
            )
            return await self.fallback.plan_turn(**call)
        self._consecutive_retryable_failures = 0
        return result

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (APITimeoutError, APIConnectionError, TimeoutError, asyncio.TimeoutError)):
            return True
        return isinstance(exc, APIStatusError) and exc.status_code >= 500


__all__ = [
    "AnthropicMessagesAdapter",
    "CascadeDialogueProvider",
    "OpenAIChatCompletionsAdapter",
    "OpenAIResponsesAdapter",
]
