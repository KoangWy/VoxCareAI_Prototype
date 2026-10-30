"""Shared data contracts between the session boundary, tools, and conversation providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class PlannedToolCall:
    name: str
    arguments: Mapping[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class ToolOutcome:
    tool_name: str
    status: str
    record: Mapping[str, Any] | None = None
    records: Sequence[Mapping[str, Any]] = ()
    error_code: str | None = None
    evidence: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True)
class SessionView:
    session_id: str
    customer_id: str
    locale: str
    turn_epoch: int
    state: str
    provider_availability: Mapping[str, bool]
    created_at: str


@dataclass(frozen=True)
class TurnResult:
    session_id: str
    turn_id: str
    turn_epoch: int
    outcome: str
    user_text: str
    assistant_text: str | None
    tool_outcomes: Sequence[ToolOutcome] = ()
    evidence: Sequence[Mapping[str, Any]] = ()
    events: Sequence[Mapping[str, Any]] = ()
    timing: Mapping[str, Any] = field(default_factory=dict)
    policy_outcome: str | None = None
    corpus_id: str | None = None
    is_fictional: bool | None = None
    clause_version: str | None = None
    missing_context: Sequence[str] = ()


MAX_TOOL_CALLS_PER_TURN = 3


class DialogueProvider(Protocol):
    async def plan_turn(
        self,
        *,
        user_text: str,
        history: Sequence[Mapping[str, str]],
        available_tools: Sequence[Mapping[str, Any]],
    ) -> Sequence[PlannedToolCall] | PlannedToolCall | None:
        """Return every tool call the model emitted, in order (empty/None = no tool).

        Adapters return a (possibly empty) list; a bare ``PlannedToolCall``
        is also accepted. The session boundary executes the read-only calls
        sequentially up to ``MAX_TOOL_CALLS_PER_TURN``.
        """
        ...
