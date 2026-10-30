"""`SessionBoundary`: the one path every caller turn takes, typed or spoken.

Owns the per-session state (history, trace, turn epoch, turn controller),
plans a turn with the conversation provider, executes the allowlisted
read-only tool calls under the session's customer scope, and renders one
grounded answer. A turn epoch fences every await: a reset or a newer turn
turns late results into ``STALE_DISCARDED`` instead of answers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..contracts import MAX_TOOL_CALLS_PER_TURN, DialogueProvider, PlannedToolCall, SessionView, ToolOutcome, TurnResult
from ..domain.policy import PolicyRetriever
from ..domain.seed import SeedOMS
from ..privacy import mask_pii
from ..trace import TraceLedger
from .render import CLARIFY_LIMIT_TEXT, CLARIFY_TEXT, render_results
from .tools import (
    CATALOG_TOOLS,
    MAX_HANDOFF_REASON_CHARS,
    ORDER_CODE,
    TOOL_NAMES,
    TOOLS,
    InvalidToolArguments,
    audit_outcome,
    normalize_order_code,
    safe_arguments,
    validate_arguments,
)

logger = logging.getLogger("voxcare.session")

_DEFAULT_POLICY_RETRIEVER = object()
MAX_TEXT_CHARS = 2_000
MAX_CLARIFICATIONS = 3
MAX_SESSIONS = 200
# Completed messages kept per session (and sent as model context); older ones drop first.
MAX_HISTORY_MESSAGES = 24
# Recent completed messages shared with the turn judge and handoff context.
CONTEXT_MESSAGES = 6


@dataclass
class _Session:
    session_id: str
    customer_id: str = "CUST-001"
    locale: str = "en-US"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    active: bool = True
    turn_epoch: int = 0
    clarification_count: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    dialogue: DialogueProvider | None = None
    trace: TraceLedger = field(default_factory=TraceLedger)
    turn_controller: Any = None
    handoffs: list[Mapping[str, Any]] = field(default_factory=list)
    live_transcript: str = ""
    last_touched_monotonic: float = field(default_factory=time.monotonic)

    def remember(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        del self.history[:-MAX_HISTORY_MESSAGES]

    def forget_unanswered(self, entry: Mapping[str, str]) -> None:
        """Drop a user message whose turn produced no answer, so the model never answers it late."""
        for index in range(len(self.history) - 1, -1, -1):
            if self.history[index] is entry:
                del self.history[index]
                return


class SessionBoundary:
    """One session-level boundary shared by typed text and committed voice transcripts."""

    def __init__(
        self,
        *,
        dialogue: DialogueProvider | None,
        oms: SeedOMS | None = None,
        dialogue_factory: Callable[[], DialogueProvider | None] | None = None,
        turn_controller: Any = None,
        policy_retriever: Any = _DEFAULT_POLICY_RETRIEVER,
        turn_controller_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.dialogue = dialogue
        self.dialogue_factory = dialogue_factory or (lambda: dialogue)
        self.oms = oms or SeedOMS()
        self.turn_controller = turn_controller
        self.turn_controller_factory = turn_controller_factory
        self.policy_retriever = PolicyRetriever() if policy_retriever is _DEFAULT_POLICY_RETRIEVER else policy_retriever
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()

    # -- session registry --

    async def _active(self, session_id: str, *, touch: bool = False) -> _Session | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.active:
                return None
            if touch:
                session.last_touched_monotonic = time.monotonic()
            return session

    def _evict_if_over_budget(self) -> None:
        """Bounded LRU (caller holds the lock): evicted sessions are discarded outright."""
        while len(self._sessions) > MAX_SESSIONS:
            oldest_id = min(self._sessions, key=lambda sid: self._sessions[sid].last_touched_monotonic)
            del self._sessions[oldest_id]

    async def create_session(self) -> SessionView:
        session = _Session(
            session_id=f"SES-{uuid.uuid4().hex}",
            dialogue=self.dialogue_factory(),
            turn_controller=self.turn_controller_factory() if self.turn_controller_factory else self.turn_controller,
        )
        async with self._lock:
            self._sessions[session.session_id] = session
            self._evict_if_over_budget()
        return self._view(session)

    async def get_session(self, session_id: str) -> SessionView | None:
        session = await self._active(session_id, touch=True)
        return self._view(session) if session is not None else None

    async def reset_session(self, session_id: str) -> SessionView | None:
        async with self._lock:
            old = self._sessions.pop(session_id, None)
            if old is None or not old.active:
                return None
            old.active = False
        return await self.create_session()

    async def get_turn_controller(self, session_id: str) -> Any | None:
        session = await self._active(session_id)
        return session.turn_controller if session is not None else None

    async def get_conversation_history(self, session_id: str) -> tuple[Mapping[str, str], ...]:
        """Recent completed dialogue context for a turn decision."""
        session = await self._active(session_id)
        if session is None:
            return ()
        return tuple(dict(item) for item in session.history[-CONTEXT_MESSAGES:])

    async def record_observation(self, session_id: str, event_type: str, **fields: Any) -> Mapping[str, Any] | None:
        session = await self._active(session_id)
        if session is None:
            return None
        return session.trace.record_observation(session_id=session_id, event_type=event_type, **fields)

    async def record_audit_event(self, session_id: str, event: Mapping[str, Any]) -> Mapping[str, Any] | None:
        session = await self._active(session_id)
        if session is None:
            return None
        return session.trace.append_audit(event)

    async def session_trace(self, session_id: str) -> tuple[Mapping[str, Any], ...] | None:
        session = await self._active(session_id)
        return session.trace.snapshot() if session is not None else None

    async def set_live_transcript(self, session_id: str, text: str) -> bool:
        """Keep only the latest server-observed ASR text available for a support request."""
        session = await self._active(session_id)
        if session is None:
            return False
        session.live_transcript = str(text)[-MAX_TEXT_CHARS:]
        return True

    async def record_handoff(
        self,
        session_id: str,
        *,
        reason: str,
        context: str | None = None,
        turn_id: str | None = None,
        turn_epoch: int | None = None,
    ) -> Mapping[str, Any] | None:
        session = await self._active(session_id, touch=True)
        if session is None:
            return None
        if context is None:
            completed = "\n".join(
                f"{item.get('role', 'unknown')}: {item.get('content', '')}"
                for item in session.history[-CONTEXT_MESSAGES:]
            )
            context = "\n".join(part for part in (completed, session.live_transcript) if part.strip())
        return self._record_handoff(session, reason, context, turn_id, turn_epoch)

    @staticmethod
    def _record_handoff(
        session: _Session, reason: str, context: str, turn_id: str | None, turn_epoch: int | None
    ) -> Mapping[str, Any]:
        handoff = {
            "request_id": str(uuid.uuid4()),
            "reason": mask_pii(reason).strip()[:MAX_HANDOFF_REASON_CHARS] or "caller_requested_support",
            "context": mask_pii(context).strip()[-500:],
            "requested_at": datetime.now(UTC).isoformat(),
            "status": "REQUEST_RECORDED",
        }
        session.handoffs.append(handoff)
        audit = session.trace.record_action(
            session_id=session.session_id,
            turn_id=turn_id,
            turn_epoch=turn_epoch,
            action_name="HANDOFF_REQUEST_RECORDED",
            request_payload={"reason": handoff["reason"], "context": handoff["context"]},
            response_payload={"request_id": handoff["request_id"], "status": handoff["status"]},
        )
        return {**handoff, "audit_event": audit}

    # -- turns --

    async def handle_turn(
        self, session_id: str, user_text: str, *, source: str = "manual", correlation_id: str | None = None
    ) -> TurnResult:
        request_id = correlation_id or str(uuid.uuid4())
        if not isinstance(user_text, str):
            return _result(session_id, request_id, 0, "INVALID_INPUT", "", None)
        if source not in {"manual", "asr_final"}:
            return _result(session_id, request_id, 0, "INVALID_INPUT", user_text[:MAX_TEXT_CHARS], None)
        text = user_text.strip()
        if not text or len(user_text) > MAX_TEXT_CHARS:
            return _result(session_id, request_id, 0, "INVALID_INPUT", "", None)
        session = await self._active(session_id, touch=True)
        if session is None:
            return _result(session_id, request_id, 0, "SESSION_EXPIRED", user_text, None)

        started = time.monotonic()
        async with session.lock:
            if not session.active:
                return _result(session_id, request_id, session.turn_epoch, "SESSION_EXPIRED", user_text, None)
            session.turn_epoch += 1
            epoch = session.turn_epoch
            history = tuple(dict(item) for item in session.history)
            session.remember("user", text)
            user_entry = session.history[-1]

        def timing() -> dict[str, float]:
            return {"request_started_monotonic": started, "provider_complete_monotonic": time.monotonic()}

        if session.dialogue is None:
            session.forget_unanswered(user_entry)
            event = _event("provider_unavailable", session, request_id, epoch, reason="conversation_not_configured")
            return _result(session_id, request_id, epoch, "DEPENDENCY_UNAVAILABLE", text, None, events=(event,))

        try:
            raw_plan = await session.dialogue.plan_turn(user_text=text, history=history, available_tools=TOOLS)
        except Exception as exc:
            session.forget_unanswered(user_entry)
            logger.warning(
                "conversation provider failed",
                extra={"provider_error_type": type(exc).__name__, "request_id": request_id},
            )
            provider_events = _drain_provider_events(session, request_id, epoch)
            error_event = _event("provider_error", session, request_id, epoch, error_type=type(exc).__name__)
            return _result(
                session_id,
                request_id,
                epoch,
                "DEPENDENCY_UNAVAILABLE",
                text,
                None,
                events=(*provider_events, error_event),
                timing=timing(),
            )

        if not _is_current(session, epoch):
            session.forget_unanswered(user_entry)
            return _result(session_id, request_id, epoch, "STALE_DISCARDED", text, None)
        provider_events = _drain_provider_events(session, request_id, epoch)
        plans = _normalise_plans(raw_plan)
        if not plans:
            return self._clarify(session, request_id, epoch, text, timing(), provider_events)

        tool_outcomes: list[ToolOutcome] = []
        audits: list[Mapping[str, Any]] = []
        for plan in plans:
            tool_outcome = await self._invoke_tool(session, plan.name, plan.arguments, request_id, epoch)
            if not _is_current(session, epoch):
                session.forget_unanswered(user_entry)
                return _result(session_id, request_id, epoch, "STALE_DISCARDED", text, None)
            if tool_outcome.status == "MISSING_INPUT":
                return self._clarify(session, request_id, epoch, text, timing(), (*provider_events, *audits))
            audits.append(self._audit_tool(session, plan, tool_outcome, request_id, epoch))
            tool_outcomes.append(tool_outcome)

        assistant_text, outcome = render_results(tool_outcomes)
        session.remember("assistant", assistant_text)
        session.clarification_count = 0
        policy = next((tool for tool in tool_outcomes if tool.tool_name == "retrieve_policy_clauses"), None)
        policy_record = (policy.record or {}) if policy is not None else {}
        return _result(
            session_id,
            request_id,
            epoch,
            outcome,
            text,
            assistant_text,
            tool_outcomes=tuple(tool_outcomes),
            events=(*provider_events, *audits),
            evidence=tuple(item for tool in tool_outcomes for item in tool.evidence),
            policy_outcome=policy.status if policy is not None else None,
            corpus_id=policy_record.get("corpus_id"),
            is_fictional=policy_record.get("is_fictional"),
            clause_version=policy_record.get("clause_version"),
            missing_context=tuple(policy_record.get("missing_context", ())),
            timing=timing(),
        )

    def _audit_tool(
        self, session: _Session, plan: PlannedToolCall, outcome: ToolOutcome, request_id: str, epoch: int
    ) -> Mapping[str, Any]:
        if outcome.status == "HANDOFF_REQUEST_RECORDED" and outcome.record:
            return outcome.record["audit_event"]
        denied = outcome.status == "OUT_OF_SCOPE"
        return session.trace.record_action(
            session_id=session.session_id,
            turn_id=request_id,
            turn_epoch=epoch,
            action_name="AUTH_SCOPE_VIOLATION" if denied else "TOOL_INVOCATION",
            action_status="Denied" if denied else "Success",
            outcome_status=audit_outcome(outcome.status),
            request_payload={"tool_name": plan.name, "arguments": safe_arguments(plan.arguments)},
            response_payload={"tool_status": outcome.status, "error_code": outcome.error_code},
        )

    async def _invoke_tool(
        self, session: _Session, name: str, arguments: Mapping[str, Any], request_id: str, epoch: int
    ) -> ToolOutcome:
        if name not in TOOL_NAMES:
            return ToolOutcome("rejected", "SERVICE_ERROR", error_code="TOOL_NOT_ALLOWLISTED")
        try:
            args = validate_arguments(name, arguments)
        except InvalidToolArguments:
            return ToolOutcome(name, "SERVICE_ERROR", error_code="INVALID_TOOL_ARGUMENTS")
        if name == "retrieve_policy_clauses":
            return await self._retrieve_policy(args)
        if name == "escalate_to_human":
            context = session.history[-1]["content"] if session.history else ""
            handoff = self._record_handoff(session, args["reason"], context, request_id, epoch)
            return ToolOutcome(name, "HANDOFF_REQUEST_RECORDED", record=handoff)
        if name in CATALOG_TOOLS:
            return self._read_catalog(session, name, args)
        return self._read_order(session, name, args["order_code"])

    async def _retrieve_policy(self, args: Mapping[str, Any]) -> ToolOutcome:
        name = "retrieve_policy_clauses"
        if self.policy_retriever is None:
            return ToolOutcome(name, "DEPENDENCY_UNAVAILABLE", error_code="POLICY_UNAVAILABLE")
        filters = {key: value for key, value in args.items() if key != "query"}
        try:
            result = await self.policy_retriever.retrieve_policy_clauses(args["query"], filters)
        except Exception as exc:
            logger.warning("policy tool failed", extra={"error_type": type(exc).__name__})
            return ToolOutcome(name, "SERVICE_ERROR", error_code="POLICY_UNAVAILABLE")
        metadata = {
            "corpus_id": result.corpus_id,
            "is_fictional": result.is_fictional,
            "clause_version": result.clause_version,
        }
        evidence = tuple({**asdict(item), **metadata} for item in result.evidence)
        record = {**metadata, "missing_context": list(result.missing_context), "error_code": result.error_code}
        return ToolOutcome(name, result.outcome, record=record, evidence=evidence, error_code=result.error_code)

    def _read_order(self, session: _Session, name: str, raw_code: str) -> ToolOutcome:
        order_code = normalize_order_code(raw_code)
        if not order_code:
            return ToolOutcome(name, "MISSING_INPUT")
        if not ORDER_CODE.fullmatch(order_code):
            return ToolOutcome(name, "NOT_FOUND")
        read = self.oms.lookup_order_details if name == "lookup_order_details" else self.oms.track_shipment_status
        for attempt in range(2):
            try:
                return read(session_customer_id=session.customer_id, order_code=order_code)
            except TimeoutError:
                if attempt == 0:
                    logger.warning("OMS read timed out; retrying once", extra={"tool_name": name})
                    continue
                logger.warning("OMS read timed out after one retry", extra={"tool_name": name})
                return ToolOutcome(name, "SERVICE_TIMEOUT", error_code="OMS_TIMEOUT")
            except Exception as exc:
                logger.error("OMS read failed", extra={"tool_name": name, "error_type": type(exc).__name__})
                break
        return ToolOutcome(name, "SERVICE_ERROR", error_code="OMS_UNAVAILABLE")

    def _read_catalog(self, session: _Session, name: str, args: Mapping[str, Any]) -> ToolOutcome:
        readers = {
            "search_catalog": (self.oms.search_catalog, "CATALOG_UNAVAILABLE"),
            "compare_products": (self.oms.compare_products, "CATALOG_UNAVAILABLE"),
            "apply_voucher": (self.oms.apply_voucher, "PROMOTION_UNAVAILABLE"),
        }
        read, error_code = readers[name]
        try:
            return read(session_customer_id=session.customer_id, **args)
        except Exception as exc:
            logger.error("catalog read failed", extra={"tool_name": name, "error_type": type(exc).__name__})
            return ToolOutcome(name, "SERVICE_ERROR", error_code=error_code)

    @staticmethod
    def _clarify(
        session: _Session,
        request_id: str,
        epoch: int,
        user_text: str,
        timing: Mapping[str, Any],
        prior_events: Sequence[Mapping[str, Any]] = (),
    ) -> TurnResult:
        session.clarification_count += 1
        if session.clarification_count > MAX_CLARIFICATIONS:
            prompt, outcome = CLARIFY_LIMIT_TEXT, "CLARIFICATION_LIMIT_REACHED"
        else:
            prompt, outcome = CLARIFY_TEXT, "NEEDS_ORDER_ID"
        session.remember("assistant", prompt)
        event = _event("clarification_requested", session, request_id, epoch, count=session.clarification_count)
        return _result(
            session.session_id,
            request_id,
            epoch,
            outcome,
            user_text,
            prompt,
            events=(*prior_events, event),
            timing=timing,
        )

    @staticmethod
    def _view(session: _Session) -> SessionView:
        configured = session.dialogue is not None
        return SessionView(
            session_id=session.session_id,
            customer_id=session.customer_id,
            locale=session.locale,
            turn_epoch=session.turn_epoch,
            state="ready" if configured else "conversation_provider_unavailable",
            provider_availability={"conversation": configured},
            created_at=session.created_at,
        )


def _is_current(session: _Session, epoch: int) -> bool:
    return session.active and session.turn_epoch == epoch


def _normalise_plans(raw: PlannedToolCall | Sequence[PlannedToolCall] | None) -> list[PlannedToolCall]:
    """Every ``plan_turn`` shape as an in-order call list capped at ``MAX_TOOL_CALLS_PER_TURN``."""
    if raw is None:
        return []
    if isinstance(raw, PlannedToolCall):
        return [raw]
    return list(raw)[:MAX_TOOL_CALLS_PER_TURN]


def _drain_provider_events(session: _Session, request_id: str, epoch: int) -> tuple[Mapping[str, Any], ...]:
    drain = getattr(session.dialogue, "drain_events", None)
    if not callable(drain):
        return ()
    events = []
    for item in drain():
        if item.get("event") != "LLM_PROVIDER_FAILOVER":
            continue
        events.append(
            session.trace.record_action(
                session_id=session.session_id,
                turn_id=request_id,
                turn_epoch=epoch,
                action_name="LLM_PROVIDER_FAILOVER",
                request_payload={
                    "from_provider": item.get("from_provider"),
                    "failure_count": item.get("failure_count"),
                },
                response_payload={"to_provider": item.get("to_provider")},
            )
        )
    return tuple(events)


def _event(event_type: str, session: _Session, request_id: str, epoch: int, **fields: Any) -> Mapping[str, Any]:
    event = session.trace.record_observation(
        session_id=session.session_id, event_type=event_type, turn_id=request_id, turn_epoch=epoch, **fields
    )
    logger.info(json.dumps(event, sort_keys=True))
    return event


def _result(
    session_id: str,
    turn_id: str,
    epoch: int,
    outcome: str,
    user_text: str,
    assistant_text: str | None,
    **fields: Any,
) -> TurnResult:
    if fields.get("timing") is None:
        fields.pop("timing", None)
    return TurnResult(session_id, turn_id, epoch, outcome, user_text, assistant_text, **fields)


__all__ = ["MAX_CLARIFICATIONS", "MAX_HISTORY_MESSAGES", "MAX_SESSIONS", "MAX_TEXT_CHARS", "SessionBoundary"]
