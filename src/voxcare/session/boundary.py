"""`SessionBoundary`: the one path every caller turn takes, typed or spoken (the turn orchestrator).

Session state lives in Redis (``session:{id}``, TTL 1800 s refreshed on
activity; guide §4.2). In-process objects that cannot be serialised — the
per-session lock, the conversation-provider instance, the turn controller and
the observation trace — are cached here and rebuilt on demand.

A turn: mask the transcript (PII never reaches the model, a log or a store
unmasked), let the :class:`~.dialogue.DialogueManager` decide and run tool calls
through the middleware, then store the masked exchange. A turn epoch fences every
await: a reset, a barge-in or a newer turn turns late results into
``STALE_DISCARDED`` instead of answers.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..contracts import DialogueProvider, SessionView, ToolOutcome, TurnResult
from ..domain import copy
from ..domain.enums import ActionName, ActionStatus, Actor, AuditOutcome
from ..middleware.tool_middleware import ToolContext
from ..observability.audit import audit_json
from ..observability.otel import tracer
from ..security import pii
from ..security.tokens import InvalidToken, SessionClaims
from ..storage.seed import FIXTURE_SESSIONS
from ..trace import TraceLedger
from . import activity
from .dialogue import MAX_HISTORY_MESSAGES
from .tools import normalize_spoken_order_codes, normalize_spoken_terms

logger = logging.getLogger("voxcare.session")

# The turn whose task wrote an audit row: a superseded turn that finishes late must not add its rows to the
# activity entry of the turn that replaced it (log SES-345e37a0, turns 16 and 17).
_TURN_EPOCH: contextvars.ContextVar[int | None] = contextvars.ContextVar("voxcare_turn_epoch", default=None)

MAX_TEXT_CHARS = 2_000
MAX_SESSIONS = 200
CONTEXT_MESSAGES = 6
DEFAULT_CUSTOMER = "CUST-001"


@dataclass
class _Local:
    session_id: str
    dialogue: DialogueProvider | None = None
    turn_controller: Any = None
    trace: TraceLedger = field(default_factory=TraceLedger)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn_epoch: int = 0
    active: bool = True
    live_transcript: str = ""
    turn_events: list[Mapping[str, Any]] = field(default_factory=list)
    events_by_epoch: dict[int | None, list[Mapping[str, Any]]] = field(default_factory=dict)
    activity: list[dict[str, Any]] = field(default_factory=list)
    last_touched_monotonic: float = field(default_factory=time.monotonic)

    def log(self, entry: dict[str, Any]) -> dict[str, Any]:
        self.activity.append(entry)
        del self.activity[: -activity.MAX_ENTRIES]
        return entry


class _TracingPlanner:
    """Wraps the conversation provider in an ``llm`` span with the T2/T3 events (guide §9.5)."""

    def __init__(self, provider: DialogueProvider, session_id: str, epoch: int) -> None:
        self.provider = provider
        self.session_id = session_id
        self.epoch = epoch

    async def plan_turn(self, **kwargs: Any) -> Any:
        with tracer().start_as_current_span(
            "llm", attributes={"session_id": self.session_id, "turn_epoch": self.epoch}
        ) as span:
            span.add_event("T2_prompt_dispatched")
            result = await self.provider.plan_turn(**kwargs)
            span.add_event("T3_first_model_output")
            return result


class SessionBoundary:
    """One session-level boundary shared by typed text and committed voice transcripts."""

    def __init__(
        self,
        *,
        dialogue: DialogueProvider | None = None,
        dialogue_factory: Callable[[], DialogueProvider | None] | None = None,
        turn_controller: Any = None,
        turn_controller_factory: Callable[[], Any] | None = None,
        runtime: Any = None,
    ) -> None:
        self.dialogue = dialogue
        self.dialogue_factory = dialogue_factory or (lambda: dialogue)
        self.turn_controller = turn_controller
        self.turn_controller_factory = turn_controller_factory
        self._runtime = runtime
        self._local: dict[str, _Local] = {}
        self._lock = asyncio.Lock()
        if runtime is not None:
            self._listen(runtime)

    # -- runtime --

    def attach(self, runtime: Any) -> None:
        self._runtime = runtime
        self._local.clear()
        self._listen(runtime)

    def detach(self) -> Any:
        runtime, self._runtime = self._runtime, None
        self._local.clear()
        return runtime

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            from ..runtime import build_memory_runtime

            self.attach(build_memory_runtime())
        return self._runtime

    def _listen(self, runtime: Any) -> None:
        def on_audit(row: Mapping[str, Any]) -> None:
            local = self._local.get(row["session_id"])
            if local is not None:
                event = local.trace.append_audit(audit_json(row))
                local.turn_events.append(event)
                local.events_by_epoch.setdefault(_TURN_EPOCH.get(), []).append(event)

        if on_audit not in runtime.audit.listeners:
            runtime.audit.listeners.append(on_audit)

    def _now(self) -> datetime:
        return self.runtime.clock.now()

    # -- session registry --

    def _ensure_local(self, session_id: str) -> _Local:
        local = self._local.get(session_id)
        if local is None:
            local = _Local(
                session_id=session_id,
                dialogue=self.dialogue_factory(),
                turn_controller=self.turn_controller_factory()
                if self.turn_controller_factory
                else self.turn_controller,
            )
            self._local[session_id] = local
            while len(self._local) > MAX_SESSIONS:
                oldest = min(self._local, key=lambda sid: self._local[sid].last_touched_monotonic)
                del self._local[oldest]
        local.last_touched_monotonic = time.monotonic()
        return local

    async def _record(self, session_id: str, *, touch: bool = False) -> dict[str, Any] | None:
        store = self.runtime.store
        record = await store.get_session(session_id)
        if record is None:
            local = self._local.pop(session_id, None)
            if local is not None:
                local.active = False
            return None
        if touch:
            await store.touch_session(session_id)
        return record

    async def create_session(
        self, *, verified: bool = True, fixture_session_id: str | None = None, customer_id: str | None = None
    ) -> SessionView:
        if fixture_session_id is not None and fixture_session_id not in FIXTURE_SESSIONS:
            raise ValueError("unknown fixture session")
        session_id = fixture_session_id or f"SES-{uuid.uuid4().hex}"
        if fixture_session_id:
            customer = FIXTURE_SESSIONS[fixture_session_id] if verified else None
        else:
            customer = (customer_id or DEFAULT_CUSTOMER) if verified else None
        now = self._now().astimezone(UTC).isoformat()
        record = {
            "session_id": session_id,
            "customer_id": customer,
            "channel": "VoiceWebRTC",
            "turn_count": 0,
            "active_order_id": None,
            "created_at": now,
            "last_active_at": now,
            "turn_epoch": 0,
            "llm_active_provider": None,
            "llm_consecutive_failures": 0,
            "slot_state": None,
            "paused_topic": None,
            "history": [],
            "fixture": fixture_session_id is not None,
            "verified": customer is not None,
            "verification_mode": "verified" if verified else "unverified",
        }
        if fixture_session_id:
            await self._drop_local(session_id)
        await self.runtime.store.put_session(record)
        self._ensure_local(session_id).log(
            activity.session_entry(customer_id=customer, verification_mode=record["verification_mode"])
        )
        return self._view(record, self._local[session_id])

    async def _drop_local(self, session_id: str) -> None:
        local = self._local.pop(session_id, None)
        if local is not None:
            local.active = False

    async def get_session(self, session_id: str) -> SessionView | None:
        record = await self._record(session_id, touch=True)
        if record is None:
            return None
        return self._view(record, self._ensure_local(session_id))

    async def reset_session(self, session_id: str) -> SessionView | None:
        record = await self._record(session_id)
        if record is None:
            return None
        await self.runtime.store.delete_session(session_id)
        await self._drop_local(session_id)
        return await self.create_session(verified=record.get("verification_mode", "verified") == "verified")

    def issue_token(self, session: SessionView | Mapping[str, Any], *, probe: bool = False) -> str:
        session_id = session["session_id"] if isinstance(session, Mapping) else session.session_id
        customer = session.get("customer_id") if isinstance(session, Mapping) else session.customer_id
        return self.runtime.tokens.issue(session_id, customer, now=datetime.now(UTC), probe=probe)

    async def verify_token(self, token: str | None, *, session_id: str | None = None) -> SessionClaims:
        claims = self.runtime.tokens.verify(token, now=datetime.now(UTC), session_id=session_id)
        if claims.probe:
            return claims
        record = await self._record(claims.session_id)
        if record is None:
            raise InvalidToken("session expired")
        return claims

    async def get_turn_controller(self, session_id: str) -> Any | None:
        if await self._record(session_id) is None:
            return None
        return self._ensure_local(session_id).turn_controller

    async def get_conversation_history(self, session_id: str) -> tuple[Mapping[str, str], ...]:
        record = await self._record(session_id)
        if record is None:
            return ()
        return tuple(dict(item) for item in record.get("history", [])[-CONTEXT_MESSAGES:])

    async def record_observation(self, session_id: str, event_type: str, **fields: Any) -> Mapping[str, Any] | None:
        if await self._record(session_id) is None:
            return None
        return self._ensure_local(session_id).trace.record_observation(
            session_id=session_id, event_type=event_type, **fields
        )

    async def record_audit_event(self, session_id: str, event: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Persist a turn-controller audit row (TURN_COMPLETION_DEGRADED / _FALLBACK_RESOLVED) in ``audit_log``."""
        if await self._record(session_id) is None:
            return None
        row = dict(event)
        stamp = row.get("timestamp")
        if isinstance(stamp, str):
            row["timestamp"] = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        elif stamp is None:
            row["timestamp"] = datetime.now(UTC)
        row["request_payload_json"] = pii.mask_payload(row.get("request_payload_json") or {})
        row["response_payload_json"] = pii.mask_payload(row.get("response_payload_json") or {})
        await self.runtime.repository.insert_audit(row)
        local = self._ensure_local(session_id)
        event = local.trace.append_audit(audit_json(row))
        summary = _TURN_CONTROL_SUMMARIES.get(str(row.get("action_name")), str(row.get("action_name")))
        local.log(activity.event_entry(summary, activity.audit_step(event)))
        return event

    async def session_trace(self, session_id: str) -> tuple[Mapping[str, Any], ...] | None:
        if await self._record(session_id) is None:
            return None
        return self._ensure_local(session_id).trace.snapshot()

    async def session_audit(self, session_id: str) -> list[dict[str, Any]] | None:
        if await self._record(session_id) is None:
            return None
        return [audit_json(row) for row in await self.runtime.repository.list_audit(session_id)]

    async def session_activity(self, session_id: str) -> dict[str, Any] | None:
        """The session's activity log (masked), oldest entry first."""
        record = await self._record(session_id)
        if record is None:
            return None
        return {
            "session_id": session_id,
            "customer_id": record.get("customer_id"),
            "entries": [dict(entry) for entry in self._ensure_local(session_id).activity],
        }

    async def session_handoffs(self, session_id: str) -> list[dict[str, Any]] | None:
        if await self._record(session_id) is None:
            return None
        return await self.runtime.store.list_handoffs(session_id)

    async def set_live_transcript(self, session_id: str, text: str) -> bool:
        if await self._record(session_id) is None:
            return False
        self._ensure_local(session_id).live_transcript = pii.mask(str(text)[-MAX_TEXT_CHARS:])
        return True

    async def note_barge_in(self, session_id: str, *, interrupted_epoch: int | None = None) -> int | None:
        """Barge-in (guide §10.3): advance the epoch so in-flight work goes stale, and pause the current topic."""
        record = await self._record(session_id)
        if record is None:
            return None
        local = self._ensure_local(session_id)
        epoch = max(local.turn_epoch, int(record.get("turn_epoch", 0)), interrupted_epoch or 0) + 1
        local.turn_epoch = epoch
        record["turn_epoch"] = epoch
        if record.get("last_topic") and record.get("last_reply"):
            record["paused_topic"] = {"label": record["last_topic"], "reply": record["last_reply"]}
            record["resume_pending"] = False
        await self.runtime.store.put_session(record)
        local.trace.record_observation(
            session_id=session_id, event_type="barge_in", interrupted_epoch=interrupted_epoch, turn_epoch=epoch
        )
        steps = [activity.step("state", f"Turn epoch advanced to {epoch}: late results from before are dropped")]
        if record.get("paused_topic"):
            steps.append(
                activity.step("state", f"Paused topic: {record['paused_topic']['label']} (resume question follows)")
            )
        local.log(activity.event_entry("Caller interrupted the reply (barge-in)", *steps))
        return epoch

    async def record_handoff(
        self,
        session_id: str,
        *,
        reason: str = "caller_requested_support",
        context: str | None = None,
        turn_id: str | None = None,
        turn_epoch: int | None = None,
    ) -> Mapping[str, Any] | None:
        """The support button and the voice ``handoff_request``: ``escalate_to_human`` through the middleware."""
        record = await self._record(session_id, touch=True)
        if record is None:
            return None
        local = self._ensure_local(session_id)
        audit_start = len(local.turn_events)
        live = context if context is not None else local.live_transcript
        transcript = list(record.get("history", []))[-MAX_HISTORY_MESSAGES:]
        if live and live.strip():
            transcript.append({"role": "user", "content": pii.mask(live)})
        result = await self.runtime.middleware.handle(
            ToolContext(
                session_id=session_id,
                customer_id=record.get("customer_id"),
                user_text=pii.mask(reason),
                turn_epoch=turn_epoch or int(record.get("turn_epoch", 0)),
                fixture_session=bool(record.get("fixture")),
                active_order_code=record.get("active_order_id"),
                transcript=transcript,
                symptom_history=record.get("symptom_history", []),
                recommendation=record.get("recommendation"),
                summary=f"Caller pressed the support button ({pii.mask(reason)}). Active order: "
                f"{record.get('active_order_id') or 'none'}.",
            ),
            "escalate_to_human",
            {"trigger_reason": "CUSTOMER_REQUESTED_ESCALATION", "priority": "NORMAL"},
        )
        entry = local.log(activity.support_entry(result, local.turn_events[audit_start:]))
        if not result.ok:
            message = (
                copy.VERIFY_ACCOUNT_PROMPT
                if result.error_code == "UNAUTHENTICATED_SESSION"
                else copy.HANDOFF_SERVICE_ERROR
            )
            return {
                "status": result.error_code,
                "message": message,
                "request_id": None,
                "ticket_number": None,
                "activity": entry,
            }
        envelope = result.envelope
        return {
            "status": envelope["transfer_status"],
            "request_id": envelope["handoff_id"],
            "handoff_id": envelope["handoff_id"],
            "ticket_number": envelope["ticket_number"],
            "business_hours_notice": envelope["business_hours_notice"],
            "message": copy.handoff_recorded(envelope["ticket_number"]),
            "audit_event": local.turn_events[-1] if local.turn_events else None,
            "activity": entry,
        }

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
        runtime = self.runtime
        record = await self._record(session_id, touch=True)
        if record is None:
            return _result(session_id, request_id, 0, "SESSION_EXPIRED", pii.mask(user_text), None)
        local = self._ensure_local(session_id)
        started = time.monotonic()
        async with local.lock:
            record = await self._record(session_id) or record
            local.turn_epoch = max(local.turn_epoch, int(record.get("turn_epoch", 0))) + 1
            epoch = local.turn_epoch
            record["turn_epoch"] = epoch
            record["turn_count"] = int(record.get("turn_count", 0)) + 1
            record["last_active_at"] = self._now().astimezone(UTC).isoformat()
            observations: list[Mapping[str, Any]] = []
            codes: tuple[str, ...] | list[str] = ()
            terms: tuple[str, ...] = ()
            if source == "asr_final":
                text, terms = normalize_spoken_terms(text)
                # Small ASR models spell or mishear codes ("or rd one zero zero two"); fix them before planning.
                text, codes = normalize_spoken_order_codes(text)
                if codes:
                    observations.append(
                        local.trace.record_observation(
                            session_id=session_id,
                            event_type="SPOKEN_ORDER_CODE_NORMALIZED",
                            turn_id=request_id,
                            turn_epoch=epoch,
                            codes=list(codes),
                        )
                    )
            masked = pii.mask(text)
            await runtime.store.put_session(record)
            local.turn_events.clear()
            _TURN_EPOCH.set(epoch)
            for stale in [key for key in local.events_by_epoch if key is None or key < epoch - 4]:
                del local.events_by_epoch[stale]
            before = activity.state_snapshot(record)
            turn_number = int(record["turn_count"])

        def is_current() -> bool:
            return local.active and local.turn_epoch == epoch and self._local.get(session_id) is local

        def timing() -> dict[str, float]:
            return {"request_started_monotonic": started, "provider_complete_monotonic": time.monotonic()}

        planner = _TracingPlanner(local.dialogue, session_id, epoch) if local.dialogue is not None else None
        with tracer().start_as_current_span(
            "turn", attributes={"session_id": session_id, "turn_epoch": epoch, "source": source}
        ) as span:
            span.add_event("T1_turn_committed")
            reply = await runtime.dialogue.respond(
                record, raw_text=text, text=masked, planner=planner, is_current=is_current
            )
            span.set_attribute("turn.outcome", reply.outcome)
            if reply.text:
                span.add_event("T4_text_forwarded_to_tts")
        provider_events = await self._drain_provider_events(local, record, session_id)
        events = [*observations, *provider_events]
        for item in reply.observations:
            events.append(
                local.trace.record_observation(session_id=session_id, turn_id=request_id, turn_epoch=epoch, **item)
            )

        def log_turn(outcome: str, reply_text: str | None) -> dict[str, Any]:
            return local.log(
                activity.turn_entry(
                    turn=turn_number,
                    source=source,
                    caller_text=masked,
                    outcome=outcome,
                    reply_text=reply_text,
                    route=reply.route,
                    results=reply.results,
                    write=reply.write,
                    audit_rows=[*provider_events, *local.events_by_epoch.pop(epoch, [])],
                    before=before,
                    after=activity.state_snapshot(record),
                    normalized_codes=list(codes),
                    normalized_terms=list(terms),
                )
            )

        if reply.outcome == "STALE_DISCARDED" or not is_current():
            entry = log_turn("STALE_DISCARDED", None)
            return _result(session_id, request_id, epoch, "STALE_DISCARDED", masked, None, activity=entry)
        if reply.outcome == "DEPENDENCY_UNAVAILABLE":
            exc = reply.provider_error
            if exc is not None:
                logger.warning(
                    "conversation provider failed: %s: %s",
                    type(exc).__name__,
                    _provider_error_detail(exc),
                    extra={"provider_error_type": type(exc).__name__, "request_id": request_id},
                )
                code = _provider_error_code(exc)
                events.append(
                    local.trace.record_observation(
                        session_id=session_id,
                        event_type="PROVIDER_ERROR",
                        turn_id=request_id,
                        turn_epoch=epoch,
                        error_type=type(exc).__name__,
                        **({"error_code": code} if code else {}),
                    )
                )
            else:
                events.append(
                    local.trace.record_observation(
                        session_id=session_id,
                        event_type="PROVIDER_UNAVAILABLE",
                        turn_id=request_id,
                        turn_epoch=epoch,
                        reason="conversation_not_configured",
                    )
                )
            await runtime.store.put_session(record)
            entry = log_turn("DEPENDENCY_UNAVAILABLE", None)
            return _result(
                session_id,
                request_id,
                epoch,
                "DEPENDENCY_UNAVAILABLE",
                masked,
                None,
                events=(*events, *local.turn_events),
                timing=timing(),
                activity=entry,
            )

        history = list(record.get("history", []))
        history.append({"role": "user", "content": masked})
        if reply.text:
            history.append({"role": "assistant", "content": reply.text})
        record["history"] = history[-MAX_HISTORY_MESSAGES:]
        await runtime.store.put_session(record)
        binding_changed = reply.binding_changed
        entry = log_turn(reply.outcome, reply.text or None)

        policy = next((result for result in reply.results if result.name == "retrieve_policy_clauses"), None)
        outcomes = tuple(
            ToolOutcome(
                tool_name=result.name,
                status="SUCCESS" if result.ok else str(result.error_code),
                record=result.envelope,
                error_code=result.error_code,
                evidence=tuple(reply.evidence) if result is policy else (),
            )
            for result in reply.results
        )
        return _result(
            session_id,
            request_id,
            epoch,
            reply.outcome,
            masked,
            reply.text or None,
            spoken_text=reply.voice or None,
            tool_outcomes=outcomes,
            tool_calls=tuple({"name": plan.name, "arguments": dict(plan.arguments)} for plan in reply.plans),
            events=(*events, *local.turn_events),
            evidence=tuple(reply.evidence),
            policy_outcome=(policy.envelope.get("outcome") if policy.ok else policy.error_code) if policy else None,
            corpus_id=policy.envelope.get("corpus_id") if policy else None,
            is_fictional=policy.envelope.get("is_fictional") if policy else None,
            clause_version=policy.envelope.get("clause_version") if policy else None,
            customer_id=record.get("customer_id"),
            binding_changed=binding_changed,
            timing=timing(),
            activity=entry,
        )

    async def _drain_provider_events(
        self, local: _Local, record: dict[str, Any], session_id: str
    ) -> list[Mapping[str, Any]]:
        drain = getattr(local.dialogue, "drain_events", None)
        if not callable(drain):
            return []
        events = []
        for item in drain():
            if item.get("event") != "LLM_PROVIDER_FAILOVER":
                continue
            row = await self.runtime.audit.write(
                session_id=session_id,
                action_name=ActionName.LLM_PROVIDER_FAILOVER,
                actor=Actor.SYSTEM,
                action_status=ActionStatus.SUCCESS,
                outcome_status=AuditOutcome.SUCCESS,
                request={
                    "primary_model": item.get("primary_model") or item.get("from_provider"),
                    "consecutive_failures": int(item.get("failure_count", 2)),
                },
                response={
                    "active_model": item.get("active_model") or item.get("to_provider"),
                    "session_scope": "remainder_of_session",
                },
            )
            record["llm_active_provider"] = item.get("active_model") or item.get("to_provider")
            await self.runtime.store.put_session(record)
            events.append(audit_json(row))
        local.turn_events[:] = [
            event for event in local.turn_events if event.get("action_name") != "LLM_PROVIDER_FAILOVER"
        ]
        return events

    def _view(self, record: Mapping[str, Any], local: _Local) -> SessionView:
        configured = local.dialogue is not None
        return SessionView(
            session_id=record["session_id"],
            customer_id=record.get("customer_id"),
            locale="en-US",
            turn_epoch=max(local.turn_epoch, int(record.get("turn_epoch", 0))),
            state="ready" if configured else "conversation_provider_unavailable",
            provider_availability={"conversation": configured},
            created_at=record["created_at"],
            verified=bool(record.get("customer_id")),
            fixture=bool(record.get("fixture")),
        )


_TURN_CONTROL_SUMMARIES = {
    "TURN_COMPLETION_DEGRADED": "End of turn unclear: the caller was offered to finish, retry or ask for support",
    "TURN_COMPLETION_FALLBACK_RESOLVED": "The caller resolved the unclear turn",
}


def _provider_error_detail(exc: BaseException) -> str:
    """The provider's own error message for the log: masked, bounded, never the request or a credential."""
    return pii.mask(" ".join(str(exc).split()))[:300] or "no detail"


def _provider_error_code(exc: BaseException) -> str | None:
    """A short machine code for the trace: the provider's error code/type, else the HTTP status."""
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        code = body.get("code") or body.get("type")
        if isinstance(code, str) and code:
            return code[:64]
    status = getattr(exc, "status_code", None)
    return f"http_{status}" if isinstance(status, int) else None


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


__all__ = ["CONTEXT_MESSAGES", "MAX_SESSIONS", "MAX_TEXT_CHARS", "SessionBoundary"]
