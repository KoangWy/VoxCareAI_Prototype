"""HandoffService: ``escalate_to_human`` records a request; it never transfers a call (guide §12, design D§7.2).

Effects of one call:

1. ``agent_activity`` row with ``outcome_code = QUEUED`` and no agent assigned;
2. a ``HandoffContextPacket`` in Redis: AI summary, masked verbatim transcript,
   symptom history and recommendation;
3. audit ``HANDOFF_REQUEST_RECORDED`` (actor Bot, Success/SUCCESS) whose response
   carries ``{transfer_status, agent_activity_state: "queued", agent_id: null}``.

PostgreSQL and Redis are two stores, so the three effects are applied in that
order rather than in one transaction (implementation choice); a failure before
the audit returns SERVICE_ERROR and the caller hears the failure line.

Ids: the first request of a fixture session uses the fixture values
``HND-20261014-0942-8821`` / ``TICK-CS-8821``; every other request gets
``HND-<yyyymmdd>-<hhmm>-<nnnn>`` / ``TICK-CS-<nnnn>`` (implementation choice).
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

from ...observability.audit import AuditWriter
from ...security import pii
from ...storage.db import Repository
from ...storage.redis_store import RedisStore
from ..clock import DEMO_TIMEZONE, Clock
from ..enums import ActionName, ActionStatus, Actor, AuditOutcome, ErrorCode
from .envelope import error, success

BUSINESS_HOURS_NOTICE = "Customer support operating hours: 08:00 - 21:00 Monday through Sunday."
FIXTURE_HANDOFF_ID = "HND-20261014-0942-8821"
FIXTURE_TICKET = "TICK-CS-8821"


@dataclass
class HandoffContext:
    session_id: str
    customer_id: str
    active_order_code: str | None = None
    transcript: Sequence[Mapping[str, str]] = ()
    symptom_history: Sequence[Mapping[str, Any]] = ()
    recommendation: str | None = None
    summary: str | None = None
    fixture_session: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class HandoffService:
    def __init__(self, repository: Repository, store: RedisStore, audit: AuditWriter, clock: Clock) -> None:
        self.repository = repository
        self.store = store
        self.audit = audit
        self.clock = clock

    async def _ids(self, context: HandoffContext) -> tuple[str, str]:
        if context.fixture_session and await self.store.get_handoff(FIXTURE_HANDOFF_ID) is None:
            existing = {row["activity_id"] for row in await self.repository.list_agent_activity()}
            if FIXTURE_HANDOFF_ID not in existing:
                return FIXTURE_HANDOFF_ID, FIXTURE_TICKET
        local = self.clock.now().astimezone(DEMO_TIMEZONE)
        existing = {row["activity_id"] for row in await self.repository.list_agent_activity()}
        for _ in range(50):
            number = f"{secrets.randbelow(9000) + 1000}"
            handoff_id = f"HND-{local:%Y%m%d}-{local:%H%M}-{number}"
            if handoff_id not in existing and await self.store.get_handoff(handoff_id) is None:
                return handoff_id, f"TICK-CS-{number}"
        raise RuntimeError("could not allocate a handoff id")

    async def escalate(self, trigger_reason: str, priority: str, context: HandoffContext) -> dict[str, Any]:
        try:
            return await self._escalate(trigger_reason, priority, context)
        except Exception:
            return error(ErrorCode.SERVICE_ERROR, "The support request could not be recorded.")

    async def _escalate(self, trigger_reason: str, priority: str, context: HandoffContext) -> dict[str, Any]:
        handoff_id, ticket = await self._ids(context)
        order_hex = None
        if context.active_order_code:
            order = await self.repository.get_order_by_code(context.active_order_code)
            if order is not None and order["customer_id"] == context.customer_id:
                order_hex = order["order_id"]
        now = self.clock.now().astimezone(UTC)
        await self.repository.insert_agent_activity(
            {
                "activity_id": handoff_id,
                "session_id": context.session_id,
                "customer_id": context.customer_id,
                "order_id": order_hex,
                "activity_code": "HANDOFF_REQUEST_RECORDED",
                "outcome_code": "QUEUED",
                "activity_at": now,
            }
        )
        transcript = [
            {"role": str(item.get("role", "")), "text": pii.mask(item.get("content", ""))}
            for item in context.transcript
        ]
        packet = {
            "handoff_id": handoff_id,
            "session_id": context.session_id,
            "customer_id": context.customer_id,
            "order_id": context.active_order_code,  # G-6: nullable for sessions without an active order
            "trigger_reason": trigger_reason,
            "priority": priority,
            "context_payload_json": {
                "ai_summary": pii.mask(context.summary or ""),
                "verbatim_transcript": transcript,
                "symptom_history": pii.mask_payload(list(context.symptom_history)),
                "recommendation": pii.mask(context.recommendation or ""),
            },
            "ticket_number": ticket,
            "created_at": now.isoformat(),
        }
        await self.store.put_handoff(packet)
        await self.audit.write(
            session_id=context.session_id,
            action_name=ActionName.HANDOFF_REQUEST_RECORDED,
            actor=Actor.BOT,
            action_status=ActionStatus.SUCCESS,
            outcome_status=AuditOutcome.SUCCESS,
            order_id=order_hex,
            request={
                "tool_name": "escalate_to_human",
                "trigger_reason": trigger_reason,
                "priority": priority,
                **({"order_code": context.active_order_code} if context.active_order_code else {}),
            },
            response={
                "transfer_status": "HANDOFF_REQUEST_RECORDED",
                "agent_activity_state": "queued",
                "agent_id": None,
                "handoff_id": handoff_id,
                "ticket_number": ticket,
            },
        )
        return success(
            transfer_status="HANDOFF_REQUEST_RECORDED",
            handoff_id=handoff_id,
            ticket_number=ticket,
            summary_recorded=True,
            business_hours_notice=BUSINESS_HOURS_NOTICE,
        )


__all__ = ["BUSINESS_HOURS_NOTICE", "FIXTURE_HANDOFF_ID", "FIXTURE_TICKET", "HandoffContext", "HandoffService"]
