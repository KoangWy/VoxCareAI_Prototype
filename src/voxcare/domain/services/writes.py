"""Guarded-write library (future-only; guide §11, design D§6). Never exposed to the LLM in Phase 1.

Protocol: **Preview → Confirm → Revalidate → Execute → Audit**.

- *Preview* simulates the rules with zero mutations. A failed preview creates no
  ``PendingAction``; otherwise one is staged ``pending`` with ``expected_version``
  = the current ``orders.version``, a UUIDv4 ``idempotency_key`` and a ``preview_hash``.
- *Confirm* takes 1-2 steps (minimum 1); ``pending → submitted`` seals the payload
  and audits ``EXECUTION_PENDING`` (outcome PENDING).
- *Revalidate/Execute*: the worker takes the OCC gate (``submitted → executing``) and
  VoxCom executes; success is spoken **only** from an explicit execution result.
- A timeout, lost response or transport error never causes ``failed``: the action
  stays ``executing``, the audit outcome is ``UNKNOWN`` and the status is queried by
  the same key. A retry always reuses the key; it is never re-minted.

In the Phase 1 dialogue only preview and confirm run (``record_only``): the
request is recorded for VoxCom and nothing executes. The full lifecycle is driven
by tests against :class:`voxcare.storage.fake_oms.FakeVoxComOMS`.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

from ...observability.audit import AuditWriter
from ...storage.db import Repository
from ...storage.redis_store import DuplicateIdempotencyKey, RedisStore
from .. import copy
from ..clock import DEMO_TIMEZONE, Clock
from ..enums import ActionName, ActionStatus, ActionType, Actor, AuditOutcome, PendingActionStatus
from ..rules import (
    can_advance_action,
    cancellable,
    confirmation_steps_valid,
    is_dispatched,
    preview_hash,
    reinstate_cancel_allowed,
    reverse_address_change_allowed,
    scope_allows,
)

logger = logging.getLogger("voxcare.writes")

TOOL_FOR_ACTION = {
    ActionType.CANCEL: "cancel_order",
    ActionType.RESCHEDULE: "reschedule_delivery",
    ActionType.ADDRESS_CHANGE: "change_delivery_address",
    ActionType.REVERSAL: "request_reversal",
}
COMPLETION_AUDIT = {
    ActionType.CANCEL: ActionName.ORDER_CANCELLED,
    ActionType.RESCHEDULE: ActionName.DELIVERY_RESCHEDULED,
}
DEFAULT_SLOT = ("09:00", "12:00")  # † window used when the caller names only a day


class OmsClient(Protocol):
    async def execute(self, key: str, action: Mapping[str, Any]) -> dict[str, Any]: ...
    async def abort(self, key: str) -> str: ...
    async def status(self, key: str) -> dict[str, Any] | None: ...
    def reversal_facts(self, order_code: str) -> dict[str, Any]: ...


def _local_datetime(value: Any) -> datetime | None:
    """A stored delivery time (aware datetime or ISO string) in the demo's local zone, or ``None``."""
    if value is None or value == "":
        return None
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return moment.astimezone(DEMO_TIMEZONE)


@dataclass
class WriteOutcome:
    """What the dialogue may say, plus the state it is allowed to claim."""

    kind: str  # PREVIEW | REFUSED | SUBMITTED | COMPLETED | FAILED | CANCELLED | IN_PROGRESS | UNKNOWN | EXPIRED | ...
    spoken: str
    action: dict[str, Any] | None = None
    audit_names: list[str] = field(default_factory=list)


class WriteService:
    def __init__(
        self,
        repository: Repository,
        store: RedisStore,
        audit: AuditWriter,
        clock: Clock,
        oms: OmsClient | None = None,
    ) -> None:
        self.repository = repository
        self.store = store
        self.audit = audit
        self.clock = clock
        self.oms = oms

    # ------------------------------------------------------------------ preview

    async def preview(
        self,
        *,
        session_id: str,
        customer_id: str | None,
        action_type: ActionType | str,
        order_code: str,
        target_date: date | None = None,
        new_address: str | None = None,
        reversal_of: Mapping[str, Any] | None = None,
        action_id: str | None = None,
        idempotency_key: str | None = None,
        steps_required: int = 1,
    ) -> WriteOutcome:
        """Zero-mutation preview. Returns PREVIEW with a staged ``pending`` action, or a refusal."""
        kind = ActionType(action_type)
        if not confirmation_steps_valid(steps_required):
            raise ValueError("confirmation takes 1-2 steps")
        order = await self.repository.get_order_by_code(order_code)
        if order is None:
            return WriteOutcome("NOT_FOUND", copy.order_not_found(order_code))
        if not scope_allows(order["customer_id"], customer_id):
            await self.audit.write(
                session_id=session_id,
                action_name=ActionName.AUTH_SCOPE_VIOLATION,
                actor=Actor.SYSTEM,
                action_status=ActionStatus.DENIED,
                outcome_status=AuditOutcome.DENIED,
                order_id=order["order_id"],
                request={
                    "tool_name": TOOL_FOR_ACTION[kind],
                    "order_code": order_code,
                    "session_customer_id": customer_id,
                },
                response={"error_code": "AUTH_SCOPE_VIOLATION", "details_concealed": True},
            )
            return WriteOutcome(
                "OUT_OF_SCOPE", copy.order_out_of_scope_full(order_code), audit_names=["AUTH_SCOPE_VIOLATION"]
            )
        shipment = await self.repository.get_shipment_by_order(order["order_id"])
        payload: dict[str, Any] = {"order_code": order_code}
        changes: dict[str, Any]
        if kind is ActionType.CANCEL:
            if is_dispatched(order["status"]):
                return WriteOutcome("OPD_03", copy.OPD_03_CANCEL_SHIPPED)
            if not cancellable(order["status"]):
                return WriteOutcome("REFUSED", copy.WRITE_NOT_CANCELLABLE)
            changes = {"status": "cancelled"}
            if order["payment_status"] == "PAID":
                changes["payment_status"] = "REFUND_PENDING"
            payload.update(
                prior_status=order["status"],
                prior_payment_status=order["payment_status"],
                fee_vnd=0,
                refund_vnd=int(order["total_amount"]) if order["payment_status"] == "PAID" else 0,
            )
            spoken = copy.cancel_preview(order_code, int(order["total_amount"]))
        elif kind is ActionType.RESCHEDULE:
            if target_date is None:
                return WriteOutcome("NEEDS_DATE", "Which day would you like the delivery moved to?")
            today = self.clock.now().astimezone(DEMO_TIMEZONE).date()
            if target_date <= today or order["status"] in {"delivered", "cancelled", "returned", "return_requested"}:
                return WriteOutcome(
                    "REFUSED", "I cannot move that delivery to that day. I can ask a human colleague to help."
                )
            current = _local_datetime(order.get("requested_delivery_time"))
            if current is not None and current.date() == target_date:  # the caller asked for the day already booked
                end_at = _local_datetime(order.get("requested_delivery_window_end"))
                window = f"{current:%H:%M}-{end_at:%H:%M}" if end_at is not None else None
                return WriteOutcome("UNCHANGED", copy.reschedule_unchanged(order_code, target_date, window))
            start, end = _window(target_date)
            changes = {"requested_delivery_time": start, "requested_delivery_window_end": end}
            payload.update(
                target_date=target_date.isoformat(), window=f"{DEFAULT_SLOT[0]}-{DEFAULT_SLOT[1]}", fee_vnd=0
            )
            spoken = copy.reschedule_preview(order_code, target_date)
        elif kind is ActionType.ADDRESS_CHANGE:
            decision = reverse_address_change_allowed(
                order_status=order["status"],
                shipment_status=shipment["status"] if shipment else None,
                oms_confirms=True,
            )
            if decision.allowed and not new_address:
                return WriteOutcome("NEEDS_ADDRESS", "What is the new delivery address?")
            if not decision.allowed:
                return WriteOutcome(
                    "REFUSED",
                    "I cannot change the address on that order by phone. I can ask a human colleague to help.",
                )
            changes = {"shipping_address": new_address}
            payload.update(new_address=new_address, fee_vnd=0)
            spoken = copy.address_preview(order_code)
        else:  # Reversal (UC-22): a new request with a new key and reversal_of_action_id
            if reversal_of is None:
                return WriteOutcome("REFUSED", copy.REVERSAL_REFUSED_GENERIC)
            facts = self.oms.reversal_facts(order_code) if self.oms is not None else {}
            decision = (
                reinstate_cancel_allowed(
                    order_status=order["status"],
                    shipment_status=shipment["status"] if shipment else None,
                    inventory_reserved=facts.get("inventory_reserved"),
                    refund_dispatched=facts.get("refund_dispatched"),
                    version_current=True,
                    oms_validates=facts.get("oms_validates"),
                )
                if reversal_of.get("action_type") == ActionType.CANCEL
                else None
            )
            request = {
                "tool_name": "request_reversal",
                "order_code": order_code,
                "reversal_of_action_id": reversal_of["action_id"],
            }
            if decision is None or not decision.allowed:
                await self.audit.write(
                    session_id=session_id,
                    action_name=ActionName.REVERSAL_REFUSED,
                    actor=Actor.SYSTEM,
                    action_status=ActionStatus.DENIED,
                    outcome_status=AuditOutcome.DENIED,
                    order_id=order["order_id"],
                    request=request,
                    response={"reason": decision.reason if decision else "UNSUPPORTED_REVERSAL", "mutation": "none"},
                )
                spoken = (
                    copy.REVERSAL_REFUSED_REFUND_DISPATCHED
                    if decision is not None and decision.reason in {"REFUND_DISPATCHED", "INVENTORY_RELEASED"}
                    else copy.REVERSAL_REFUSED_GENERIC
                )
                return WriteOutcome("REFUSED", spoken, audit_names=["REVERSAL_REFUSED"])
            snapshot = reversal_of.get("action_payload_json", {})
            changes = {
                "status": snapshot.get("prior_status", "confirmed"),
                "payment_status": snapshot.get("prior_payment_status", order["payment_status"]),
            }
            payload.update(reversal_of_action_id=reversal_of["action_id"], restore_status=changes["status"])
            spoken = copy.REINSTATE_PREVIEW
        key = idempotency_key or str(uuid.uuid4())
        if reversal_of is not None and key == reversal_of.get("idempotency_key"):
            raise ValueError("a reversal needs a new idempotency key")
        payload["changes"] = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in changes.items()}
        action = {
            "action_id": action_id or f"ACT-{secrets.randbelow(9000) + 1000}",
            "session_id": session_id,
            "target_order_id": order["order_id"],
            "target_order_code": order_code,
            "action_type": str(kind),
            "action_payload_json": payload,
            "preview_hash": preview_hash(payload),
            "idempotency_key": key,
            "status": str(PendingActionStatus.PENDING),
            "expected_version": int(order["version"]),
            "expires_at": (self.clock.now() + timedelta(seconds=self.store.pending_ttl_s)).astimezone(UTC).isoformat(),
            "confirmations": 0,
            "steps_required": steps_required,
            "customer_id": customer_id,
        }
        try:
            await self.store.create_pending_action(action)
        except DuplicateIdempotencyKey:
            raise ValueError("idempotency key already in use") from None
        names = []
        if kind is ActionType.REVERSAL:
            await self.audit.write(
                session_id=session_id,
                action_name=ActionName.REVERSAL_REQUESTED,
                actor=Actor.SYSTEM,
                action_status=ActionStatus.SUCCESS,
                outcome_status=AuditOutcome.PENDING,
                order_id=order["order_id"],
                mutation_idempotency_key=key,
                request={**request, "action_id": action["action_id"], "idempotency_key": key},
                response={"pending_action_status": "pending", "eligible": True},
            )
            names.append("REVERSAL_REQUESTED")
        return WriteOutcome("PREVIEW", spoken, action, names)

    # ------------------------------------------------------------------ confirm

    async def confirm(self, action_id: str) -> WriteOutcome:
        """One explicit oral confirmation. The last required step seals ``pending → submitted``."""
        action = await self.store.get_pending_action(action_id)
        if action is None:
            return WriteOutcome("EXPIRED", copy.WRITE_PREVIEW_EXPIRED)
        if action["status"] != PendingActionStatus.PENDING:
            return WriteOutcome("NOT_PENDING", copy.REPLAY_STILL_PROCESSING, action)
        current = await self.repository.get_order_by_code(action["target_order_code"])
        if current is None or preview_hash(action["action_payload_json"]) != action["preview_hash"]:
            return await self._fail(action, "PREVIEW_INVALID")
        action["confirmations"] += 1
        if action["confirmations"] < action["steps_required"]:
            await self.store.save_pending_action(action)
            return WriteOutcome("CONFIRM_AGAIN", "Please confirm once more.", action)
        await self._advance(action, PendingActionStatus.SUBMITTED)
        await self._audit(
            action,
            ActionName.EXECUTION_PENDING,
            ActionStatus.SUCCESS,
            AuditOutcome.PENDING,
            response={"pending_action_status": "submitted"},
        )
        return WriteOutcome(
            "SUBMITTED", copy.write_recorded(action["target_order_code"]), action, ["EXECUTION_PENDING"]
        )

    # ------------------------------------------------------------------ execute (worker)

    async def execute(self, action_id: str) -> WriteOutcome:
        """Revalidate under the OCC gate and send the sealed request to VoxCom (same key on every retry)."""
        if self.oms is None:
            raise RuntimeError("no OMS client: Phase 1 records requests only")
        action = await self.store.get_pending_action(action_id)
        if action is None:
            return WriteOutcome("EXPIRED", copy.WRITE_PREVIEW_EXPIRED)
        if action["status"] == PendingActionStatus.SUBMITTED:
            order = await self.repository.get_order_by_code(action["target_order_code"])
            if order is None or int(order["version"]) != int(action["expected_version"]):
                return await self._fail(action, "STATE_CONFLICT", oms_status="FAILED_NO_COMMIT")
            await self._advance(action, PendingActionStatus.EXECUTING)
        elif action["status"] != PendingActionStatus.EXECUTING:
            return self._terminal_reply(action)
        return await self._send(action)

    async def replay(self, action_id: str) -> WriteOutcome:
        """Re-send an action by its original key (Scenario 16); the key is never re-minted."""
        if self.oms is None:
            raise RuntimeError("no OMS client")
        action = await self.store.get_pending_action(action_id)
        if action is None:
            return WriteOutcome("EXPIRED", copy.WRITE_PREVIEW_EXPIRED)
        response = await self._call_execute(action)
        if response is None:
            return await self._unknown(action)
        return await self._apply_response(action, response)

    async def _send(self, action: dict[str, Any]) -> WriteOutcome:
        response = await self._call_execute(action)
        if response is None:
            return await self._unknown(action)
        return await self._apply_response(action, response)

    async def _call_execute(self, action: Mapping[str, Any]) -> dict[str, Any] | None:
        """Original request plus one same-key retry; ``None`` when both responses are lost."""
        assert self.oms is not None  # noqa: S101
        request = {
            "order_code": action["target_order_code"],
            "expected_version": action["expected_version"],
            "changes": action["action_payload_json"]["changes"],
            "action_type": action["action_type"],
        }
        for _ in range(2):
            try:
                return await self.oms.execute(action["idempotency_key"], request)
            except TimeoutError:
                logger.warning("OMS response lost for %s; retrying with the same key", action["action_id"])
        return None

    async def _apply_response(self, action: dict[str, Any], response: Mapping[str, Any]) -> WriteOutcome:
        status = response.get("status")
        code = action["target_order_code"]
        if response.get("idempotent_replay"):
            if action["status"] == PendingActionStatus.EXECUTING:
                await self._complete(action, response)
            return WriteOutcome("ALREADY_APPLIED", copy.REPLAY_ALREADY_APPLIED, action)
        if status in {"IN_PROGRESS", "ACCEPTED"}:
            return WriteOutcome("IN_PROGRESS", copy.REPLAY_STILL_PROCESSING, action)
        if status == "COMMITTED":
            return await self._complete(action, response)
        if status == "ABORTED":
            return WriteOutcome("CANCELLED", copy.withdrawn_executing(code), action)
        return await self._fail(action, "STATE_CONFLICT", oms_status="FAILED_NO_COMMIT")

    async def _complete(self, action: dict[str, Any], response: Mapping[str, Any]) -> WriteOutcome:
        await self._advance(action, PendingActionStatus.COMPLETED)
        kind = ActionType(action["action_type"])
        payload = action["action_payload_json"]
        code = action["target_order_code"]
        result = {
            "pending_action_status": "completed",
            "oms_status": response.get("status", "COMMITTED"),
            "new_version": response.get("new_version"),
        }
        if kind in COMPLETION_AUDIT:
            await self._audit(
                action, COMPLETION_AUDIT[kind], ActionStatus.SUCCESS, AuditOutcome.SUCCESS, response=result
            )
            name = str(COMPLETION_AUDIT[kind])
        else:
            await self._audit(
                action, ActionName.TOOL_INVOCATION, ActionStatus.SUCCESS, AuditOutcome.SUCCESS, response=result
            )
            name = "TOOL_INVOCATION"
        if kind is ActionType.CANCEL:
            spoken = copy.cancel_done(code, payload.get("refund_vnd", 0))
        elif kind is ActionType.RESCHEDULE:
            spoken = copy.reschedule_done(code, date.fromisoformat(payload["target_date"]))
        elif kind is ActionType.REVERSAL:
            spoken = copy.reinstated(code)
        else:
            spoken = f"The delivery address for {code} is updated."
        return WriteOutcome("COMPLETED", spoken, action, [name])

    async def _unknown(self, action: dict[str, Any]) -> WriteOutcome:
        """Both responses lost: stay ``executing``, audit UNKNOWN, then query status by the same key."""
        await self._audit(
            action,
            ActionName.MUTATION_OUTCOME_UNKNOWN,
            ActionStatus.SUCCESS,
            AuditOutcome.UNKNOWN,
            response={"mutation_outcome": "UNKNOWN", "reason": "original_and_same_key_retry_responses_lost"},
        )
        return WriteOutcome("UNKNOWN", copy.REPLAY_OUTCOME_UNKNOWN, action, ["MUTATION_OUTCOME_UNKNOWN"])

    async def reconcile(self, action_id: str) -> WriteOutcome:
        """Query VoxCom by the original key; resolve only on an authoritative answer."""
        assert self.oms is not None  # noqa: S101
        action = await self.store.get_pending_action(action_id)
        if action is None:
            return WriteOutcome("EXPIRED", copy.WRITE_PREVIEW_EXPIRED)
        try:
            status = await self.oms.status(action["idempotency_key"])
        except TimeoutError:
            status = None
        if not status:
            return WriteOutcome("UNKNOWN", copy.REPLAY_OUTCOME_UNKNOWN, action)
        if status.get("state") == "completed":
            return await self._complete(action, status)
        if status.get("state") == "failed":
            return await self._fail(action, "STATE_CONFLICT", oms_status="FAILED_NO_COMMIT")
        return WriteOutcome("IN_PROGRESS", copy.REPLAY_STILL_PROCESSING, action)

    # ------------------------------------------------------------------ withdrawal (UC-21)

    async def withdraw(self, action_id: str, *, actor: Actor = Actor.CALLER) -> WriteOutcome:
        action = await self.store.get_pending_action(action_id)
        if action is None:
            return WriteOutcome("EXPIRED", copy.WRITE_PREVIEW_EXPIRED)
        code = action["target_order_code"]
        state = action["status"]
        if state == PendingActionStatus.PENDING:  # Case A: discarded locally, zero OMS interaction
            await self._advance(action, PendingActionStatus.CANCELLED)
            await self._audit(
                action,
                ActionName.ACTION_ABORTED_PRE_SUBMIT,
                ActionStatus.ABORTED,
                AuditOutcome.ABORTED,
                actor=actor,
                response={"pending_action_status": "cancelled", "oms_interaction": False},
            )
            return WriteOutcome("CANCELLED", copy.withdrawn_pending(code), action, ["ACTION_ABORTED_PRE_SUBMIT"])
        if state == PendingActionStatus.SUBMITTED:  # Case B: intercepted before execution
            await self._advance(action, PendingActionStatus.CANCELLED)
            await self._audit(
                action,
                ActionName.ACTION_ABORTED_POST_SUBMIT,
                ActionStatus.ABORTED,
                AuditOutcome.ABORTED,
                response={"pending_action_status": "cancelled", "mutations": 0},
            )
            return WriteOutcome("CANCELLED", copy.withdrawn_submitted(code), action, ["ACTION_ABORTED_POST_SUBMIT"])
        if state == PendingActionStatus.EXECUTING:  # Case C: ask VoxCom to abort; confirm only on its answer
            if self.oms is None:
                return WriteOutcome("ASKING", copy.WITHDRAW_ASKING_OMS, action)
            try:
                answer = await self.oms.abort(action["idempotency_key"])
            except TimeoutError:
                return WriteOutcome("ASKING", copy.WITHDRAW_ASKING_OMS, action)
            if answer == "ABORTED":
                await self._advance(action, PendingActionStatus.CANCELLED)
                await self._audit(
                    action,
                    ActionName.WITHDRAWAL_CONFIRMED,
                    ActionStatus.SUCCESS,
                    AuditOutcome.ABORTED,
                    response={"pending_action_status": "cancelled", "oms_status": "ABORTED"},
                )
                return WriteOutcome("CANCELLED", copy.withdrawn_executing(code), action, ["WITHDRAWAL_CONFIRMED"])
            if answer == "ALREADY_COMMITTED":
                status = await self.oms.status(action["idempotency_key"]) or {"status": "COMMITTED"}
                done = await self._complete(action, status)
                return WriteOutcome("TOO_LATE", copy.WITHDRAW_TOO_LATE, action, done.audit_names)
            return WriteOutcome("ASKING", copy.WITHDRAW_ASKING_OMS, action)
        if state == PendingActionStatus.COMPLETED:
            return WriteOutcome("TOO_LATE", copy.WITHDRAW_TOO_LATE, action)
        return self._terminal_reply(action)

    # ------------------------------------------------------------------ helpers

    def _terminal_reply(self, action: Mapping[str, Any]) -> WriteOutcome:
        code = action["target_order_code"]
        if action["status"] == PendingActionStatus.COMPLETED:
            return WriteOutcome("ALREADY_APPLIED", copy.REPLAY_ALREADY_APPLIED, dict(action))
        if action["status"] == PendingActionStatus.FAILED:
            return WriteOutcome("FAILED", copy.replay_conflict(code), dict(action))
        return WriteOutcome("CANCELLED", copy.withdrawn_pending(code), dict(action))

    async def _advance(self, action: dict[str, Any], to_status: PendingActionStatus) -> None:
        if not can_advance_action(action["status"], to_status):
            raise ValueError(f"illegal action transition {action['status']} -> {to_status}")
        action["status"] = str(to_status)
        await self.store.save_pending_action(action)

    async def _fail(self, action: dict[str, Any], reason: str, *, oms_status: str | None = None) -> WriteOutcome:
        if action["status"] in {
            PendingActionStatus.PENDING,
            PendingActionStatus.SUBMITTED,
            PendingActionStatus.EXECUTING,
        }:
            await self._advance(action, PendingActionStatus.FAILED)
        response: dict[str, Any] = {"pending_action_status": "failed", "error_code": reason}
        if oms_status:
            response["oms_status"] = oms_status
        await self._audit(
            action, ActionName.TOOL_INVOCATION, ActionStatus.FAILED, AuditOutcome.FAILED, response=response
        )
        return WriteOutcome("FAILED", copy.replay_conflict(action["target_order_code"]), action, ["TOOL_INVOCATION"])

    async def _audit(
        self,
        action: Mapping[str, Any],
        name: ActionName,
        status: ActionStatus,
        outcome: AuditOutcome,
        *,
        actor: Actor = Actor.SYSTEM,
        response: Mapping[str, Any] | None = None,
    ) -> None:
        request = {
            "action_id": action["action_id"],
            "tool_name": TOOL_FOR_ACTION[ActionType(action["action_type"])],
            "idempotency_key": action["idempotency_key"],
            "order_code": action["target_order_code"],
        }
        reversal_of = action["action_payload_json"].get("reversal_of_action_id")
        if reversal_of:
            request["reversal_of_action_id"] = reversal_of
        await self.audit.write(
            session_id=action["session_id"],
            action_name=name,
            actor=actor,
            action_status=status,
            outcome_status=outcome,
            order_id=action["target_order_id"],
            mutation_idempotency_key=action["idempotency_key"],
            request=request,
            response=response,
        )


def _window(day: date) -> tuple[datetime, datetime]:
    start_h, start_m = (int(part) for part in DEFAULT_SLOT[0].split(":"))
    end_h, end_m = (int(part) for part in DEFAULT_SLOT[1].split(":"))
    return (
        datetime(day.year, day.month, day.day, start_h, start_m, tzinfo=DEMO_TIMEZONE),
        datetime(day.year, day.month, day.day, end_h, end_m, tzinfo=DEMO_TIMEZONE),
    )


__all__ = ["COMPLETION_AUDIT", "TOOL_FOR_ACTION", "OmsClient", "WriteOutcome", "WriteService"]
