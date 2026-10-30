"""Small in-memory, session-correlated audit and timing event ledger."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any


class TraceLedger:
    """Keep only explicit, structured prototype events; never store credentials/audio."""

    MAX_EVENTS = 500

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def record_action(
        self,
        *,
        session_id: str,
        action_name: str,
        action_status: str = "Success",
        outcome_status: str = "SUCCESS",
        turn_id: str | None = None,
        turn_epoch: int | None = None,
        order_id: str | None = None,
        request_payload: Mapping[str, Any] | None = None,
        response_payload: Mapping[str, Any] | None = None,
        actor: str = "System",
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "audit_id": str(uuid.uuid4()),
            "session_id": session_id,
            "order_id": order_id,
            "mutation_idempotency_key": None,
            "actor": actor,
            "action_name": action_name,
            "action_status": action_status,
            "outcome_status": outcome_status,
            "request_payload_json": dict(request_payload or {}),
            "response_payload_json": dict(response_payload or {}),
            "timestamp": datetime.now(UTC).isoformat(),
            "turn_id": turn_id,
            "turn_epoch": turn_epoch,
        }
        return self._append(event)

    def _append(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append with the size bound; oldest events drop first."""
        self._events.append(event)
        del self._events[: -self.MAX_EVENTS]
        return event

    def record_observation(self, *, session_id: str, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type.upper(),
            "session_id": session_id,
            "observed_at_utc": datetime.now(UTC).isoformat(),
            **fields,
        }
        return self._append(event)

    def append_audit(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Retain the controller's schema-valid audit row with its original UUID."""
        copied = dict(event)
        return self._append(copied)

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._events)
