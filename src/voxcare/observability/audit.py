"""Audit writer (guide §16, design D§3.3.10 / D§3.5): append-only rows in ``audit_log``.

- Every row gets a fresh UUIDv4 ``audit_id``.
- ``tool_name`` goes in ``request_payload_json`` (there is no column for it).
- Reads, handoffs, denials and provider events carry a null ``mutation_idempotency_key``;
  a later resolution of the same mutation gets a new ``audit_id`` and keeps the same key.
- ``order_id`` is the 32-char hex FK; the order **code** is also written into
  ``request_payload_json`` (G-4).
- Payloads are PII-masked before they are persisted.

Each written row is also handed to ``listeners`` (the session trace shown in the UI).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC
from typing import Any

from ..domain.clock import Clock
from ..domain.enums import ActionName, ActionStatus, Actor, AuditOutcome
from ..security import pii
from ..storage.db import Repository

logger = logging.getLogger("voxcare.audit")

AUDIT_FIELDS = (
    "audit_id",
    "session_id",
    "order_id",
    "mutation_idempotency_key",
    "actor",
    "action_name",
    "action_status",
    "outcome_status",
    "request_payload_json",
    "response_payload_json",
    "timestamp",
)

Listener = Callable[[Mapping[str, Any]], None]


class AuditWriter:
    def __init__(self, repository: Repository, clock: Clock) -> None:
        self.repository = repository
        self.clock = clock
        self.listeners: list[Listener] = []

    async def write(
        self,
        *,
        session_id: str,
        action_name: ActionName | str,
        actor: Actor | str,
        action_status: ActionStatus | str,
        outcome_status: AuditOutcome | str,
        order_id: str | None = None,
        mutation_idempotency_key: str | None = None,
        request: Mapping[str, Any] | None = None,
        response: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "audit_id": str(uuid.uuid4()),
            "session_id": session_id,
            "order_id": order_id,
            "mutation_idempotency_key": mutation_idempotency_key,
            "actor": str(Actor(actor)),
            "action_name": str(ActionName(action_name)),
            "action_status": str(ActionStatus(action_status)),
            "outcome_status": str(AuditOutcome(outcome_status)),
            "request_payload_json": pii.mask_payload(dict(request or {})),
            "response_payload_json": pii.mask_payload(dict(response or {})),
            "timestamp": self.clock.now().astimezone(UTC),
        }
        await self.repository.insert_audit(row)
        for listener in list(self.listeners):
            try:
                listener(row)
            except Exception:  # a broken listener never blocks the audit append
                logger.exception("audit listener failed")
        return row


def audit_json(row: Mapping[str, Any]) -> dict[str, Any]:
    """The D§3.5 JSON shape of one row (timestamps as ``YYYY-MM-DDTHH:MM:SSZ``)."""
    out = {field: row.get(field) for field in AUDIT_FIELDS}
    stamp = out["timestamp"]
    if hasattr(stamp, "astimezone"):
        out["timestamp"] = stamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    for key in ("audit_id", "mutation_idempotency_key"):
        if out[key] is not None:
            out[key] = str(out[key])
    return out


__all__ = ["AUDIT_FIELDS", "AuditWriter", "audit_json"]
