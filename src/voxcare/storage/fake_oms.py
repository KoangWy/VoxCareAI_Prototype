"""A fake VoxCom OMS for the guarded-write library (guide §11; design D§6). Test and demo use only.

It executes a sealed action under the OCC gate against the repository::

    UPDATE orders SET ..., version = version + 1, updated_at = NOW()
    WHERE order_code = :order_code AND version = :expected_version;   -- 0 rows → STATE_CONFLICT

and keeps an idempotency table keyed by the action's UUIDv4 key, so a replay of
the same key never executes twice (D§6.4):

=================  ==================================================================
key state          ``execute`` answer
=================  ==================================================================
executing          409 / 202 ``IN_PROGRESS`` with ``retry_after: 1``; no second execution
completed          the cached result with ``idempotent_replay: True``
failed             ``FAILED_NO_COMMIT`` (permanent STATE_CONFLICT)
=================  ==================================================================

Knobs for the scenarios: ``hold`` keeps a request in flight (accepted, not yet
committed); ``lose_responses`` commits but loses that many responses
(the caller sees a timeout); ``abort_policy`` decides an abort of an in-flight
request ("accept" before commit, or "too_late" when the commit already landed).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .db import Repository


class OmsResponseLost(TimeoutError):
    """The OMS committed (or not) but the response never arrived."""


@dataclass
class _Entry:
    action: dict[str, Any]
    state: str = "executing"  # executing | completed | failed | cancelled
    committed: bool = False
    result: dict[str, Any] = field(default_factory=dict)


class FakeVoxComOMS:
    def __init__(self, repository: Repository, now: Callable[[], datetime]) -> None:
        self.repository = repository
        self.now = now
        self.table: dict[str, _Entry] = {}
        self.executions: list[str] = []  # keys that reached a real UPDATE (never two per key)
        self.hold = False
        self.lose_responses = 0
        self.abort_policy = "accept"
        self.status_available = True
        # Reversal facts the OMS owns (guide §11.4)
        self.refund_dispatched: dict[str, bool] = {}
        self.inventory_reserved: dict[str, bool] = {}
        self.carrier_results: dict[str, str] = {}

    # -- execution --

    async def execute(self, key: str, action: Mapping[str, Any]) -> dict[str, Any]:
        entry = self.table.get(key)
        if entry is not None:
            if self.lose_responses > 0:  # the same-key retry's response is lost too
                self.lose_responses -= 1
                raise OmsResponseLost(key)
            if entry.state == "executing":
                return {"http": 409, "status": "IN_PROGRESS", "retry_after": 1}
            if entry.state == "completed":
                return {**entry.result, "idempotent_replay": True}
            if entry.state == "cancelled":
                return {"http": 409, "status": "ABORTED"}
            return {"http": 409, "status": "FAILED_NO_COMMIT", "error_code": "STATE_CONFLICT"}
        entry = self.table[key] = _Entry(action=dict(action))
        if self.hold:
            return {"http": 202, "status": "ACCEPTED", "retry_after": 1}
        return await self._commit(key)

    async def commit_held(self, key: str) -> dict[str, Any]:
        """Finish a request that ``hold`` kept in flight."""
        entry = self.table[key]
        if entry.state != "executing" or entry.committed:
            return entry.result
        return await self._commit(key)

    async def _commit(self, key: str) -> dict[str, Any]:
        entry = self.table[key]
        action = entry.action
        changes = {
            column: datetime.fromisoformat(value)
            if column.startswith("requested_delivery") and isinstance(value, str)
            else value
            for column, value in dict(action["changes"]).items()
        }
        rows = await self.repository.update_order_occ(
            action["order_code"], int(action["expected_version"]), changes, self.now()
        )
        self.executions.append(key)
        if rows == 0:
            entry.state = "failed"
            entry.result = {"http": 409, "status": "FAILED_NO_COMMIT", "error_code": "STATE_CONFLICT"}
        else:
            entry.state = "completed"
            entry.committed = True
            entry.result = {
                "http": 200,
                "status": "COMMITTED",
                "order_code": action["order_code"],
                "new_version": int(action["expected_version"]) + 1,
            }
        if self.lose_responses > 0:
            self.lose_responses -= 1
            raise OmsResponseLost(key)
        return dict(entry.result)

    async def abort(self, key: str) -> str:
        """``ABORTED`` before commit, ``ALREADY_COMMITTED`` after it, ``NOT_FOUND`` for an unknown key."""
        entry = self.table.get(key)
        if entry is None:
            return "NOT_FOUND"
        if entry.committed or self.abort_policy == "too_late":
            if not entry.committed:
                await self._commit_quietly(key)
            return "ALREADY_COMMITTED"
        entry.state = "cancelled"
        entry.result = {"http": 200, "status": "ABORTED"}
        return "ABORTED"

    async def _commit_quietly(self, key: str) -> None:
        lost, self.lose_responses = self.lose_responses, 0
        try:
            await self._commit(key)
        finally:
            self.lose_responses = lost

    async def status(self, key: str) -> dict[str, Any] | None:
        """Operation status by idempotency key (reconciliation after an UNKNOWN outcome)."""
        await asyncio.sleep(0)
        if not self.status_available:
            raise OmsResponseLost(key)
        entry = self.table.get(key)
        if entry is None:
            return None
        return {"state": entry.state, **entry.result}

    # -- reversal facts --

    def reversal_facts(self, order_code: str) -> dict[str, Any]:
        return {
            "refund_dispatched": self.refund_dispatched.get(order_code, False),
            "inventory_reserved": self.inventory_reserved.get(order_code, True),
            "oms_validates": True,
            "carrier_result": self.carrier_results.get(order_code),
        }


__all__ = ["FakeVoxComOMS", "OmsResponseLost"]
