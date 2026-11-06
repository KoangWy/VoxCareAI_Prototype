"""Transient entities in Redis (guide §4.2): Session, PendingAction, idempotency index, HandoffContextPacket, throttle.

Key names are an implementation choice; fields and TTLs are design values
(D§3.3.11, 3.3.12, 3.3.16). With ``REDIS_URL`` unset the same code runs
against an in-process ``fakeredis`` server, so every command path is identical;
``/healthz`` and ``/api/config`` report which one is active.

TTLs are wall-clock seconds enforced by Redis itself. The *domain* clock
(pinned to the demo anchor) is never used for expiry, so a pending action does
expire 300 s after preview even while the demo date stays fixed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

SESSION_TTL_S = 1800
PENDING_ACTION_TTL_S = 300
THROTTLE_WINDOW_S = 120


class DuplicateIdempotencyKey(ValueError):
    """``idem:{key}`` already exists: a key is unique and a retry must reuse the original action."""


def create_redis_client(url: str | None) -> tuple[Any, str]:
    """``(client, backend)``: a real Redis for ``url``, else an in-process fakeredis server."""
    if url:
        from redis.asyncio import Redis

        return Redis.from_url(url, decode_responses=True, socket_timeout=1.5, socket_connect_timeout=1.5), "redis"
    import fakeredis

    return fakeredis.FakeAsyncRedis(decode_responses=True), "fakeredis"


class RedisStore:
    def __init__(
        self,
        client: Any,
        *,
        backend: str = "redis",
        session_ttl_s: int = SESSION_TTL_S,
        pending_ttl_s: int = PENDING_ACTION_TTL_S,
        throttle_window_s: int = THROTTLE_WINDOW_S,
        now_s: Callable[[], float] = time.time,
    ) -> None:
        self.client = client
        self.backend = backend
        self.session_ttl_s = session_ttl_s
        self.pending_ttl_s = pending_ttl_s
        self.throttle_window_s = throttle_window_s
        self.now_s = now_s

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def close(self) -> None:
        close = getattr(self.client, "aclose", None) or getattr(self.client, "close", None)
        if close is not None:
            await close()

    # -- Session (session:{session_id}, TTL 1800 s refreshed on activity) --

    async def put_session(self, record: Mapping[str, Any]) -> None:
        await self.client.set(f"session:{record['session_id']}", _dump(record), ex=self.session_ttl_s)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        raw = await self.client.get(f"session:{session_id}")
        return json.loads(raw) if raw else None

    async def update_session(self, session_id: str, **fields: Any) -> dict[str, Any] | None:
        record = await self.get_session(session_id)
        if record is None:
            return None
        record.update(fields)
        await self.put_session(record)
        return record

    async def touch_session(self, session_id: str) -> bool:
        return bool(await self.client.expire(f"session:{session_id}", self.session_ttl_s))

    async def delete_session(self, session_id: str) -> None:
        await self.client.delete(f"session:{session_id}", f"throttle:{session_id}")

    async def session_ttl(self, session_id: str) -> int:
        ttl = int(await self.client.ttl(f"session:{session_id}"))
        return ttl if ttl > 0 else self.session_ttl_s

    # -- PendingAction (pending_action:{action_id} 300 s) + idem:{key} uniqueness index --

    async def create_pending_action(self, action: Mapping[str, Any]) -> None:
        key = str(action["idempotency_key"])
        claimed = await self.client.set(f"idem:{key}", action["action_id"], ex=self.pending_ttl_s, nx=True)
        if not claimed:
            raise DuplicateIdempotencyKey(key)
        await self.client.set(f"pending_action:{action['action_id']}", _dump(action), ex=self.pending_ttl_s)

    async def get_pending_action(self, action_id: str) -> dict[str, Any] | None:
        raw = await self.client.get(f"pending_action:{action_id}")
        return json.loads(raw) if raw else None

    async def save_pending_action(self, action: Mapping[str, Any]) -> None:
        """Rewrite the action keeping its remaining TTL (the confirmation window never restarts)."""
        await self.client.set(f"pending_action:{action['action_id']}", _dump(action), keepttl=True)

    async def action_for_key(self, idempotency_key: str) -> str | None:
        return await self.client.get(f"idem:{idempotency_key}")

    # -- HandoffContextPacket (handoff:{handoff_id}, expires with the session) --

    async def put_handoff(self, packet: Mapping[str, Any]) -> None:
        ttl = await self.session_ttl(packet["session_id"])
        await self.client.set(f"handoff:{packet['handoff_id']}", _dump(packet), ex=ttl)
        await self.client.rpush(f"session_handoffs:{packet['session_id']}", packet["handoff_id"])
        await self.client.expire(f"session_handoffs:{packet['session_id']}", ttl)

    async def get_handoff(self, handoff_id: str) -> dict[str, Any] | None:
        raw = await self.client.get(f"handoff:{handoff_id}")
        return json.loads(raw) if raw else None

    async def list_handoffs(self, session_id: str) -> list[dict[str, Any]]:
        ids = await self.client.lrange(f"session_handoffs:{session_id}", 0, -1)
        packets = [await self.get_handoff(handoff_id) for handoff_id in ids]
        return [packet for packet in packets if packet is not None]

    # -- throttle:{session_id}: sorted set of distinct SKUs scored by time (FR-SEC-01) --

    async def throttle_window(self, session_id: str) -> set[str]:
        key = f"throttle:{session_id}"
        now = self.now_s()
        await self.client.zremrangebyscore(key, "-inf", now - self.throttle_window_s)
        return set(await self.client.zrange(key, 0, -1))

    async def throttle_record(self, session_id: str, skus: Iterable[str]) -> int:
        key = f"throttle:{session_id}"
        now = self.now_s()
        mapping = {sku: now for sku in skus}
        if mapping:
            await self.client.zadd(key, mapping)
        await self.client.zremrangebyscore(key, "-inf", now - self.throttle_window_s)
        await self.client.expire(key, self.throttle_window_s)
        return int(await self.client.zcard(key))


def _dump(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


__all__ = [
    "PENDING_ACTION_TTL_S",
    "SESSION_TTL_S",
    "THROTTLE_WINDOW_S",
    "DuplicateIdempotencyKey",
    "RedisStore",
    "create_redis_client",
]
