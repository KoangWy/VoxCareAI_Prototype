"""Shared test helpers: a scripted conversation adapter and boundaries over memory or PostgreSQL/Redis.

``VOXCARE_TEST_DATABASE_URL`` / ``VOXCARE_TEST_REDIS_URL`` point the store-parametrised
tests at real servers; without them those parameters are skipped.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Sequence
from typing import Any

import pytest

from voxcare.config import load_data_settings
from voxcare.contracts import PlannedToolCall
from voxcare.domain.clock import FixedClock
from voxcare.runtime import Runtime, _policy_store, assemble, build_memory_runtime
from voxcare.session.boundary import SessionBoundary
from voxcare.storage.db import MemoryRepository, PostgresRepository
from voxcare.storage.redis_store import RedisStore, create_redis_client

P = PlannedToolCall
TEST_DATABASE_URL = os.getenv("VOXCARE_TEST_DATABASE_URL", "")
TEST_REDIS_URL = os.getenv("VOXCARE_TEST_REDIS_URL", "")
BACKENDS = [
    "memory",
    pytest.param("postgres", marks=pytest.mark.skipif(not TEST_DATABASE_URL, reason="VOXCARE_TEST_DATABASE_URL unset")),
]


class ScriptedPlanner:
    """Deterministic stand-in for the LLM: one planned call list per turn, in order (guide §18 level a)."""

    def __init__(self, turns: Iterable[Sequence[PlannedToolCall]] = ()) -> None:
        self.turns = [turn if hasattr(turn, "note") else list(turn) for turn in turns]
        self.seen: list[dict[str, Any]] = []

    async def plan_turn(self, *, user_text, history, available_tools):  # type: ignore[no-untyped-def]
        self.seen.append(
            {"user_text": user_text, "history": list(history), "tools": [t["name"] for t in available_tools]}
        )
        return self.turns.pop(0) if self.turns else []


async def postgres_runtime(*, clock: FixedClock | None = None, **overrides: Any) -> Runtime:
    """A runtime on a fresh PostgreSQL schema (and real Redis when configured)."""
    env = {
        "VOXCARE_DATA_BACKEND": "postgres",
        "VOXCARE_POLICY_EMBEDDING": "hashing",
        "CHROMA_PATH": "",
        "DATABASE_URL": TEST_DATABASE_URL,
        "VOXCARE_DB_SCHEMA": f"t_{uuid.uuid4().hex[:12]}",
    }
    settings = load_data_settings(env)
    settings = type(settings)(**{**settings.__dict__, **overrides})
    repo = await PostgresRepository(TEST_DATABASE_URL, schema=settings.db_schema).open()
    await repo.ensure_schema(reset=True)
    client, backend = create_redis_client(TEST_REDIS_URL or None)
    if TEST_REDIS_URL:
        await client.flushdb()
    store = RedisStore(
        client, backend=backend, session_ttl_s=settings.session_ttl_s, pending_ttl_s=settings.pending_action_ttl_s
    )
    policy_store, embedding, _ = _policy_store(settings)
    return assemble(settings, clock or FixedClock(), repo, store, policy_store, embedding)


async def drop_runtime(runtime: Runtime) -> None:
    repo = runtime.repository
    if isinstance(repo, PostgresRepository):

        def drop() -> None:
            with repo._pool.connection() as conn:
                conn.execute(f'DROP SCHEMA IF EXISTS "{repo.schema}" CASCADE')

        import asyncio

        await asyncio.to_thread(drop)
    await runtime.aclose()


async def make_runtime(backend: str = "memory", **overrides: Any) -> Runtime:
    if backend == "postgres":
        return await postgres_runtime(**overrides)
    return build_memory_runtime(**overrides)


def boundary_for(runtime: Runtime, planner: ScriptedPlanner | None = None) -> SessionBoundary:
    planner = planner if planner is not None else ScriptedPlanner()
    return SessionBoundary(dialogue_factory=lambda: planner, runtime=runtime)


def actions(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [row["action_name"] for row in rows]


def is_memory(runtime: Runtime) -> bool:
    return isinstance(runtime.repository, MemoryRepository)


__all__ = [
    "BACKENDS",
    "TEST_DATABASE_URL",
    "TEST_REDIS_URL",
    "P",
    "ScriptedPlanner",
    "actions",
    "boundary_for",
    "drop_runtime",
    "is_memory",
    "make_runtime",
    "postgres_runtime",
]
