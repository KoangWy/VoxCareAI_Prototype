"""Runtime assembly: stores, services, middleware and the dialogue manager for one process.

``build_runtime(settings)`` wires the design's tiers (guide §2.1):

- PostgreSQL (``DATABASE_URL``; without it an embedded ``pgserver`` cluster in ``data/pg``
  when that optional package is installed) or the in-memory twin (``VOXCARE_DATA_BACKEND=memory``);
- Redis (``REDIS_URL``; without it an in-process fakeredis server);
- Chroma ``voxcom_policy_v1`` (persistent at ``CHROMA_PATH``, default ``data/chroma``).

The policy vector index is cached per process and embedding model, so tests that
build many runtimes index the corpus once.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DataSettings, load_data_settings
from .domain.clock import Clock, FixedClock, SystemClock
from .domain.services.catalog import CatalogService, PromotionService
from .domain.services.handoff import HandoffService
from .domain.services.orders import CustomerOrdersService, OrderService, ShipmentService
from .domain.services.policy import PolicyService
from .domain.services.writes import WriteService
from .middleware.throttle import CatalogThrottle
from .middleware.tool_middleware import ToolMiddleware
from .observability.audit import AuditWriter
from .security import pii
from .security.tokens import TokenService
from .session.dialogue import DialogueManager, Flags
from .storage import seed
from .storage.db import MemoryRepository, PostgresRepository, Repository
from .storage.embeddings import CALIBRATED_THRESHOLDS, create_embedder
from .storage.fake_oms import FakeVoxComOMS
from .storage.redis_store import RedisStore, create_redis_client
from .storage.vector_store import PolicyVectorStore, embedding_text, load_corpus

logger = logging.getLogger("voxcare.runtime")
APP_ROOT = Path(__file__).resolve().parents[2]
_VECTOR_STORES: dict[tuple[str, str | None], PolicyVectorStore] = {}


@dataclass
class Runtime:
    settings: DataSettings
    clock: Clock
    repository: Repository
    store: RedisStore
    policy_store: PolicyVectorStore | None
    embedding: str
    threshold: float
    audit: AuditWriter
    middleware: ToolMiddleware
    writes: WriteService
    dialogue: DialogueManager
    tokens: TokenService
    oms: FakeVoxComOMS
    pg_server: Any = None
    notes: list[str] = field(default_factory=list)

    async def aclose(self) -> None:
        await self.repository.close()
        await self.store.close()
        stop = getattr(self.pg_server, "stop", None)
        if callable(stop):  # the embedded PostgreSQL, stopped cleanly with the app (never by Ctrl+C)
            await asyncio.to_thread(stop)

    def describe(self) -> dict[str, Any]:
        return {
            "relational_store": self.repository.backend,
            "session_store": self.store.backend,
            "vector_store": "chroma",
            "chroma_path": self.settings.chroma_path,
            "policy_embedding": self.embedding,
            "rag_threshold": self.threshold,
            "rag_top_k": self.settings.rag_top_k,
            "clock": self.settings.clock,
            "anchor_now": self.clock.now().isoformat(),
            "flags": {
                "FR-AUTH-01": self.settings.fr_auth_01,
                "FR-AUTH-02": self.settings.fr_auth_02,
                "FR-SEC-01": self.settings.fr_sec_01,
            },
            "write_mode": self.settings.write_mode,
            "notes": list(self.notes),
        }

    async def health(self) -> dict[str, Any]:
        """Component checks for ``/healthz`` (each bounded so the probe answers within 2 s)."""

        async def check(name: str, probe: Any) -> tuple[str, dict[str, Any]]:
            try:
                ok = bool(await asyncio.wait_for(probe(), timeout=1.5))
                return name, {"ok": ok}
            except Exception as exc:
                return name, {"ok": False, "error": type(exc).__name__}

        async def chroma() -> bool:
            if self.policy_store is None:
                return False
            return await asyncio.to_thread(self.policy_store.count) == len(self.policy_store.clauses)

        results = await asyncio.gather(
            check("postgres" if self.repository.backend == "postgres" else "relational_memory", self.repository.ping),
            check(self.store.backend, self.store.ping),
            check("chroma", chroma),
        )
        return dict(results)


def _policy_store(settings: DataSettings) -> tuple[PolicyVectorStore | None, str, list[str]]:
    notes: list[str] = []
    path = settings.chroma_path
    if path and not Path(path).is_absolute():
        path = str(APP_ROOT / path)
    key = (settings.policy_embedding, path)
    if key in _VECTOR_STORES:
        return _VECTOR_STORES[key], settings.policy_embedding, notes
    corpus = load_corpus()
    embedder = create_embedder(settings.policy_embedding, [embedding_text(clause) for clause in corpus["clauses"]])
    store = PolicyVectorStore(embedder, path=path, corpus=corpus).build()
    try:
        # Load the embedding model now: when the collection is reused, the first caller question would otherwise
        # pay the model load and hit the 1.5 s tool timeout (log SES-345e37a0, turn 16: retried after a timeout).
        embedder.embed(["warm-up"])
    except Exception as exc:  # a failing embedder surfaces on the first query and in /healthz
        notes.append(f"policy embedder warm-up failed: {exc}")
    _VECTOR_STORES[key] = store
    return store, settings.policy_embedding, notes


def _start_embedded_postgres() -> tuple[str, Any]:
    from .storage.embedded_pg import EmbeddedPostgres

    server = EmbeddedPostgres(data_dir=APP_ROOT / "data" / "pg", log_file=APP_ROOT / "data" / "pg.log")
    return server.start(), server


async def build_runtime(settings: DataSettings | None = None, *, clock: Clock | None = None) -> Runtime:
    settings = settings or load_data_settings()
    clock = clock or (FixedClock() if settings.clock == "demo" else SystemClock())
    notes: list[str] = []
    pg_server = None
    repository: Repository
    if settings.backend == "memory":
        repository = MemoryRepository()
        notes.append("VOXCARE_DATA_BACKEND=memory: seed rows in process memory instead of PostgreSQL")
    else:
        dsn = settings.database_url
        if not dsn:
            dsn, pg_server = await asyncio.to_thread(_start_embedded_postgres)
            notes.append("embedded PostgreSQL (pgserver) in data/pg")
        postgres = PostgresRepository(dsn, schema=settings.db_schema)
        await postgres.open()
        await postgres.ensure_schema(reset=settings.db_reset_on_start)
        repository = postgres
    client, redis_backend = create_redis_client(settings.redis_url)
    store = RedisStore(
        client,
        backend=redis_backend,
        session_ttl_s=settings.session_ttl_s,
        pending_ttl_s=settings.pending_action_ttl_s,
    )
    if redis_backend == "fakeredis":
        notes.append("REDIS_URL is not set: in-process fakeredis server (same commands, not shared across processes)")
    policy_store, embedding, policy_notes = await asyncio.to_thread(_policy_store, settings)
    notes.extend(policy_notes)
    return assemble(settings, clock, repository, store, policy_store, embedding, notes=notes, pg_server=pg_server)


def assemble(
    settings: DataSettings,
    clock: Clock,
    repository: Repository,
    store: RedisStore,
    policy_store: PolicyVectorStore | None,
    embedding: str,
    *,
    notes: list[str] | None = None,
    pg_server: Any = None,
) -> Runtime:
    threshold = settings.rag_threshold if settings.rag_threshold is not None else CALIBRATED_THRESHOLDS[embedding]
    timeout_s = settings.tool_timeout_ms / 1000
    audit = AuditWriter(repository, clock)
    pii.masker.learn(
        names=[row["full_name"] for row in seed.CUSTOMERS] + [row["driver_name"] for row in seed.SHIPMENTS],
        addresses=[row["default_address"] for row in seed.CUSTOMERS] + [row["shipping_address"] for row in seed.ORDERS],
    )
    middleware = ToolMiddleware(
        orders=OrderService(repository, timeout_s=timeout_s),
        shipments=ShipmentService(repository, timeout_s=timeout_s),
        customer_orders=CustomerOrdersService(repository, timeout_s=timeout_s),
        policy=PolicyService(policy_store, threshold=threshold, top_k=settings.rag_top_k, timeout_s=timeout_s),
        catalog=CatalogService(repository, timeout_s=timeout_s),
        promotions=PromotionService(repository, timeout_s=timeout_s),
        handoff=HandoffService(repository, store, audit, clock),
        throttle=CatalogThrottle(store, enabled=settings.fr_sec_01),
        audit=audit,
        clock=clock,
    )
    oms = FakeVoxComOMS(repository, clock.now)
    writes = WriteService(repository, store, audit, clock, oms if settings.write_mode == "fake_oms" else None)
    dialogue = DialogueManager(
        middleware=middleware,
        writes=writes,
        repository=repository,
        clock=clock,
        flags=Flags(
            fr_auth_01=settings.fr_auth_01,
            fr_auth_02=settings.fr_auth_02,
            fr_sec_01=settings.fr_sec_01,
            write_mode=settings.write_mode,
        ),
    )
    return Runtime(
        settings=settings,
        clock=clock,
        repository=repository,
        store=store,
        policy_store=policy_store,
        embedding=embedding,
        threshold=threshold,
        audit=audit,
        middleware=middleware,
        writes=writes,
        dialogue=dialogue,
        tokens=TokenService(ttl_s=settings.session_ttl_s),
        oms=oms,
        pg_server=pg_server,
        notes=list(notes or []),
    )


def build_memory_runtime(*, clock: Clock | None = None, **overrides: Any) -> Runtime:
    """Synchronous in-memory runtime for unit tests: MemoryRepository, fakeredis, hashing embedding."""
    env = {"VOXCARE_DATA_BACKEND": "memory", "VOXCARE_POLICY_EMBEDDING": "hashing", "CHROMA_PATH": ""}
    base = load_data_settings(env)
    settings = DataSettings(**{**base.__dict__, **overrides})
    client, backend = create_redis_client(None)
    store = RedisStore(
        client, backend=backend, session_ttl_s=settings.session_ttl_s, pending_ttl_s=settings.pending_action_ttl_s
    )
    policy_store, embedding, _ = _policy_store(settings)
    return assemble(settings, clock or FixedClock(), MemoryRepository(), store, policy_store, embedding)


__all__ = ["APP_ROOT", "Runtime", "assemble", "build_memory_runtime", "build_runtime"]
