"""Stores: PostgreSQL schema/seed parity with the in-memory twin, append-only audit, OCC; Redis keys and TTLs."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest
from support import TEST_DATABASE_URL, TEST_REDIS_URL

from voxcare.storage import seed
from voxcare.storage.db import MemoryRepository, PostgresRepository, delivered_on
from voxcare.storage.redis_store import DuplicateIdempotencyKey, RedisStore, create_redis_client

needs_pg = pytest.mark.skipif(not TEST_DATABASE_URL, reason="VOXCARE_TEST_DATABASE_URL unset")


@pytest.fixture
async def pg():
    repo = await PostgresRepository(TEST_DATABASE_URL, schema=f"t_{uuid.uuid4().hex[:12]}").open()
    assert await repo.ensure_schema(reset=True) is True
    yield repo

    def drop() -> None:
        with repo._pool.connection() as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{repo.schema}" CASCADE')

    await asyncio.to_thread(drop)
    await repo.close()


def _strip(rows):
    return [{k: v for k, v in row.items() if k not in {"created_at", "updated_at"}} for row in rows]


@needs_pg
async def test_postgres_serves_the_same_rows_as_memory(pg):
    memory = MemoryRepository()
    for code in ("ORD-1001", "ORD-1002", "ORD-1003", "ORD-2002", "ORD-9999"):
        a, b = await pg.get_order_by_code(code), await memory.get_order_by_code(code)
        assert _strip([a] if a else []) == _strip([b] if b else [])
    order = await pg.get_order_by_code("ORD-1002")
    assert order["order_id"] == seed.order_id_for("ORD-1002") and len(order["order_id"]) == 32
    assert order["total_amount"] == 18_990_000 and isinstance(order["total_amount"], int)
    assert order["requested_delivery_time"] == datetime(2026, 10, 16, 2, tzinfo=UTC)  # 09:00 +07:00
    assert await pg.get_order_items(order["order_id"]) == await memory.get_order_items(order["order_id"])
    shipment = await pg.get_shipment_by_tracking("GHN-VN-882910")
    assert shipment == await memory.get_shipment_by_tracking("GHN-VN-882910")
    events = await pg.get_shipment_events(shipment["shipment_id"])
    assert [e["event_id"] for e in events] == ["EVT-1002-1", "EVT-1002-2"]
    ship_1003 = await pg.get_shipment_by_order(seed.order_id_for("ORD-1003"))
    assert delivered_on(await pg.get_shipment_events(ship_1003["shipment_id"])).isoformat() == "2026-10-02"
    for args in (
        ("TV_AV", None, None, "55 inch"),
        ("TV_AV", "lg", None, None),
        ("TV_AV", None, 12_000_000, None),
        ("KITCHEN", None, None, None),
    ):
        assert [r["sku"] for r in await pg.search_products(*args)] == [
            r["sku"] for r in await memory.search_products(*args)
        ]
    assert (await pg.get_promotion("VOXCOM_500K"))["discount_cap"] == 500_000
    assert (await pg.find_customer_by_phone("+84900000001"))["customer_id"] == "CUST-001"
    assert sorted(await pg.customer_names()) == sorted(await memory.customer_names())
    assert await pg.ping()


@needs_pg
async def test_audit_log_is_append_only(pg):
    audit_id = str(uuid.uuid4())
    await pg.insert_audit(
        {
            "audit_id": audit_id,
            "session_id": "S",
            "order_id": None,
            "mutation_idempotency_key": None,
            "actor": "Bot",
            "action_name": "TOOL_INVOCATION",
            "action_status": "Success",
            "outcome_status": "SUCCESS",
            "request_payload_json": {"tool_name": "x"},
            "response_payload_json": {},
            "timestamp": datetime.now(UTC),
        }
    )

    def tamper() -> tuple[int, int]:
        with pg._pool.connection() as conn:
            updated = conn.execute(
                "UPDATE audit_log SET outcome_status = 'FAILED' WHERE audit_id = %s", (audit_id,)
            ).rowcount
            deleted = conn.execute("DELETE FROM audit_log WHERE audit_id = %s", (audit_id,)).rowcount
            return updated, deleted

    assert await asyncio.to_thread(tamper) == (0, 0)
    rows = await pg.list_audit("S")
    assert len(rows) == 1 and rows[0]["outcome_status"] == "SUCCESS"


@needs_pg
async def test_occ_update_requires_the_expected_version(pg):
    now = datetime.now(UTC)
    assert await pg.update_order_occ("ORD-1001", 2, {"status": "cancelled"}, now) == 0  # stale version: 0 rows
    assert await pg.update_order_occ("ORD-1001", 1, {"status": "cancelled"}, now) == 1
    order = await pg.get_order_by_code("ORD-1001")
    assert (order["status"], order["version"]) == ("cancelled", 2)
    with pytest.raises(ValueError):
        await pg.update_order_occ("ORD-1001", 2, {"customer_id": "CUST-002"}, now)


@needs_pg
async def test_schema_is_created_once_and_check_constraints_hold(pg):
    assert await pg.ensure_schema() is False  # already seeded

    def bad_carrier() -> None:
        with pg._pool.connection() as conn:
            conn.execute("UPDATE shipment SET carrier_name = 'DHL' WHERE shipment_id = 'SHIP-1002'")

    with pytest.raises(Exception, match="check"):
        await asyncio.to_thread(bad_carrier)


@pytest.fixture(
    params=[
        "fakeredis",
        pytest.param("redis", marks=pytest.mark.skipif(not TEST_REDIS_URL, reason="VOXCARE_TEST_REDIS_URL unset")),
    ]
)
async def store(request):
    client, backend = create_redis_client(TEST_REDIS_URL if request.param == "redis" else None)
    if backend == "redis":
        await client.flushdb()
    yield RedisStore(client, backend=backend)
    await client.aclose()


async def test_session_and_pending_action_ttls(store):
    await store.put_session({"session_id": "SES-T", "customer_id": "CUST-001"})
    assert 1795 <= await store.client.ttl("session:SES-T") <= 1800
    action = {"action_id": "ACT-1", "idempotency_key": "k-1", "status": "pending"}
    await store.create_pending_action(action)
    assert 295 <= await store.client.ttl("pending_action:ACT-1") <= 300
    assert await store.action_for_key("k-1") == "ACT-1"
    with pytest.raises(DuplicateIdempotencyKey):  # a key is unique
        await store.create_pending_action({"action_id": "ACT-2", "idempotency_key": "k-1", "status": "pending"})
    await store.client.expire("pending_action:ACT-1", 100)
    await store.save_pending_action({**action, "status": "submitted"})
    assert await store.client.ttl("pending_action:ACT-1") <= 100  # the confirmation window never restarts


async def test_handoff_packet_expires_with_the_session(store):
    await store.put_session({"session_id": "SES-H"})
    await store.client.expire("session:SES-H", 600)
    await store.put_handoff({"handoff_id": "HND-1", "session_id": "SES-H", "context_payload_json": {}})
    assert 590 <= await store.client.ttl("handoff:HND-1") <= 600
    assert [p["handoff_id"] for p in await store.list_handoffs("SES-H")] == ["HND-1"]


async def test_throttle_sorted_set(store):
    times = iter([1000.0, 1001.0, 1200.0, 1200.0])
    store.now_s = lambda: next(times)
    assert await store.throttle_record("SES-Z", {"A", "B"}) == 2
    assert await store.throttle_record("SES-Z", {"C"}) == 3
    assert await store.throttle_window("SES-Z") == set()  # 120 s later the window is empty
    assert await store.client.zcard("throttle:SES-Z") == 0
