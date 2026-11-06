"""Relational store: the ``Repository`` interface, its PostgreSQL implementation, and an in-memory twin.

PostgreSQL is the store of record (guide §3.1, §4.1). The in-memory repository
serves the same seed rows through the same interface; it backs the fast unit
tests and the no-infrastructure quick start (``VOXCARE_DATA_BACKEND=memory``),
and the acceptance suite runs against both.

The PostgreSQL implementation uses psycopg's synchronous pool behind
``asyncio.to_thread``: psycopg's async mode cannot run on Windows'
default Proactor event loop, and this keeps one code path on every OS.
Every read appends the table name to ``read_log`` (the "DB spy" of Scenario 2).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from importlib.resources import files
from typing import Any, Protocol

from ..domain.clock import DEMO_TIMEZONE
from . import seed

logger = logging.getLogger("voxcare.storage")

DEFAULT_SCHEMA = "voxcare"
STATEMENT_TIMEOUT_MS = 1500  # implementation choice: reuse the 1500 ms tool timeout (guide §3.2)
# Session time zone. "GMT" (offset 0, no DST, the same instants as UTC) is built into PostgreSQL, so it also
# works on servers shipped without the IANA time-zone files (the pgserver Windows wheel has none, and
# rejects "UTC" at connect time). Every stored and returned value is timezone-aware either way.
SESSION_TIME_ZONE = "GMT"


class Repository(Protocol):
    backend: str
    read_log: list[str]

    async def ping(self) -> bool: ...
    async def get_customer(self, customer_id: str) -> dict[str, Any] | None: ...
    async def find_customer_by_phone(self, phone_e164: str) -> dict[str, Any] | None: ...
    async def customer_names(self) -> list[str]: ...
    async def get_order_by_code(self, order_code: str) -> dict[str, Any] | None: ...
    async def get_order_by_id(self, order_id: str) -> dict[str, Any] | None: ...
    async def list_orders_by_customer(self, customer_id: str) -> list[dict[str, Any]]: ...
    async def get_order_items(self, order_id: str) -> list[dict[str, Any]]: ...
    async def get_shipment_by_order(self, order_id: str) -> dict[str, Any] | None: ...
    async def get_shipment_by_tracking(self, tracking_code: str) -> dict[str, Any] | None: ...
    async def get_shipment_events(self, shipment_id: str) -> list[dict[str, Any]]: ...
    async def get_installations(self, order_id: str) -> list[dict[str, Any]]: ...
    async def search_products(
        self, category: str, brand: str | None, max_price_vnd: int | None, keyword: str | None
    ) -> list[dict[str, Any]]: ...
    async def get_products_by_skus(self, skus: Sequence[str]) -> list[dict[str, Any]]: ...
    async def get_promotion(self, voucher_code: str) -> dict[str, Any] | None: ...
    async def insert_audit(self, row: Mapping[str, Any]) -> None: ...
    async def list_audit(self, session_id: str | None = None) -> list[dict[str, Any]]: ...
    async def insert_agent_activity(self, row: Mapping[str, Any]) -> None: ...
    async def list_agent_activity(self, session_id: str | None = None) -> list[dict[str, Any]]: ...
    async def update_order_occ(
        self, order_code: str, expected_version: int, changes: Mapping[str, Any], now: datetime
    ) -> int: ...
    async def set_stock(self, sku: str, quantity: int) -> None: ...
    async def close(self) -> None: ...


def delivered_on(events: Iterable[Mapping[str, Any]]) -> date | None:
    """Local delivery date from the DELIVERED shipment event (the validated delivery date of POL-RET-001)."""
    for event in events:
        if event.get("event_type") == "DELIVERED":
            when = event["event_time"]
            return when.astimezone(DEMO_TIMEZONE).date() if isinstance(when, datetime) else date.fromisoformat(when)
    return None


def plain(value: Any) -> Any:
    """JSON-friendly copy of a row value: Decimal → int/float, UUID → str (datetimes stay datetimes)."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


_ORDER_COLUMNS = {
    "status",
    "payment_status",
    "requested_delivery_time",
    "requested_delivery_window_end",
    "shipping_address",
}


# --------------------------------------------------------------------------- in-memory


class MemoryRepository:
    """Seed rows in dictionaries. Same interface and results as :class:`PostgresRepository`."""

    backend = "memory"

    def __init__(self) -> None:
        self.read_log: list[str] = []
        self.tables: dict[str, list[dict[str, Any]]] = {name: copy.deepcopy(rows) for name, rows in seed.TABLES}
        self.tables["audit_log"] = []
        self.tables["agent_activity"] = []
        self.fail_with: BaseException | None = None  # tests: raise this from every call
        self.delay_s: float = 0.0  # tests: simulate a slow store

    async def _touch(self, table: str) -> None:
        self.read_log.append(table)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail_with is not None:
            raise self.fail_with

    def _rows(self, table: str) -> list[dict[str, Any]]:
        return self.tables[table]

    @staticmethod
    def _one(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
        for row in rows:
            return copy.deepcopy(row)
        return None

    async def ping(self) -> bool:
        return self.fail_with is None

    async def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        await self._touch("customer")
        return self._one(row for row in self._rows("customer") if row["customer_id"] == customer_id)

    async def find_customer_by_phone(self, phone_e164: str) -> dict[str, Any] | None:
        await self._touch("customer")
        return self._one(row for row in self._rows("customer") if row["phone_number"] == phone_e164)

    async def customer_names(self) -> list[str]:
        await self._touch("customer")
        return [row["full_name"] for row in self._rows("customer")]

    async def get_order_by_code(self, order_code: str) -> dict[str, Any] | None:
        await self._touch("orders")
        return self._one(row for row in self._rows("orders") if row["order_code"] == order_code)

    async def get_order_by_id(self, order_id: str) -> dict[str, Any] | None:
        await self._touch("orders")
        return self._one(row for row in self._rows("orders") if row["order_id"] == order_id)

    async def list_orders_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        await self._touch("orders")
        rows = [copy.deepcopy(row) for row in self._rows("orders") if row["customer_id"] == customer_id]
        return sorted(rows, key=lambda row: row["created_at"])

    async def get_order_items(self, order_id: str) -> list[dict[str, Any]]:
        await self._touch("order_item")
        rows = [copy.deepcopy(row) for row in self._rows("order_item") if row["order_id"] == order_id]
        return sorted(rows, key=lambda row: row["item_id"])

    async def get_shipment_by_order(self, order_id: str) -> dict[str, Any] | None:
        await self._touch("shipment")
        return self._one(row for row in self._rows("shipment") if row["order_id"] == order_id)

    async def get_shipment_by_tracking(self, tracking_code: str) -> dict[str, Any] | None:
        await self._touch("shipment")
        return self._one(row for row in self._rows("shipment") if row["tracking_code"] == tracking_code)

    async def get_shipment_events(self, shipment_id: str) -> list[dict[str, Any]]:
        await self._touch("shipment_event")
        rows = [copy.deepcopy(row) for row in self._rows("shipment_event") if row["shipment_id"] == shipment_id]
        return sorted(rows, key=lambda row: row["event_time"])

    async def get_installations(self, order_id: str) -> list[dict[str, Any]]:
        await self._touch("installation")
        rows = [copy.deepcopy(row) for row in self._rows("installation") if row["order_id"] == order_id]
        return sorted(rows, key=lambda row: row["installation_id"])

    def _joined(self) -> list[dict[str, Any]]:
        products = {row["product_id"]: row for row in self._rows("product")}
        joined = []
        for variant in self._rows("product_variant"):
            product = products[variant["product_id"]]
            joined.append(copy.deepcopy({**product, **variant}))
        return sorted(joined, key=lambda row: row["sku"])

    async def search_products(
        self, category: str, brand: str | None, max_price_vnd: int | None, keyword: str | None
    ) -> list[dict[str, Any]]:
        await self._touch("product_variant")
        result = []
        for row in self._joined():
            if row["category"] != category:
                continue
            if brand and row["brand"].casefold() != brand.casefold():
                continue
            if max_price_vnd is not None and row["price"] > max_price_vnd:
                continue
            if keyword and not _keyword_matches(keyword, row):
                continue
            result.append(row)
        return result

    async def get_products_by_skus(self, skus: Sequence[str]) -> list[dict[str, Any]]:
        await self._touch("product_variant")
        wanted = set(skus)
        return [row for row in self._joined() if row["sku"] in wanted]

    async def get_promotion(self, voucher_code: str) -> dict[str, Any] | None:
        await self._touch("promotion")
        return self._one(row for row in self._rows("promotion") if row["voucher_code"] == voucher_code)

    async def insert_audit(self, row: Mapping[str, Any]) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self._rows("audit_log").append(copy.deepcopy(dict(row)))

    async def list_audit(self, session_id: str | None = None) -> list[dict[str, Any]]:
        rows = [copy.deepcopy(row) for row in self._rows("audit_log")]
        return [row for row in rows if session_id is None or row["session_id"] == session_id]

    async def insert_agent_activity(self, row: Mapping[str, Any]) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self._rows("agent_activity").append(copy.deepcopy(dict(row)))

    async def list_agent_activity(self, session_id: str | None = None) -> list[dict[str, Any]]:
        rows = [copy.deepcopy(row) for row in self._rows("agent_activity")]
        return [row for row in rows if session_id is None or row["session_id"] == session_id]

    async def update_order_occ(
        self, order_code: str, expected_version: int, changes: Mapping[str, Any], now: datetime
    ) -> int:
        unknown = set(changes) - _ORDER_COLUMNS
        if unknown:
            raise ValueError(f"not an updatable order column: {sorted(unknown)}")
        for row in self._rows("orders"):
            if row["order_code"] == order_code and row["version"] == expected_version:
                row.update(changes)
                row["version"] += 1
                row["updated_at"] = now
                return 1
        return 0

    async def set_stock(self, sku: str, quantity: int) -> None:
        for row in self._rows("product_variant"):
            if row["sku"] == sku:
                row["stock_quantity"] = quantity

    async def close(self) -> None:
        return None


def _keyword_matches(keyword: str, row: Mapping[str, Any]) -> bool:
    """Every keyword token must appear in the SKU, name, brand or screen size ("55 inch" matches 55-inch models)."""
    haystack = " ".join(str(row.get(key) or "") for key in ("sku", "product_name", "brand", "screen_size")).lower()
    haystack = haystack.replace("-", " ")
    tokens = [token for token in keyword.lower().replace("-", " ").replace('"', " inch ").split() if token]
    return all(token in haystack or (token == "inches" and "inch" in haystack) for token in tokens)  # noqa: S105


# --------------------------------------------------------------------------- PostgreSQL


class PostgresRepository:
    """psycopg 3 against the guide §4.1 schema (``schema.sql``) in a dedicated PostgreSQL schema."""

    backend = "postgres"

    def __init__(self, dsn: str, *, schema: str = DEFAULT_SCHEMA, min_size: int = 1, max_size: int = 8) -> None:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        if not schema.isidentifier():
            raise ValueError("schema must be a plain identifier")
        self.schema = schema
        self.dsn = dsn
        self.read_log: list[str] = []
        self._pool = ConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={
                "row_factory": dict_row,
                "autocommit": True,
                "options": (
                    f"-c search_path={schema} -c statement_timeout={STATEMENT_TIMEOUT_MS} "
                    f"-c timezone={SESSION_TIME_ZONE}"
                ),
            },
            open=False,
            name="voxcare",
        )

    async def open(self) -> PostgresRepository:
        await asyncio.to_thread(self._pool.open, True, 10.0)
        return self

    async def close(self) -> None:
        await asyncio.to_thread(self._pool.close)

    # -- schema management --

    def _ensure_schema_sync(self, *, reset: bool) -> bool:
        """Create the schema and seed it when missing (or when ``reset``). Returns True when it seeded."""
        with self._pool.connection() as conn:
            if reset:
                conn.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
            exists = conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = 'customer'",
                (self.schema,),
            ).fetchone()
            if exists:
                return False
            ddl = files("voxcare.storage").joinpath("schema.sql").read_text(encoding="utf-8")
            with conn.transaction():
                conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
                conn.execute(f'SET LOCAL search_path TO "{self.schema}"')
                conn.execute(ddl)
                for table, rows in seed.TABLES:
                    for row in rows:
                        columns = list(row)
                        placeholders = ", ".join(["%s"] * len(columns))
                        values = [_db_value(row[column]) for column in columns]
                        conn.execute(
                            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",  # noqa: S608
                            values,
                        )
            return True

    async def ensure_schema(self, *, reset: bool = False) -> bool:
        seeded = await asyncio.to_thread(self._ensure_schema_sync, reset=reset)
        if seeded:
            logger.info("PostgreSQL schema %s created and seeded", self.schema)
        return seeded

    # -- helpers --

    def _query_sync(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            return [plain(dict(row)) for row in conn.execute(sql, params).fetchall()]

    async def _query(self, table: str, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        self.read_log.append(table)
        return await asyncio.to_thread(self._query_sync, sql, params)

    async def _first(self, table: str, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = await self._query(table, sql, params)
        return rows[0] if rows else None

    def _execute_sync(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._pool.connection() as conn:
            return conn.execute(sql, params).rowcount

    # -- reads --

    async def ping(self) -> bool:
        rows = await asyncio.to_thread(self._query_sync, "SELECT 1 AS ok")
        return bool(rows and rows[0]["ok"] == 1)

    async def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        return await self._first("customer", "SELECT * FROM customer WHERE customer_id = %s", (customer_id,))

    async def find_customer_by_phone(self, phone_e164: str) -> dict[str, Any] | None:
        return await self._first("customer", "SELECT * FROM customer WHERE phone_number = %s", (phone_e164,))

    async def customer_names(self) -> list[str]:
        return [row["full_name"] for row in await self._query("customer", "SELECT full_name FROM customer")]

    async def get_order_by_code(self, order_code: str) -> dict[str, Any] | None:
        return await self._first("orders", "SELECT * FROM orders WHERE order_code = %s", (order_code,))

    async def get_order_by_id(self, order_id: str) -> dict[str, Any] | None:
        return await self._first("orders", "SELECT * FROM orders WHERE order_id = %s", (order_id,))

    async def list_orders_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        return await self._query(
            "orders", "SELECT * FROM orders WHERE customer_id = %s ORDER BY created_at", (customer_id,)
        )

    async def get_order_items(self, order_id: str) -> list[dict[str, Any]]:
        return await self._query(
            "order_item", "SELECT * FROM order_item WHERE order_id = %s ORDER BY item_id", (order_id,)
        )

    async def get_shipment_by_order(self, order_id: str) -> dict[str, Any] | None:
        return await self._first("shipment", "SELECT * FROM shipment WHERE order_id = %s", (order_id,))

    async def get_shipment_by_tracking(self, tracking_code: str) -> dict[str, Any] | None:
        return await self._first("shipment", "SELECT * FROM shipment WHERE tracking_code = %s", (tracking_code,))

    async def get_shipment_events(self, shipment_id: str) -> list[dict[str, Any]]:
        return await self._query(
            "shipment_event",
            "SELECT * FROM shipment_event WHERE shipment_id = %s ORDER BY event_time",
            (shipment_id,),
        )

    async def get_installations(self, order_id: str) -> list[dict[str, Any]]:
        return await self._query(
            "installation",
            "SELECT * FROM installation WHERE order_id = %s ORDER BY installation_id",
            (order_id,),
        )

    _PRODUCT_SELECT = (
        "SELECT p.product_id, p.product_name, p.category, p.brand, p.base_price, p.warranty_months, "
        "p.specifications_json, v.variant_id, v.sku, v.color, v.storage_capacity, v.screen_size, v.price, "
        "v.stock_quantity FROM product_variant v JOIN product p USING (product_id)"
    )

    async def search_products(
        self, category: str, brand: str | None, max_price_vnd: int | None, keyword: str | None
    ) -> list[dict[str, Any]]:
        sql = self._PRODUCT_SELECT + " WHERE p.category = %s"
        params: list[Any] = [category]
        if brand:
            sql += " AND lower(p.brand) = lower(%s)"
            params.append(brand)
        if max_price_vnd is not None:
            sql += " AND v.price <= %s"
            params.append(max_price_vnd)
        rows = await self._query("product_variant", sql + " ORDER BY v.sku", params)
        return [row for row in rows if not keyword or _keyword_matches(keyword, row)]

    async def get_products_by_skus(self, skus: Sequence[str]) -> list[dict[str, Any]]:
        return await self._query(
            "product_variant", self._PRODUCT_SELECT + " WHERE v.sku = ANY(%s) ORDER BY v.sku", (list(skus),)
        )

    async def get_promotion(self, voucher_code: str) -> dict[str, Any] | None:
        return await self._first("promotion", "SELECT * FROM promotion WHERE voucher_code = %s", (voucher_code,))

    # -- appends --

    async def insert_audit(self, row: Mapping[str, Any]) -> None:
        await asyncio.to_thread(
            self._execute_sync,
            "INSERT INTO audit_log (audit_id, session_id, order_id, mutation_idempotency_key, actor, action_name, "
            "action_status, outcome_status, request_payload_json, response_payload_json, timestamp) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                row["audit_id"],
                row["session_id"],
                row["order_id"],
                row["mutation_idempotency_key"],
                row["actor"],
                row["action_name"],
                row["action_status"],
                row["outcome_status"],
                _json(row.get("request_payload_json")),
                _json(row.get("response_payload_json")),
                row["timestamp"],
            ),
        )

    async def list_audit(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id is None:
            return await asyncio.to_thread(self._query_sync, "SELECT * FROM audit_log ORDER BY timestamp, audit_id")
        return await asyncio.to_thread(
            self._query_sync, "SELECT * FROM audit_log WHERE session_id = %s ORDER BY timestamp", (session_id,)
        )

    async def insert_agent_activity(self, row: Mapping[str, Any]) -> None:
        await asyncio.to_thread(
            self._execute_sync,
            "INSERT INTO agent_activity (activity_id, session_id, customer_id, order_id, activity_code, outcome_code, "
            "activity_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                row["activity_id"],
                row["session_id"],
                row["customer_id"],
                row["order_id"],
                row["activity_code"],
                row["outcome_code"],
                row["activity_at"],
            ),
        )

    async def list_agent_activity(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id is None:
            return await asyncio.to_thread(self._query_sync, "SELECT * FROM agent_activity ORDER BY activity_at")
        return await asyncio.to_thread(
            self._query_sync, "SELECT * FROM agent_activity WHERE session_id = %s ORDER BY activity_at", (session_id,)
        )

    # -- OCC write (fake OMS only; never reachable from a Phase 1 tool) --

    async def update_order_occ(
        self, order_code: str, expected_version: int, changes: Mapping[str, Any], now: datetime
    ) -> int:
        unknown = set(changes) - _ORDER_COLUMNS
        if unknown:
            raise ValueError(f"not an updatable order column: {sorted(unknown)}")
        assignments = ", ".join(f"{column} = %s" for column in changes)
        sql = (
            f"UPDATE orders SET {assignments}{', ' if assignments else ''}version = version + 1, updated_at = %s "  # noqa: S608
            "WHERE order_code = %s AND version = %s"
        )
        params = [*changes.values(), now, order_code, expected_version]
        return await asyncio.to_thread(self._execute_sync, sql, params)

    async def set_stock(self, sku: str, quantity: int) -> None:
        await asyncio.to_thread(
            self._execute_sync, "UPDATE product_variant SET stock_quantity = %s WHERE sku = %s", (quantity, sku)
        )


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, default=str)


def _db_value(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    return value


def utc_now_from(clock_now: datetime) -> datetime:
    return clock_now.astimezone(UTC)


__all__ = [
    "DEFAULT_SCHEMA",
    "MemoryRepository",
    "PostgresRepository",
    "Repository",
    "delivered_on",
    "plain",
]
