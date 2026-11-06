# Data layer

Three stores, as in G§3.1 / G§4. The tools never touch them directly: each tool calls one domain service, and every service reads through the `Repository` protocol (`storage/db.py`), the `RedisStore` (`storage/redis_store.py`) or the `PolicyVectorStore` (`storage/vector_store.py`). `runtime.py` builds the three from `.env` and `/healthz` checks each.

## PostgreSQL — system of record

- **Schema:** `storage/schema.sql` (DDL G§4.1): `customer`, `product`, `product_variant`, `orders`, `order_item`, `shipment`, `shipment_event`, `installation`, `warranty_rma`, `promotion`, `agent_activity`, `audit_log`. All tables live in a dedicated schema (`VOXCARE_DB_SCHEMA`, default `voxcare`), so the database can be shared.
- **Fixtures:** `storage/seed.py` inserts the G§17 rows: CUST-001 / CUST-002, ORD-1001 (no shipment yet), ORD-1002 (in transit, ETA 2026-10-15), ORD-1003 (delivered 2026-10-02), ORD-2002 (another customer), the Samsung / LG 55-inch TVs, VOXCOM_500K and the retired VOXCOM_TET_500K, and the fixture sessions. Order ids are 32-character hex (`order_id_for`), order codes are `ORD-####`.
- **Audit log** is append-only: the `voxcare_app` role has `UPDATE`, `DELETE` and `TRUNCATE` revoked (when the connecting user may manage roles), and two rules turn any `UPDATE` / `DELETE` into a no-op for every user. `order_id` is the hex FK; `order_code` and `tool_name` are inside `request_payload_json` (G-4). Rows are masked for PII before they are written.
- **Concurrency:** writes use optimistic concurrency (`UPDATE orders SET …, version = version + 1 WHERE order_code = %s AND version = %s`, G§11.4); a lost race is a `VERSION_CONFLICT`.
- **Driver:** psycopg 3 with a synchronous connection pool, called through `asyncio.to_thread`, so it runs on Windows' default event loop. Each connection sets `search_path`, `statement_timeout=1500` and `timezone=GMT` (offset 0 like UTC, but built into PostgreSQL, so it also works with the embedded server on Windows, which ships no time-zone files).

`scripts/init_db.py` creates the schema when missing and seeds it; `--reset` drops and reseeds (fresh demo fixtures). The app does the same on start (`VOXCARE_DB_RESET_ON_START=on` to always reseed).

**Where PostgreSQL comes from:**

1. `DATABASE_URL` set → that server (Docker: `docker compose up -d`; the bundled compose file exposes `postgresql://voxcare:voxcare@127.0.0.1:5432/voxcare`).
2. `DATABASE_URL` empty and the `localdb` group installed (`uv sync --group localdb`, Python 3.12 or older) → an embedded PostgreSQL in `data/pg`, run by `storage/embedded_pg.py` with the binaries from the `pgserver` wheel. The server runs in its own hidden console and process group, so Ctrl+C in the terminal stops the app, and the app then stops PostgreSQL cleanly (`pg_ctl stop -m fast`). Its log is `data/pg.log`, outside the data directory: on Windows a log inside it collides with the fsync PostgreSQL runs during crash recovery (sharing violation). Start-up waits up to 120 s, enough for a crash recovery. A server left running by an earlier run is stopped and restarted cleanly; one started by another live process (`init_db.py` next to the app) is shared. Data persists between runs; stop the app and delete `data/pg` to start over.
3. `VOXCARE_DATA_BACKEND=memory` → `MemoryRepository`, the same rows and the same queries in process memory. Nothing persists. It is what the unit tests use.

## Redis — volatile session state

| Key | Content | TTL |
| --- | --- | --- |
| `session:{session_id}` | bound `customer_id`, verification state, active order, slot state, turn epoch, paused topic, LLM failover state, masked history | 1800 s (`VOXCARE_SESSION_TTL_S`), refreshed on every turn |
| `pending_action:{action_id}` | guarded-write `PendingAction` (preview hash, status, confirmations) | 300 s (`VOXCARE_PENDING_ACTION_TTL_S`) |
| `idem:{idempotency_key}` | `action_id`; created with `NX`, so a key is never reused for a second action | 300 s |
| `handoff:{handoff_id}`, `session_handoffs:{session_id}` | `HandoffContextPacket` (trigger reason, priority, order, ticket, masked summary, transcript and symptom history) | expires with the session |
| `throttle:{session_id}` | sorted set of distinct catalog SKUs, scored by time (FR-SEC-01: 4 SKUs per 120 s) | 120 s window |

`REDIS_URL` empty → an in-process fakeredis server with the same commands. It is not shared between processes, which is fine for one uvicorn worker.

## Chroma — policy corpus

Collection `voxcom_policy_v1`, nine clauses from `domain/data/policies.json`, cosine space, embeddings computed by `storage/embeddings.py`. Persisted in `CHROMA_PATH` (default `data/chroma`); an empty `CHROMA_PATH` uses a temporary directory per process. The collection carries a fingerprint of the corpus, the embedding model and the embedded-text recipe and is rebuilt when any of them changes. Retrieval and the threshold: [POLICY_RETRIEVAL.md](POLICY_RETRIEVAL.md).

## Fake VoxCom OMS

`storage/fake_oms.py` stands in for the fulfilment system the guarded writes would call. It applies cancellations, reschedules, address changes and reversals to the repository under OCC, keeps an idempotency ledger keyed by `mutation_idempotency_key`, and has test knobs to hold a commit, lose a response, refuse an abort, or mark a refund as dispatched. It is used only when `VOXCARE_WRITE_MODE=fake_oms`; the default `record_only` never mutates an order.
