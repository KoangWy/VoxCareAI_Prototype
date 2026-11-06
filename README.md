# VoxCare AI

[![CI](https://github.com/KoangWy/VoxCareAI_Prototype/actions/workflows/ci.yml/badge.svg)](https://github.com/KoangWy/VoxCareAI_Prototype/actions/workflows/ci.yml)

English, browser-based **voice customer-support prototype** for the fictional electronics retailer **VoxCom**, built to the P1 implementation specification (`docs/IMPLEMENTATION_SPEC.md` in the thesis repository, design `research/42_voxcare_p1_detailed_design.md`). A caller speaks in the browser; the server transcribes the speech, decides when the caller has finished their turn, lets a language model propose read-only tool calls, runs them through an authorizing middleware, and speaks back a line rendered only from tool results — verbatim from the design wherever the design fixes the wording.

Every customer, order, shipment, product, voucher and policy clause is fictional. Caller verification is simulated by binding a fixture customer (OPD-04); production needs an OTP challenge.

## Features

- **Streaming voice loop** — browser microphone (AudioWorklet) → 16 kHz PCM over an authenticated WebSocket (`VoxCare-Session-Token`, HS256 JWT) → streaming ASR → judged end-of-turn control → LLM tool planning → streaming TTS.
- **Three-tier data layer** — PostgreSQL (DDL §4.1, seed fixtures §17, append-only `audit_log`), Redis (session TTL 1800 s, pending action 300 s, idempotency index, handoff packet, throttle window) and ChromaDB (`voxcom_policy_v1`, 9 fictional clauses, cosine).
- **Eight scope-guarded tools** — the design's seven read/record tools plus `list_customer_orders` (gap G-3). Every proposal passes the tool middleware: allowlist, untrusted-input stripping, a pre-invocation validator, the verified-session gate, the FR-SEC-01 anti-scraping throttle, one retry on timeout only, and an audit row. The model has no database access.
- **Verbatim spoken lines** — every line of the design's error matrices, the handoff script, the open-policy-decision fallbacks (OPD-01…06), withdrawal and replay lines lives in `domain/copy.py`.
- **Proposals behind flags (default on)** — FR-AUTH-01 phone-number verification sub-dialogue, FR-AUTH-02 scope refusal with audit, FR-SEC-01 throttle.
- **Bounded after-sales diagnosis** (≤ 3 turns), **context-packaged human handoff** (audit + `agent_activity` QUEUED + Redis packet; never a transfer), **interrupted-conversation resume** question (UC-23).
- **Guarded-write library** (future-only) — Preview → Confirm (1–2 steps) → Revalidate (OCC) → Execute → Audit, withdrawal cases A/B/C, reversal with a new key, replay by key K1 — tested against a fake VoxCom OMS. In the demo, writes are previewed and recorded for VoxCom, never executed.
- **Session activity log** — the operator view shows, for every turn, how it was handled (fixed rule or model plan), each tool call with the checks it passed or failed, the audit rows it wrote (same `audit_id` as in PostgreSQL), what changed, and the reply; masked as stored, downloadable as Markdown. No model names or timings.
- **Observability** — audit rows in the design's golden format, OpenTelemetry spans per turn (`turn`, `llm`, `tool:<name>`, `tts`, T1…T6 events), JSON logs masked for PII, `/healthz` with a WSS handshake probe.

## How it works

```text
Browser (static/)                        Server (src/voxcare/)
mic → AudioWorklet → 16 kHz PCM ──WSS──► voice/connection (token check) ──► ASR (Deepgram | Vosk)
local VAD ──────────────────────────────► turns/ judge (baseline | Jev | OpenRouter) ── COMMIT
                                                   ▼
                                          session/boundary ─ PII mask ─► session/dialogue
                                                   │  model proposes tools (OpenAI → Anthropic failover | keyword)
                                                   ▼
                                          middleware/tool_middleware ─► domain/services ─► PostgreSQL · Redis · Chroma
                                                   ▼ envelope → session/render (domain/copy.py lines)
speaker ◄── PCM16 playback ◄────────────── voice/dispatch ──► TTS (ElevenLabs | Piper)
```

## Quick start (Windows, macOS, Linux)

Requirements: Python 3.12 (3.13/3.14 work without the embedded database), [uv](https://docs.astral.sh/uv/), a current Chromium-based browser, Node.js 22+ only for the frontend tests.

```sh
uv sync --all-groups              # app + dev tools + demo speech (vosk, piper) + embedded PostgreSQL (localdb)
cp .env.example .env              # PowerShell: Copy-Item .env.example .env
uv run python scripts/init_db.py  # creates and seeds the database, builds the policy index
uv run uvicorn voxcare.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The first start downloads Chroma's `all-MiniLM-L6-v2` embedding model (about 80 MB) once.

**Choosing the stores** (all in `.env`; details in [docs/DATA_LAYER.md](docs/DATA_LAYER.md)):

| Setup | What to set |
| --- | --- |
| Docker Desktop | `docker compose up -d`, then `DATABASE_URL=postgresql://voxcare:voxcare@127.0.0.1:5432/voxcare` and `REDIS_URL=redis://127.0.0.1:6379/0` |
| No Docker (Python 3.12) | leave `DATABASE_URL` empty: an embedded PostgreSQL (`pgserver`) runs in `data/pg`; leave `REDIS_URL` empty: an in-process fakeredis server |
| No database at all | `VOXCARE_DATA_BACKEND=memory` (same seed rows in process memory; for quick UI checks) |

**Choosing the conversation model**: `VOXCARE_CONVERSATION_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`; `openai` with `OPENAI_API_KEY` (+ Anthropic as sticky fallback); or `keyword` — an offline deterministic planner (not an LLM) for demos without credits.

**Speech**: `uv run --group demo python scripts/fetch_demo_models.py` downloads Vosk + Piper for the local stack (`VOXCARE_ASR_PROVIDER=vosk`, `VOXCARE_TTS_PROVIDER=piper`); Deepgram + ElevenLabs keys select the design providers.

## Using the demo

| Ask (verified caller) | Spec scenario | What you hear |
| --- | --- | --- |
| "Can you check the delivery status of my order ORD-1002 and my earlier purchases?" | 1 | "Order ORD-1002 is in transit with GHN. Estimated delivery date: 2026-10-15. Your account also shows ORD-1003, delivered on October 2." |
| "Please check order ORD-2002." | 3 | refusal + `AUTH_SCOPE_VIOLATION` audit, nothing leaked |
| "What is the warranty period for a Samsung refrigerator compressor and can I return it if it has a manufacturing defect?" | 4 | Clause 3.1 and 4.1, conditional on technician certification |
| "Do you accept trade-ins for 10-year-old microwave ovens with cash back?" | 5 | out-of-domain admission, no fee or timeline |
| "Compare the Samsung and the LG 55-inch TV, and check if voucher VOXCOM_500K applies." | 7 | "Samsung 55: 11,490,000 VND. LG 55: 18,990,000 VND. With VOXCOM_500K the Samsung is 10,990,000 VND." |
| "My washer stops during spin cycle and displays error E4." → answer two questions → "exchange" | 8 | conditional eligibility; request recorded, no RMA ticket |
| "Let me speak to a human supervisor right now!" | 9 | §12 line with ticket `TICK-CS-…`; no transfer claim |
| "Where is my package?" → "ORD-1001" | 17 | "Could you please tell me your order code?" → null-ETA line |
| "Move the delivery of ORD-1001 to next Monday." → "Yes" | 11 | preview line, then "recorded … nothing has changed yet" |
| "When will I get my refund?" | OPD-02 | the fallback line only |

Choose **Unverified caller** in the side panel for Scenario 2: the bot asks for the phone number used at checkout (demo: `0900 000 001`), binds the customer, then answers. The support button records a handoff the same way.

## Development

```sh
uv run pytest -q                                      # 400+ tests (41 need the real stores, 1 the embedded PostgreSQL); manual script: docs/MANUAL_TESTS.md
uv run ruff check . && uv run ruff format --check .
npm test && node scripts/check_static.mjs             # frontend tests and static checks
```

Tests run on the in-memory twin by default. Set `VOXCARE_TEST_DATABASE_URL` and `VOXCARE_TEST_REDIS_URL` to run the acceptance, audit and security suites against real PostgreSQL and Redis too (CI does). See [docs/TESTING.md](docs/TESTING.md).

## Documentation

- [Spec conformance](docs/SPEC_CONFORMANCE.md) — the 17 scenarios, NFR status, gap decisions G-1…G-14, deviations
- [Architecture](docs/ARCHITECTURE.md) · [Data layer](docs/DATA_LAYER.md) · [Policy retrieval and G-10 calibration](docs/POLICY_RETRIEVAL.md)
- [Voice protocol](docs/VOICE_PROTOCOL.md) · [Turn control](docs/TURN_CONTROL.md) · [Frontend](docs/FRONTEND.md) · [Testing](docs/TESTING.md) · [Policy sources](docs/POLICY_SOURCES.md)

## Limitations

- English only (Phase 1). Caller verification is simulated (OPD-04). Writes are never executed in the demo.
- No NFR is measured by this repository yet: the scripts in `scripts/` (`nfr_latency_report.py`, `uptime_probe.py`, `load_test.k6.js`, `pii_log_scan.py`) produce the evidence when run against a live setup.
- The policy threshold is calibrated for the chosen embedding model (0.32 for all-MiniLM-L6-v2, measured on 50 labeled queries); the design's 0.70 belongs to an unnamed model (G-10).
- PII masking is pattern- and dictionary-based (phones, addresses, fixture names), not a general PII detector; use fictional inputs.

## License

No license has been chosen yet; all rights are reserved by the author until one is added.
