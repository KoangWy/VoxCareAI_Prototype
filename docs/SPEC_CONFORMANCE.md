# Spec conformance (P1 build guide)

How this prototype maps to the P1 implementation specification (`docs/IMPLEMENTATION_SPEC.md` in the thesis repository, cited as **G§n**) and the detailed design (`research/42_voxcare_p1_detailed_design.md`, cited as **D§n**). "Implementation choice" marks anything the guide leaves open; every such line is also tagged in the code.

## Build order (G§19.2)

| M | Deliverable | Where | Exit test |
| --- | --- | --- | --- |
| M1 | Schema, seed, enums, rules, audit writer | `storage/schema.sql`, `storage/seed.py`, `domain/enums.py`, `domain/rules.py`, `observability/audit.py` | `test_fsm.py`, `test_rules.py`, `test_audit_writer.py` (golden rows) |
| M2 | Services, tools, middleware, validator, throttle, PII masker | `domain/services/`, `middleware/`, `security/pii.py` | Scenarios 1, 2, 3, 7, 17; `test_security_scope.py` |
| M3 | Chroma corpus and retrieval | `storage/vector_store.py`, `storage/embeddings.py`, `domain/services/policy.py` | Scenarios 4, 5; `test_policy_retrieval.py` |
| M4 | LLM adapters, failover, text dialogue loop, output guards | `providers/llm.py`, `session/dialogue.py`, `session/render.py` | Scenarios 1–9, 17 with a scripted planner; `test_llm_failover.py` |
| M5 | Audio gateway, ASR/TTS adapters, browser, T0–T6 | `voice/`, `providers/asr.py`, `providers/tts.py`, `static/`, `observability/otel.py` | voice protocol tests; `scripts/offline_voice_e2e.py` |
| M6 | Turn controller: HOLD bounds, degraded state, barge-in, epochs | `turns/`, `voice/` | Scenario 10; `test_turns.py`, `test_turn_control.py`, `test_voice_barge_in.py` |
| M7 | Guarded-write library and fake OMS | `domain/services/writes.py`, `storage/fake_oms.py` | Scenarios 6, 11–16 |
| M8 | Measurement | `scripts/nfr_latency_report.py`, `uptime_probe.py`, `load_test.k6.js`, `pii_log_scan.py` | **not run yet** — needs a live setup (see NFRs) |
| M9 | Speculative reads (optional) | `session/speculative.py` | guard and cache only; `test_no_speculative_write.py` |

## Acceptance scenarios (G§18)

All 17 run at the dialogue level (a), with a scripted planner instead of the LLM, in `tests/acceptance/test_scenarios.py`. Each runs on the in-memory twin and, when `VOXCARE_TEST_DATABASE_URL` / `VOXCARE_TEST_REDIS_URL` are set, again on PostgreSQL + Redis. The live level (b) is optional in the guide and has not been run as a suite.

| # | Asserted | Notes |
| --- | --- | --- |
| 1 | lookup + track + `list_customer_orders`; lookup response has no shipment fields; exact spoken line | history sentence via the G-3 tool; also a two-turn variant |
| 2 | `UNAUTHENTICATED_SESSION` before any `orders` read (repository read log); Scenario 2 line | FR-AUTH-01 sub-dialogue then binds the customer by phone number |
| 3 | 403 `OUT_OF_SCOPE`; `AUTH_SCOPE_VIOLATION` / `Denied` / `DENIED` audit row; no ORD-2002 data in the reply | full Scenario 3 sentence (G-5) |
| 4 | POL-WAR-001 and POL-RET-001 cited, corpus metadata present, conditional wording | threshold per G-10 below |
| 5 | `INSUFFICIENT_EVIDENCE`, http 200, best score below threshold; no fee or timeline words | |
| 6 | pending → submitted → executing → completed; version 1 → 2; `EXECUTION_PENDING` then `ORDER_CANCELLED`; no success line before the result | library level and dialogue level with `VOXCARE_WRITE_MODE=fake_oms` |
| 7 | three calls in order; discount 500000; no invented attribute; exact line | |
| 8 | ≤ 3 diagnostic turns; `MOTOR_DRIVE_FAULT`; RETURNS policy; `escalate_to_human(COMPLEX_EXCEPTION, NORMAL)`; no `create_rma_request` | |
| 9 | audit + `agent_activity` QUEUED + Redis packet; §12 line; no transfer claim | |
| 10 | HOLD at 0.12, COMMIT at 0.95 at epoch 7; degraded after 2 non-commits or 3000 ms; barge-in → epoch 8, stale frames dropped; resume question; decline does not undo | scripted Jev stub (G-13) |
| 11 | preview with zero mutations; 1–2 confirmations; OCC update; demo records only | OCC execution tested against the fake OMS |
| 12 | withdrawal cases A / B / C with their audits and lines | Case C uses the Scenario 12 wording (G-14) |
| 13 | withdrawal after commit → completed + `ORDER_CANCELLED`; routes to reversal | |
| 14 | reinstatement with a new key ≠ K1; `reversal_of_action_id`; version 2 → 3; `REVERSAL_REQUESTED` | |
| 15 | `REVERSAL_REFUSED` after refund dispatch; no mutation; escalate | |
| 16 | replay K1 in four states; key never re-minted; never two executions | plus permanent-conflict and lost-response cases |
| 17 | validator blocks the null id (tool not called); slot re-ask; null-ETA line; no date word | G-2 null shipment shape |

Cross-cutting files (G§18): `test_security_scope.py` (50 cross-customer, 50 unauthenticated, 20 scraping bursts, plus flag-off checks), `test_fsm.py`, `test_turns.py` / `test_turn_control.py`, `test_no_speculative_write.py`, `test_llm_failover.py`, `test_pii_logs.py`.

## NFRs (G§19.1)

Every NFR is a **target, not yet measured**. The repository produces the evidence; the numbers do not exist until the scripts run against a live setup.

| NFR | Target | Evidence path | Status |
| --- | --- | --- | --- |
| NFR-01a | median voice-to-voice ≤ 1000 ms | `client_playback` span (T0, T6) per turn → `nfr_latency_report.py` | not measured; needs ≥ 150 English turns |
| NFR-01b | TTFA ≤ 755 ms (distribution) | `ttfa_playback_start_ms` = T6 − T0, browser clock | not measured; browser playback start is not an acoustic confirmation |
| NFR-02 | full turn p50 ≤ 2.0 s, p95 ≤ 3.5 s | `turn`, `llm`, `tool:*`, `tts` spans | not measured |
| NFR-03 | intent ≥ 85 %, slot F1 ≥ 80 % | — | not measured; no labeled intent set (N ≥ 30 per intent) exists yet |
| NFR-04 | uptime ≥ 99.5 % | `/healthz?wss=1` (stores + WSS handshake ≤ 2.0 s) → `uptime_probe.py` | not measured |
| NFR-05 | 0 plain-text PII in logs | masking log filter + `pii_log_scan.py`; `test_pii_logs.py` | tested on fixture traffic only; cross-border audio is not addressed (Deepgram / ElevenLabs are external; Vosk / Piper stay local) |
| NFR-06 | 50 concurrent calls, TTFA p95 ≤ 1200 ms | `load_test.k6.js` at 10 / 25 / 50 | not measured; the k6 script loads typed turns and the WSS handshake, not real audio |
| NFR-07 | 0 lines changed in `domain/rules.py` on a provider swap | `rules.py` imports only the standard library and `enums`, so no provider can reach it; `test_llm_failover.py` swaps the conversation model mid-session | holds by construction; a full Config A / Config B regression run is not set up |
| NFR-08 | 0.0 % leakage; every burst intercepted | `test_security_scope.py` | passes (0 / 100 leaked, 20 / 20 bursts intercepted) |

## Design gaps (G§20): decisions taken

| Gap | Decision in this prototype |
| --- | --- |
| G-1 | Window start in `requested_delivery_time`, `requested_delivery_window_end` added; rendered `YYYY-MM-DD HH:MM-HH:MM`. |
| G-2 | ORD-1001 tracking returns `shipment:{order_code, estimated_delivery_date:null, …null}` and `events: []`; no synthetic row. |
| G-3 | **Added the scope-guarded read tool `list_customer_orders`** (design change, chosen by the team). It returns only the session customer's orders, with `delivered_on` from the DELIVERED event, and needs a verified session. |
| G-4 | `audit_log.order_id` keeps the FK (32-char hex id); `order_code` is also written into `request_payload_json`. |
| G-5 | The matrix line is the tool failure line; the Scenario 2 line is the FR-AUTH-01 prompt; the Scenario 3 sentence is the full refusal. |
| G-6 | Packet `order_id` nullable; SES-VOX-8839205 bound to CUST-001 as fixture filler. |
| G-7 | Promotion is standalone; `apply_voucher` is a quote and never links to an order. |
| G-8 | VOXCOM_500K is a fixed 500,000 VND discount (equal to the cap). To confirm with the design owner. |
| G-9 | Every tool needs a valid session token; a bound customer is required for lookup, track, `search_catalog`, `escalate_to_human` and `list_customer_orders`. |
| G-10 | Embedding model **all-MiniLM-L6-v2** (Chroma's default), threshold **0.32, measured on 50 queries** (32 in-domain, 18 out-of-domain): accuracy 0.82, 27/32 grounded, 14/18 refused, correct top-1 clause 31/32; Scenarios 4 and 5 pass. At the design's 0.70 the same set scores 0.42 (3/32 grounded). The offline hashing embedding is calibrated separately (0.09 → 0.84). Configurable with `VOXCARE_RAG_THRESHOLD`. Details: [POLICY_RETRIEVAL.md](POLICY_RETRIEVAL.md). |
| G-11 | Fixtures (D§13) are the source of truth; audit examples are format-only (ORD-1002 ETA is 2026-10-15). |
| G-12 | `search_catalog` / `compare_products` success shapes follow G§7.4 / G§7.5. |
| G-13 | Jev integration uses the existing adapter plus a scripted stub in tests; the baseline judge is the default. |
| G-14 | Case C uses the Scenario 12 wording ("No changes were made to order ORD-1001."). |

## Implementation choices and known deviations

- **Data access.** PostgreSQL through psycopg 3's synchronous connection pool on worker threads, because the async driver cannot run on Windows' default Proactor event loop. All tables live in a dedicated schema (`VOXCARE_DB_SCHEMA`, default `voxcare`). `audit_log` is append-only by role (`REVOKE UPDATE, DELETE, TRUNCATE`) and by rule. The in-memory twin implements the same repository protocol for unit tests and the no-database quick start.
- **Redis.** A real server when `REDIS_URL` is set, otherwise an in-process fakeredis server with the same commands (single process only).
- **Writes.** The guarded-write library is complete and tested against the fake OMS. The Phase 1 dialogue records write requests (`VOXCARE_WRITE_MODE=record_only`, default) and never claims an execution. `fake_oms` mode runs the full Preview → Confirm → Execute path in the dialogue.
- **Verification** is simulated: FR-AUTH-01 binds the fixture customer whose phone number the caller gives (OPD-04). Production needs an OTP challenge.
- **TTFT.** Tool planning is not streamed, so the `llm` span's T3 marks plan completion, not the first token. The failover timeout (1.5 s, `VOXCARE_LLM_TTFT_TIMEOUT_MS`) applies to that call and only when a fallback model is configured.
- **Barge-in message.** The browser reports a caller onset during playback as `vad_speech_start` with `agent_speaking: true` (protocol 2); the server also accepts the design's `{"event":"BARGE_IN_TRIGGERED","interrupted_epoch":n}` and treats it the same way.
- **Uplink frames** are 512 bytes / 16 ms (`ASR_CHUNK_BYTES` / `ASR_CHUNK_MS`). The server does not reject other sizes (scripts stream 20 ms frames).
- **Speculative reads (M9)** are not performed; the guard that keeps write tools out of that path is in place and tested.
- **Conversation model.** OpenAI → Anthropic cascade as designed, or Anthropic as primary. A deterministic keyword planner (`VOXCARE_CONVERSATION_PROVIDER=keyword`) exists for offline demos and load tests; it is not an LLM and says nothing about model accuracy.
- **Clock.** The demo pins time to 2026-10-14 09:42 +07:00 (`VOXCARE_CLOCK=demo`) so the fixtures' dates read as designed; `system` uses real time.
- **Dialogue robustness** (from recorded demo sessions; tests in `tests/test_session_regressions.py`):
  - an order code given after "Could you please tell me your order code?" repeats lookup and tracking itself instead of asking the model again;
  - after verification, the exact tool calls the gate refused run again, so the caller's original intent is kept;
  - a write preview is confirmed only by the very next answer: any other turn withdraws it (`ACTION_ABORTED_PRE_SUBMIT`) and the reply says the change was not submitted, so a later "yes" to an unrelated question can never seal it;
  - rescheduling to the day already booked answers that nothing needs to change, with no preview;
  - a missing new address is asked for instead of refused;
  - a repair request without a recognised symptom asks what is wrong; an undiagnosable fault offers a technician request (only the washer E4/spin tree is diagnosable by phone);
  - when the model proposes no tool, a question naming a policy topic still searches the policy corpus, a request for a list of promotions gets a capability answer, a policy question naming no topic lists the topics, and only the rest is a clarification; the limit counts unclear turns in a row.
- **PII masking** covers phone numbers, addresses, fixture names and credentials by pattern and dictionary; it is not a general PII detector.
