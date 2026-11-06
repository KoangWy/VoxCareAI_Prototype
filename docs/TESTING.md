# Testing

## Automated checks

Run before every change; CI (`.github/workflows/ci.yml`) runs the same set, with PostgreSQL 16 and Redis 7 service containers.

```sh
uv sync --all-groups
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
node --test 'tests/frontend/*.test.mjs'
node scripts/check_static.mjs
```

By default every test runs on the in-memory twin (`MemoryRepository`, fakeredis, hashing embedding, temporary Chroma), so the suite needs no services and no network. To run the acceptance, audit, storage and security suites against real stores too, point the tests at them. Each PostgreSQL test creates, seeds and drops its own random schema (`t_…`), so the app's `voxcare` schema is untouched; the Redis database is **flushed** by each test, so use a spare database number:

```sh
docker compose up -d
export VOXCARE_TEST_DATABASE_URL=postgresql://voxcare:voxcare@127.0.0.1:5432/voxcare
export VOXCARE_TEST_REDIS_URL=redis://127.0.0.1:6379/15
uv run pytest -q          # the parametrized suites now also run with [postgres]
```

(PowerShell: `$env:VOXCARE_TEST_DATABASE_URL = "postgresql://…"`.)

| Suite | Covers |
| --- | --- |
| `tests/acceptance/test_scenarios.py` | the 17 acceptance scenarios of G§18 at the dialogue level (scripted planner), on memory and PostgreSQL |
| `tests/test_security_scope.py` | NFR-08: 50 cross-customer, 50 unauthenticated, 20 scraping-burst cases, sliding window, flags off |
| `tests/test_fsm.py`, `tests/test_rules.py` | every valid and forbidden order transition; voucher, return window, diagnosis, reversal rules |
| `tests/test_tool_middleware.py`, `tests/test_dialogue.py` | allowlist, validator, auth gate, retry-on-timeout only, audit rows; dialogue branches (verification, slots, OPD, writes, resume) |
| `tests/test_storage.py`, `tests/test_policy_retrieval.py` | repository parity, OCC, append-only audit, Redis keys and TTLs; Chroma retrieval, topic and applicability filters |
| `tests/test_audit_writer.py`, `tests/test_pii_logs.py`, `tests/test_privacy.py` | golden audit rows; NFR-05 log scan on fixture traffic; provider-edge mask |
| `tests/test_session_tokens.py`, `tests/test_healthz.py`, `tests/test_http_api.py`, `tests/test_config.py` | token issue / verify / expiry on REST and WebSocket; `/healthz` incl. the WSS probe; REST surface; configuration validation |
| `tests/test_llm_failover.py`, `tests/test_provider_cascade.py`, `tests/test_*adapters.py`, `tests/test_jev.py`, `tests/test_openrouter_judge.py` | failover after two qualifying failures and its audit; provider adapters with fakes only |
| `tests/test_no_speculative_write.py` | no write tool can reach the speculative path |
| `tests/test_session_regressions.py` | turn-by-turn replays of recorded demo sessions: slot fill, verification resume, stale write previews, same-day reschedule, repair intake, policy fallbacks, clarification limit |
| `tests/test_embedded_pg.py` | the embedded PostgreSQL: start, crash recovery, sharing, clean stop (runs where the `localdb` group is installed) |
| `docs/MANUAL_TESTS.md` | the manual script for the browser, with expected replies and what to check in the activity log |
| `tests/test_activity.py`, `tests/frontend/activity.test.mjs` | session activity log: route, checks, audit ids matching `audit_log`, masked phone only, no model names or timings, endpoint and Markdown export; browser rendering |
| `tests/test_turns.py`, `tests/test_turn_control.py` | turn controller: commit rule, budgets, deadlines, re-arming, stale-result quarantine, barge-in scoring |
| `tests/test_voice_recovery_protocol.py`, `tests/test_voice_barge_in.py`, `tests/test_protocol_contract.py`, `tests/test_speech_text.py` | the real WebSocket flow with fake ASR / TTS / LLM / judges; message-name contract; voice rendering |
| `tests/frontend/*.test.mjs` | VAD on synthetic clips, playback guards, mic races, reducers, reconnect, resampling, 512-byte uplink frames, token headers |

`tests/conftest.py` removes every `VOXCARE_*` variable and provider key before each test and sets the in-memory store settings, so the suite never depends on a developer's `.env` and never reaches a real provider.

## Measurement scripts (NFRs, not yet run)

| Script | NFR | Output |
| --- | --- | --- |
| `scripts/nfr_latency_report.py spans_*.json` | NFR-01, NFR-02 | p50 / p90 / p95 / max of turn, LLM, tool, TTS first audio and TTFA (T6 − T0), with N |
| `scripts/uptime_probe.py --duration 3600` | NFR-04 | measured uptime over the run (`/healthz?wss=1`, handshake ≤ 2.0 s) |
| `k6 run -e STREAMS=50 scripts/load_test.k6.js` | NFR-06 | turn latency and WSS handshake under 10 / 25 / 50 virtual callers (typed turns, not audio) |
| `scripts/pii_log_scan.py server.log` | NFR-05 | plain-text phone, address, name and credential hits in a log file |
| `scripts/calibrate_policy_threshold.py` | G-10 | grounding accuracy per threshold on 50 labeled queries ([POLICY_RETRIEVAL.md](POLICY_RETRIEVAL.md)) |

Spans come from `GET /api/sessions/{id}/spans` (with the session token). Report every result as measured, with its N; none of these numbers exists yet.

## Offline voice check (no keys, no network)

```sh
uv run --group demo python scripts/offline_voice_e2e.py               # plain, hesitation, echo
uv run --group demo python scripts/offline_voice_e2e.py plain,echo
```

Runs the `live_voice_e2e.py` scenarios with real Piper, Vosk, turn control, tools and rendering on the in-memory stores, but replaces the two network dependencies: the keyword planner (`providers/keyword.py`) stands in for the conversation model and the application baseline for the turn judge. It shows that the voice loop works end to end on a machine with only the demo models; it says nothing about the real model's tool choices. The semantic barge-in scenarios need a real judge and are not in the default list.

## Live checks (real providers)

These scripts use the providers configured in `.env` and may cost a little provider credit. They print PASS/FAIL lines without secrets and exit non-zero on any failure.

```sh
uv run python scripts/live_voice_e2e.py                     # all scenarios
uv run python scripts/live_voice_e2e.py --scenarios plain,echo
uv run python scripts/live_mercury_check.py                 # OpenRouter judge on the controlled cases + one WebSocket turn
```

`live_voice_e2e.py` needs the local demo stack (Vosk + Piper models under `models/`, a gateway key in `OPENAI_API_KEY`, and the key for the configured `VOXCARE_TURN_MODE`). It synthesises English utterances with Piper, streams them as 20 ms frames with browser-style VAD messages over the real `/ws` endpoint, and checks five scenarios:

1. **plain** — a policy question is transcribed, committed, answered, and spoken (`audio_start` / `audio_end` ids match);
2. **hesitation** — "Let me think." holds, the continuation commits;
3. **echo** — the assistant's own audio fed back as microphone input commits no turn;
4. **backchannel** (semantic mode) — "uh-huh" during the reply never interrupts it;
5. **interruption** (semantic mode) — a floor grab gets a `taking_floor` verdict and an `interrupt`, and no `audio_end` follows it.

Piper adds random variation to every synthesis, and the small Vosk model occasionally mis-hears a sentence (for example "what does the return window for my tv"), which a judge may correctly treat as incomplete. Rerun a single failing scenario before treating it as a regression.

## Turn-decision evaluation

`scripts/evaluate_turn_cases.py` compares paired baseline/judged observations of the cases in `evaluation/jev_cases.json`; see [TURN_CONTROL.md](TURN_CONTROL.md#controlled-cases-and-evaluation). Without input it reports that no accuracy or latency result exists, which is the current state.

## Manual acceptance

Browser and microphone acceptance is manual; follow the checklist in [FRONTEND.md](FRONTEND.md#manual-acceptance-checklist).
