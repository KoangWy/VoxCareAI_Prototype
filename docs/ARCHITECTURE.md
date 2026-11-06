# Architecture

VoxCare AI is one FastAPI process that serves a static browser client, a REST API and one authenticated WebSocket per voice session. Business data lives in PostgreSQL, volatile session state in Redis, and the policy corpus in Chroma ([DATA_LAYER.md](DATA_LAYER.md)). The language model only proposes tool calls; every spoken line is rendered from tool results.

## Module map

```text
src/voxcare/
  app.py               ASGI entry point: `voxcare.app:app` and the process-wide `boundary`
  config.py            every VOXCARE_* variable, parsed and validated (ConfigError names the variable)
  runtime.py           builds the stores and services from DataSettings; describe() and health()
  contracts.py         PlannedToolCall, ToolOutcome, SessionView, TurnResult, DialogueProvider
  privacy.py           deterministic mask applied at external-provider edges (LLM, judge, TTS)
  trace.py             bounded per-session event ledger (500 events, oldest dropped first)

  domain/              pure business layer (no I/O in rules.py)
    enums.py           every enum of G§5
    rules.py           scope, order FSM, write lifecycle, voucher, return window, diagnosis, reversal
    copy.py            every verbatim spoken line (matrices, handoff, OPD-01…06, writes)
    clock.py           FixedClock (demo anchor 2026-10-14 09:42 +07:00) and SystemClock
    services/          orders, shipment, customer orders, policy, catalog, promotion, handoff, writes; envelope
    data/policies.json the 9-clause fictional VoxCom policy corpus

  storage/
    schema.sql, seed.py    DDL G§4.1 and the G§17 fixtures
    db.py                  Repository protocol: PostgresRepository (psycopg 3) and MemoryRepository
    redis_store.py         sessions, pending actions + idempotency index, handoff packets, throttle window
    vector_store.py        Chroma voxcom_policy_v1; embeddings.py: MiniLM or hashing embedder
    fake_oms.py            fake VoxCom OMS for the guarded-write library

  middleware/
    registry.py        the 8 tool schemas (strict), Phase 1 allowlist, verified-customer set
    validator.py       pre-invocation argument checks (missing slot → re-ask, tool not called)
    throttle.py        FR-SEC-01 catalog throttle
    tool_middleware.py allowlist → strip untrusted input → validate → auth gate → throttle → call → audit

  security/            pii.py (masker + log filter), tokens.py (VoxCare-Session-Token, HS256)
  observability/       audit.py (AuditWriter), otel.py (spans + ring buffer), logging.py (JSON, masked)

  session/
    boundary.py        SessionBoundary: session records, tokens, history, turn handling, handoff, audit
    dialogue.py        DialogueManager: verification, slots, pending writes, diagnosis, OPD, resume, LLM plan
    intents.py         deterministic detectors (human request, OPD topics, write intents, dates)
    render.py          answer text from tool envelopes + output guards (no write or ETA claims)
    speech.py          voice rendering of the same answer for TTS
    speculative.py     speculative-read guard (writes can never enter it)
    activity.py        per-turn activity log: route, tool checks, audit rows, state changes, reply (masked)
    tools.py           spoken order-code normalization; registry re-export

  http/                bootstrap.py (lifespan builds the runtime), routes.py, schemas.py
  voice/               WebSocket voice session (protocol, connection, ASR events, evaluation, dispatch)
  turns/               TurnController, baseline judge, Jev and OpenRouter adapters
  providers/           ASR (Deepgram, Vosk), LLM (OpenAI, Anthropic, cascade, keyword planner), TTS (ElevenLabs, Piper)
```

## Turn pipeline

```text
caller text (typed, or committed ASR final)
  → SessionBoundary.handle_turn: token-bound session, PII-masked history, `turn` span (T1)
  → DialogueManager: deterministic branches first (resume answer, verification, handoff offer,
    pending write yes/no, withdrawal, diagnosis, human request, OPD topics, write intents),
    otherwise the model plans tool calls (`llm` span, T2/T3)
  → ToolMiddleware for each call (`tool:<name>` span, audit row)
  → render_results: verbatim lines from domain/copy.py; guards drop any write or ETA claim
    the envelopes do not support
  → TurnResult (screen text) and spoken_reply (TTS, `tts` span T4/T5)
```

## REST surface

```text
POST /session  (= POST /api/sessions)  {verified, fixture_session_id?} → session + VoxCare-Session-Token
POST /api/sessions/{id}/token          refresh the token (≤ session TTL)
POST /api/sessions/{id}/turns          typed turn (same path as voice turns)
GET  /api/sessions/{id}/trace | /audit | /handoffs | /spans
GET  /api/sessions/{id}/activity[?format=markdown]   the session activity log (operator view, export)
POST /api/sessions/{id}/handoff        record a support request (never a live transfer)
POST /api/sessions/{id}/reset          discard the session and create a fresh one (new token)
GET  /api/config                       provider and store configuration, never credentials
GET  /healthz[?wss=1]                  PostgreSQL, Redis, Chroma (+ WSS handshake ≤ 2 s); 503 when degraded
```

Every session route needs the token (`VoxCare-Session-Token` header or `Authorization: Bearer`); the voice socket takes it as `?token=` and closes with 4401 / 4404 otherwise.

Typed turns never touch the turn judge: typed text has no microphone timing to judge.

## Voice flow (WebSocket)

```text
connect  → session hello (protocol 2, providers, barge_in mode)
start    → ASR attaches, TTS warms up → voice_started
PCM16    → ASR → transcript_partial / transcript_final (versioned)
vad_speech_start → new HOLD episode (only a local-VAD onset resets the budget)
vad_quiet        → evaluation: judge → turn_control
    COMMIT   → turn (grounded result) → audio_start / audio_chunk + PCM … / audio_end
    HOLD     → wait; re-judged after more silence when that can change the verdict
    DEGRADED → recovery_required → caller picks manual_finish / caller_retry / handoff_request
reset / stop
```

`SessionBoundary.handle_turn` is the single path for both typed and committed voice turns, so tools, scoping, auditing, and rendering never diverge.

Two voice-only steps sit on either side of that path, both deterministic:

- **In:** a committed transcript (`source="asr_final"`) passes through `normalize_spoken_order_codes` (`session/tools.py`) before planning. Spelled or misheard codes such as "o r d one zero zero two", "or rd one zero zero two" or "order forty five sixty seven" become `ORD-####`; exactly four digits must follow, so "order two TVs" is untouched. Each rewrite is traced as `SPOKEN_ORDER_CODE_NORMALIZED`. Typed text is never rewritten.
- **Out:** the `turn` message carries the full grounded `assistant_text` for the screen, while TTS receives `spoken_reply(...)` (`session/speech.py`): the same tool outcomes rendered for listening. Policy answers speak the first sentence of the top-ranked clause verbatim, say how many more clauses apply, and point to the transcript for the full text; ISO and day/month/year dates become "October 15, 2026"; `VND` becomes "dong"; `ORD-1002` is read "O R D one zero zero two"; clause ids and SKU parentheticals are dropped. No fact is added. `text_sent_to_tts` records `spoken_chars`.

## Turn decisions

`TurnController` (one per session) runs episodes fenced by `(session, turn_epoch, hold_episode_id, decision_request_id)`. A COMMIT requires a stable/final transcript, positive mic silence past a confidence-scaled confirm window, a complete *and* addressed judgment, and no holding-floor signal. Two non-commit verdicts or the absolute HOLD deadline degrade the episode to an explicit caller choice; a timeout or error never commits. Details: [TURN_CONTROL.md](TURN_CONTROL.md).

## Provider matrix

| Role | Design provider | Demo alternative (env-selected) | Shared contract |
| --- | --- | --- | --- |
| ASR | Deepgram Nova-2 | Vosk offline (`VOXCARE_ASR_PROVIDER=vosk`) | normalized transcript events |
| Conversation | OpenAI Responses `gpt-6-luna` | chat-completions gateway (`VOXCARE_OPENAI_TRANSPORT=chat`) | `plan_turn` → ordered tool calls |
| Fallback | Anthropic `claude-haiku-5-5` | — | sticky after 2 retryable failures |
| TTS | ElevenLabs Flash v2.5 | Piper offline (`VOXCARE_TTS_PROVIDER=piper`) | PCM16 mono 16 kHz chunks |
| Turn decision | TypeSafe Jev (300 ms / 3000 ms) | OpenRouter Decisions (2500 ms / 9000 ms) | the same four Noul questions |

`plan_turn` returns every tool call the model emitted, in order. Adapters allow several calls in one response (`parallel_tool_calls=True`; Anthropic `disable_parallel_tool_use=False`), because Scenario 1 needs lookup and tracking together and Scenario 7 three calls in order, and fall back gracefully when a gateway rejects the parameter; the boundary executes up to `MAX_TOOL_CALLS_PER_TURN = 3` calls sequentially and composes one answer (a recorded handoff wins outright; otherwise fragments join in order and the outcome is the first non-ANSWERED outcome).

## Data and honesty boundaries

- Eight tools: `lookup_order_details`, `track_shipment_status`, `list_customer_orders` (G-3), `retrieve_policy_clauses`, `search_catalog`, `compare_products`, `apply_voucher`, `escalate_to_human`. Unknown tools and the future write tools are rejected by the allowlist; the model has no database access.
- Order, shipment, catalog, history and handoff tools need a verified customer; another customer's order returns 403 `OUT_OF_SCOPE` with an `AUTH_SCOPE_VIOLATION` audit row and no details (FR-AUTH-02).
- Answer text is rendered from tool envelopes (`session/render.py`) with the design's verbatim lines; the spoken form is a deterministic rewrite of it (`session/speech.py`). Output guards drop any write-success or delivery-date claim that no envelope supports.
- Writes are previewed and recorded in Phase 1 (`VOXCARE_WRITE_MODE=record_only`); the guarded-write library runs end to end only against the fake OMS.
- PII is masked in logs, audit rows, stored history and handoff packets (`security/pii.py`), and at every external-provider edge (`privacy.py`). Live transcripts on the caller's own screen keep the raw words.
- Bounds: 200 live sessions, 24 history messages per session, 500 trace events per session, 64 KiB request bodies, 64 KiB audio frames, 24 MiB audio per voice session.
