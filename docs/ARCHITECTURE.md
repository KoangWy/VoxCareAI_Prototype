# Architecture

VoxCare AI is a single FastAPI process that serves a static browser client, a small REST API, and one WebSocket per voice session. All business data is fictional and read-only; the only shared mutable state is the in-memory session registry.

## Module map

```text
src/voxcare/
  app.py               ASGI entry point: `voxcare.app:app` and the process-wide `boundary`
  config.py            every VOXCARE_* variable, parsed and validated (ConfigError names the variable)
  contracts.py         PlannedToolCall, ToolOutcome, SessionView, TurnResult, DialogueProvider
  privacy.py           deterministic PII mask applied at external-provider edges
  trace.py             bounded per-session event ledger (500 events, oldest dropped first)

  http/
    bootstrap.py       loads .env, builds the app, the SessionBoundary and per-session turn controllers
    routes.py          /api/config, /healthz, /api/sessions/... (typed turns, trace, handoff, reset)
    schemas.py         request bodies

  voice/               one WebSocket voice session
    protocol.py        every wire message name and error code (machine copy of VOICE_PROTOCOL.md)
    connection.py      socket loop + one handler per client message type
    asr_events.py      provider transcripts → versioned active utterance → transcript_* broadcasts
    evaluate.py        single evaluation slot; judge → COMMIT / HOLD / DEGRADED routing
    turn_control.py    turn_control emission, HOLD-deadline watcher, recovery offers
    dispatch.py        committed turn → `turn` result → streamed TTS audio; barge-in interrupt
    state.py           VoiceState (per-connection buffers, fences, timers)

  turns/
    controller.py      TurnController, TurnConfig, BaselineTurnJudge, episode/decision model
    prompts.py         shared judge state and the four Noul questions
    jev.py             TypeSafe Jev adapter (SDK)
    openrouter.py      OpenRouter Decisions adapter (HTTP, same questions)

  session/
    boundary.py        SessionBoundary: sessions, history, tool execution under customer scope
    tools.py           the 7 model-visible tool schemas and their argument contracts
    render.py          grounded answer text rendered from tool records

  providers/
    base.py            ProviderUnavailable, provider_status(), system prompt, message/tool shaping
    asr.py             Deepgram Nova-2 (WebSocket) and Vosk (offline) — same event shape
    llm.py             OpenAI Responses, chat-completions gateway, Anthropic fallback, cascade
    tts.py             ElevenLabs Flash v2.5 (WebSocket) and Piper (offline, in-process) — same PCM contract
    factories.py       select adapters from validated settings

  domain/
    seed.py            SeedOMS: orders, shipments, catalog, vouchers (fixtures, read-only)
    policy.py          deterministic lexical retrieval over the policy corpus
    data/policies.json the 9-clause fictional VoxCom policy corpus
```

## Typed request flow (REST)

```text
POST /api/sessions                 validate turn + provider settings → create session (CUST-001)
POST /api/sessions/{id}/turns      handle_turn: plan tools → run allowlisted read-only tools → grounded text
GET  /api/sessions/{id}/trace      bounded event ledger for the session
POST /api/sessions/{id}/handoff    record a support request (never a live transfer)
POST /api/sessions/{id}/reset      discard the session and create a fresh one
GET  /api/config                   provider configuration, never credentials or endpoint URLs
```

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

`plan_turn` returns every tool call the model emitted, in order. Adapters ask for at most one call per turn (`parallel_tool_calls=False`) and fall back gracefully when a gateway rejects the parameter; the boundary executes up to `MAX_TOOL_CALLS_PER_TURN = 3` calls sequentially and composes one answer (a recorded handoff wins outright; otherwise fragments join in order and the outcome is the first non-ANSWERED outcome).

## Data and honesty boundaries

- Seven read-only tools: `lookup_order_details`, `track_shipment_status`, `retrieve_policy_clauses`, `search_catalog`, `compare_products`, `apply_voucher`, `escalate_to_human`. Unknown tools are rejected; write tools do not exist.
- Order tools enforce the session customer server-side; another customer's order returns `OUT_OF_SCOPE` with no details.
- The model only plans; spoken text is rendered from tool records (`session/render.py`).
- `mask_pii` is applied to text sent to the LLM, the judge, and TTS, and to handoff/audit payloads. Stored history, live transcripts, and broadcasts keep the caller's raw words so the demo can show what was heard.
- Bounds: 200 live sessions (oldest-touched evicted), 24 history messages per session, 500 trace events per session, 64 KiB request bodies, 64 KiB audio frames, 24 MiB audio per voice session.
