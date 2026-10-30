# VoxCare AI

[![CI](https://github.com/KoangWy/VoxCareAI_Prototype/actions/workflows/ci.yml/badge.svg)](https://github.com/KoangWy/VoxCareAI_Prototype/actions/workflows/ci.yml)

English, browser-based **voice customer-support prototype** for the fictional electronics retailer **VoxCom**. A caller speaks in the browser; the server transcribes the speech, decides when the caller has finished their turn, lets a language model choose read-only lookup tools, and speaks back an answer that is rendered only from tool results.

Every customer, order, shipment, product, voucher and policy clause is fictional. The session is bound server-side to the simulated customer `CUST-001`; this is a demo, not production authentication.

## Features

- **Streaming voice loop** — browser microphone (Web Audio, AudioWorklet) → 16 kHz PCM over WebSocket → streaming ASR → turn decision → LLM tool planning → streaming TTS back to the browser.
- **Judged end-of-turn control** — a deterministic baseline, TypeSafe Jev, or an OpenRouter decisions model answers four questions (turn complete, holding the floor, addressed to the assistant, taking the floor). Code owns every action: silence alone never commits a turn, side-talk never reaches the assistant, and a stuck turn ends in an explicit caller choice (finish / try again / request support).
- **Grounded, read-only answers** — seven allowlisted tools: order lookup, shipment tracking, policy retrieval, catalog search, product comparison, voucher quote, and support-request recording. The model only plans tool calls; the spoken text is rendered from the tool records, so it cannot invent an order status, price, ETA or policy term.
- **Optional semantic barge-in** — speak over the assistant: a backchannel ("uh-huh") keeps the reply playing, a real floor grab ("wait, stop") interrupts it.
- **Two provider stacks, one contract** — the design providers (Deepgram, OpenAI, ElevenLabs) or a fully local demo stack (Vosk, Piper, any OpenAI-compatible gateway), selected by environment variables only.
- **Honest observability** — per-session trace (`GET /api/sessions/{id}/trace`), audit rows for tool calls and turn recovery, and no latency claim that the system cannot measure.

## How it works

```text
Browser (static/)                         Server (src/voxcare/)
─────────────────                         ─────────────────────
mic → AudioWorklet → resample 16 kHz ──►  voice/connection  ──► ASR (Deepgram | Vosk)
local VAD: vad_speech_start / vad_quiet ► voice/evaluate    ──► turns/ judge (baseline | Jev | OpenRouter)
                                                 │ COMMIT
                                                 ▼
                                          session/boundary  ──► LLM plans tools (OpenAI | gateway, Anthropic fallback)
                                                 │               └► read-only tools over domain/ fixtures
                                                 ▼ grounded text
speaker ◄── PCM16 playback ◄───────────── voice/dispatch    ──► TTS (ElevenLabs | Piper)
```

Typed text (`POST /api/sessions/{id}/turns`) goes through the same session boundary but skips the turn judge, since typed text has no microphone timing. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full flow.

## Quick start

Requirements: Python 3.12–3.14, [uv](https://docs.astral.sh/uv/), a current Chromium or Safari browser, and Node.js 22+ only for the frontend tests.

```sh
git clone https://github.com/KoangWy/VoxCareAI_Prototype.git
cd VoxCareAI_Prototype
uv sync --all-groups          # app + dev tools + the offline demo group (vosk, piper-tts)
cp .env.example .env          # then fill in the stack you want (below)
uv run uvicorn voxcare.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, allow microphone access, and press the microphone button. Without any provider configured the page still loads, reports which providers are missing, and never fakes an answer.

### Option A — local demo stack (no speech-provider keys)

Speech runs locally; only the LLM (and, optionally, the turn judge) needs a key.

```sh
uv run --group demo python scripts/fetch_demo_models.py   # Vosk + Piper voice into models/ (~130 MB, git-ignored)
```

Then in `.env`:

```sh
VOXCARE_ASR_PROVIDER=vosk
VOXCARE_VOSK_MODEL=models/vosk-model-small-en-us-0.15
VOXCARE_TTS_PROVIDER=piper
VOXCARE_PIPER_MODEL=models/en_US-lessac-medium.onnx
VOXCARE_OPENAI_TRANSPORT=chat
VOXCARE_OPENAI_BASE_URL=https://opencode.ai/zen/go/v1   # any OpenAI-compatible gateway
VOXCARE_OPENAI_MODEL=deepseek-v4.1-flash                # a tool-calling model id from the gateway's /v1/models
OPENAI_API_KEY=<gateway key>
# Optional judged turns via OpenRouter:
# VOXCARE_TURN_MODE=openrouter
# OPENROUTER_API_KEY=<openrouter key>
```

### Option B — design provider set

```sh
DEEPGRAM_API_KEY=...          # Nova-2 streaming ASR
OPENAI_API_KEY=...            # Responses API, model gpt-6-luna
ANTHROPIC_API_KEY=...         # optional sticky fallback (claude-haiku-5-5)
ELEVENLABS_API_KEY=...        # Flash v2.5 streaming TTS
ELEVENLABS_VOICE_ID=...
# VOXCARE_TURN_MODE=jev + TYPESAFE_API_KEY=... for TypeSafe Jev turn decisions
```

## Configuration

All settings are environment variables, read from `.env` at startup; [`.env.example`](.env.example) documents every one. They are validated in one place (`src/voxcare/config.py`): an invalid value fails session creation with the variable name (`TURN_CONFIG_INVALID:<VAR>` / `PROVIDER_CONFIG_INVALID:<VAR>`) instead of failing mid-call, and `GET /api/config` reports the problem without exposing credentials.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOXCARE_ASR_PROVIDER` | `deepgram` | `deepgram` or `vosk` (+ `VOXCARE_VOSK_MODEL`) |
| `VOXCARE_TTS_PROVIDER` | `elevenlabs` | `elevenlabs` or `piper` (+ `VOXCARE_PIPER_MODEL`) |
| `VOXCARE_OPENAI_TRANSPORT` | `responses` | `responses` (OpenAI Responses API) or `chat` (chat-completions gateway) |
| `VOXCARE_OPENAI_MODEL`, `VOXCARE_OPENAI_BASE_URL` | `gpt-6-luna`, — | Conversation model and endpoint |
| `VOXCARE_ANTHROPIC_MODEL` | `claude-haiku-5-5` | Fallback model after two consecutive retryable primary failures |
| `VOXCARE_TURN_MODE` | `baseline` | `baseline`, `jev` (TypeSafe), or `openrouter` |
| `VOXCARE_JEV_MODEL` | per mode | `jev-latest` or `inception/mercury-decide` |
| `VOXCARE_JEV_TIMEOUT_MS`, `VOXCARE_HOLD_DEADLINE_MS` | 300 / 3000 (OpenRouter: 2500 / 9000) | Judge budget and HOLD deadline |
| `VOXCARE_BARGE_IN` | `off` | `off` (half-duplex) or `semantic` |
| `VOXCARE_LOG_LEVEL` | `INFO` | Python log level |

Turn thresholds and silence windows (`VOXCARE_TURN_COMPLETE_THRESHOLD`, `VOXCARE_EOS_*`, …) are documented in [docs/TURN_CONTROL.md](docs/TURN_CONTROL.md). Variables from the host that are not prefixed `VOXCARE_` (for example `OPENAI_MODEL`) are ignored on purpose.

## Using the demo

Try questions such as:

| Ask | What happens |
| --- | --- |
| "Where is order ORD-1002 and when will it arrive?" | Order + shipment lookup: shipped with GHN, carrier ETA 2026-10-15 |
| "What's the status of ORD-1001?" | Confirmed; no shipment or ETA yet |
| "Check order ORD-2002" | Refused: belongs to another (simulated) customer |
| "Do you have Samsung TVs under 15 million VND?" | Catalog search |
| "Compare TV-SAM-55 and TV-LG-55" | Side-by-side on recorded attributes only |
| "Can I use VOXCOM_500K on a 12,000,000 VND TV?" | Voucher quote (read-only; nothing is changed) |
| "What is the return window for my LG TV?" | Fictional policy clauses with their IDs |
| "I want to talk to a person" | Support request recorded; no live transfer happens |

Seeded orders: `ORD-1001` confirmed, `ORD-1002` shipped, `ORD-1003` delivered, `ORD-2002` another customer, `ORD-9999` does not exist. The server accepts variants like `ord 1002` or `ORD1002`, and the model is instructed to convert spelled-out codes from speech recognition ("o r d one zero zero two") to the `ORD-####` form.

**Reset session** starts a fresh simulated session and fences every late result from the old one. The text box under the conversation sends the same request path without the microphone.

## Development

```sh
uv run pytest -q                          # backend unit + WebSocket tests (fakes only, no network)
uv run ruff check . && uv run ruff format --check .
node --test 'tests/frontend/*.test.mjs'   # headless frontend tests (or: npm test)
node scripts/check_static.mjs             # DOM ids / imports / static-asset constraints
```

The test suite clears every `VOXCARE_*` variable and provider key before each test, so it never reaches a real provider. Live checks against real providers are separate scripts; see [docs/TESTING.md](docs/TESTING.md).

## Project structure

```text
src/voxcare/
  app.py            ASGI entry point (voxcare.app:app)
  config.py         every VOXCARE_* setting, validated
  contracts.py      shared dataclasses and the DialogueProvider protocol
  privacy.py        pattern-based PII mask for external-provider edges
  trace.py          bounded per-session event ledger
  http/             FastAPI app factory and REST routes
  voice/            WebSocket voice flow (connection, ASR fan-in, evaluation, dispatch, wire protocol)
  turns/            turn controller and judge adapters (baseline, TypeSafe Jev, OpenRouter)
  session/          session boundary, tool registry, grounded answer rendering
  providers/        ASR, LLM, and TTS adapters + factories
  domain/           fictional OMS fixtures and policy corpus
static/             browser client (plain ES modules, no build step)
tests/              pytest suite; tests/frontend/ holds the node tests
scripts/            model download, live end-to-end harnesses, evaluation tooling
evaluation/         controlled turn-decision case definitions
docs/               architecture, protocol, turn control, frontend, testing, policy sources
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — module map, request and voice flows, provider matrix
- [Voice protocol](docs/VOICE_PROTOCOL.md) — the authoritative browser ↔ server WebSocket contract
- [Turn control](docs/TURN_CONTROL.md) — the judged end-of-turn state machine, budgets, thresholds, evidence
- [Frontend](docs/FRONTEND.md) — browser modules, VAD, playback, recovery, manual acceptance checklist
- [Testing](docs/TESTING.md) — automated checks, live harnesses, evaluation workflow
- [Policy sources](docs/POLICY_SOURCES.md) — the fictional policy corpus and its provenance

## Limitations

- English only; simulated identity; fictional, read-only business data. No business record can be changed and no live human transfer exists.
- The local demo stack is a documented deviation from the design provider set. Latency and accuracy are **not measured** on either stack; the design targets (e.g. 755 ms time-to-first-audio, ≥85 % accuracy) are not claimed.
- PII masking at the provider edges is pattern-based and incomplete; stored history and live transcripts keep the caller's raw words. Use fictional inputs only.
- Semantic barge-in is verified with the headless live harness, not with a physical microphone, so it stays off by default.

## License

No license has been chosen yet; all rights are reserved by the author until one is added.
