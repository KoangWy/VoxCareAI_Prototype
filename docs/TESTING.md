# Testing

## Automated checks

Run before every change; CI (`.github/workflows/ci.yml`) runs the same set.

```sh
uv sync --all-groups
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
node --test 'tests/frontend/*.test.mjs'
node scripts/check_static.mjs
```

| Suite | Covers |
| --- | --- |
| `tests/test_turns.py`, `tests/test_turn_control.py` | turn controller: commit rule, budgets, deadlines, re-arming, stale-result quarantine, barge-in scoring |
| `tests/test_session_boundary.py`, `tests/test_catalog_tools.py`, `tests/test_policy*.py`, `tests/test_seed_fixtures.py` | tool registry, customer scoping, grounded rendering, fixtures, policy retrieval |
| `tests/test_voice_recovery_protocol.py`, `tests/test_voice_barge_in.py` | the real WebSocket flow with fake ASR/TTS/LLM/judges: recovery, re-judging, barge-in, close codes, deadline watcher |
| `tests/test_protocol_contract.py` | server ↔ client message-name contract |
| `tests/test_*adapters.py`, `tests/test_provider_cascade.py`, `tests/test_jev.py`, `tests/test_openrouter_judge.py` | provider adapters and fallback, with fakes only |
| `tests/test_config.py`, `tests/test_http_api.py`, `tests/test_privacy.py` | configuration validation, REST surface, PII mask |
| `tests/frontend/*.test.mjs` | VAD on synthetic clips, playback guards, mic races, reducers, reconnect, resampling, barge-in routing |

`tests/conftest.py` removes every `VOXCARE_*` variable and provider key before each test, so the suite never depends on a developer's `.env` and never reaches a real provider.

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
