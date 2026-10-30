# Turn control

How the voice path decides that the caller has finished speaking. The implementation is `src/voxcare/turns/controller.py` (state machine), `src/voxcare/turns/prompts.py` (judge state and questions), and `src/voxcare/voice/evaluate.py` (when judgments run).

## Status

The controller is covered by deterministic tests with injected judgments, and the full voice path has been exercised headlessly against real providers (see [Evidence](#evidence)). No physical-microphone acceptance run, no N≥30-per-intent evaluation, and no latency benchmark exists. Do not quote the design targets (TTFA 755 ms, accuracy ≥85 %) as results of this prototype.

## The judge contract

A judge sees text and timing only — never audio: the live transcript, recent conversation, the assistant's last message, mic silence since the last word, ASR finality and stability, the transcript's recent growth, and elapsed turn time. It answers four yes/no ("Noul") questions as probabilities:

| Question | Meaning |
| --- | --- |
| `turn_complete` | the caller has handed the floor to the assistant |
| `holding_floor` | the latest words ask for more time ("let me think", "hold on") |
| `addressed_to_agent` | the words are meant for the assistant, not someone in the room |
| `taking_floor` | asked only while the assistant is speaking: a real interruption, not a backchannel |

Three judges implement it:

- `BaselineTurnJudge` (default) — deterministic: complete once ASR is final and the mic has been quiet ≥500 ms.
- `TypeSafeJevJudge` (`VOXCARE_TURN_MODE=jev`) — TypeSafe SDK `AsyncTypeSafeClient.system_one`, retries disabled, `TYPESAFE_API_KEY`.
- `OpenRouterDecisionsJudge` (`VOXCARE_TURN_MODE=openrouter`) — `POST https://openrouter.ai/api/alpha/decisions` with the same state and questions, `OPENROUTER_API_KEY`, default model `inception/mercury-decide`. One HTTP client is shared by all sessions and closed at shutdown.

A missing key or SDK is a configuration error (`jev_unconfigured`), distinct from a provider failure (`jev_error`); 429/5xx map to `retryable_provider_error` and timeouts to `jev_timeout`. Provider response text is never logged.

## When a turn commits

Code owns every action. A judgment yields `COMMIT` only when all of these hold:

1. local VAD reports quiet and the measured mic silence is positive;
2. the transcript is stable (ASR final, or ≥90 % of words no longer revised);
3. `holding_floor` is below its bar, or `turn_complete` clears the full bar anyway;
4. `turn_complete` clears a **silence-relaxed bar**: 0.72, sliding linearly to 0.55 as quiet grows to 3000 ms (no relaxation while the caller holds the floor in words);
5. `addressed_to_agent` clears its bar — complete side-talk never reaches the assistant;
6. the assistant is not speaking (speech heard during playback is judged as a possible interruption instead);
7. the mic has been quiet for a **confirm window** scaled by confidence: 350 ms for a certain verdict up to 1000 ms for a borderline one.

Sustained silence is evidence too: past 5 s of quiet (12 s while the caller holds the floor) the turn commits if it is addressed and not held; otherwise the verdict counts as a non-commit.

A complete-looking sentence alone never commits, and a timeout, error, or deadline never commits.

## Episodes, budget, and deadline

An episode opens at a local-VAD onset. Its absolute HOLD deadline is anchored at the first quiet+final observation (not at onset, so ASR and judge latency do not eat it) and re-armed on every non-commit verdict observed during quiet. Each non-commit verdict counts toward a fixed budget of **two**; the second one — or the deadline — degrades the episode:

- the controller returns `DEGRADED` and records `TURN_COMPLETION_DEGRADED` (`outcome_status=PENDING`);
- once the caller is quiet the server sends `recovery_required` with three explicit choices: `manual_finish` (submit the heard words), `caller_retry` (discard them and listen again), `handoff_request` (record a support request);
- simply speaking again also works: a new local-VAD onset replaces the degraded episode with a fresh one.

Only a local-VAD onset (`source="local_vad"`) resets the budget; ASR partials, timers, timeouts, and provider callbacks never do. An empty transcript (the VAD opened on noise) holds without spending budget.

Every judgment carries a `decision_request_id`; a result applies only while `(session_id, turn_epoch, hold_episode_id, decision_request_id)` still identifies the active request. New speech, new text, a retry, or a reset quarantines late results. The host's deadline watcher follows the controller's deadline as it is re-armed, so a HOLD episode always ends in a verdict or a caller action.

## When judgments run

`voice/evaluate.py` keeps one evaluation slot per connection — at most one judgment in flight and one pending on a timer:

| Trigger | Delay |
| --- | --- |
| `vad_quiet` with an empty transcript | immediately (arms the episode) |
| `vad_quiet` with final ASR text | 150 ms debounce |
| `vad_quiet` with non-final text | 700 ms, giving ASR finality a chance to land first |
| ASR final while quiet | 150 ms debounce; replaces an in-flight judgment of older text |
| HOLD for `turn_not_complete`, `confirming`, `asr_not_stable`, `missing_mic_silence`, `jev_timeout`, `jev_error` | one re-judge after 1000 ms more quiet (for `confirming`: the rest of the confirm window) |

Holds for `caller_holding_floor`, side-talk, and backchannels are not re-judged — the same words would get the same answer; new speech or the deadline moves them on. Mic silence is the browser's `silence_ms` at quiet plus the server time elapsed since, so later judgments see the silence grow.

## Barge-in

With `VOXCARE_BARGE_IN=semantic`, an onset the browser reports while the reply is audible (`agent_speaking: true`) keeps the audio playing and the words are judged against the assistant's speech. A `taking_floor` verdict at or above 0.5 sends `interrupt`, cancels TTS, and the words then get one normal judgment as the caller's new turn. A backchannel holds (`backchannel_hold`) and is discarded, so it can never turn into a recovery prompt. In the default `off` mode any onset retires the reply, and the browser drops onsets during playback as speaker echo.

## Settings

All parsed and validated in `src/voxcare/config.py`; invalid values fail session creation with the variable name.

| Setting | Variable | Baseline / Jev | OpenRouter |
| --- | --- | ---: | ---: |
| Judge budget | `VOXCARE_JEV_TIMEOUT_MS` | 300 ms | 2500 ms |
| HOLD deadline | `VOXCARE_HOLD_DEADLINE_MS` | 3000 ms | 9000 ms |
| Non-commit budget | fixed | 2 | 2 |
| `turn_complete` bar | `VOXCARE_TURN_COMPLETE_THRESHOLD` | 0.72 | 0.72 |
| Relaxed floor | `VOXCARE_EOS_THRESHOLD_FLOOR` | 0.55 | 0.55 |
| Relaxation span | `VOXCARE_EOS_DECAY_MS` | 3000 ms | 3000 ms |
| `holding_floor` bar | `VOXCARE_HOLDING_FLOOR_THRESHOLD` | 0.50 | 0.50 |
| `addressed_to_agent` bar | `VOXCARE_ADDRESSED_TO_AGENT_THRESHOLD` | 0.50 | 0.50 |
| `taking_floor` bar | `VOXCARE_TAKING_FLOOR_THRESHOLD` | 0.50 | 0.50 |
| Stable-word floor | `VOXCARE_MIN_STABLE_WORD_RATIO` | 0.90 | 0.90 |
| Confirm window | `VOXCARE_EOS_CONFIRM_QUIET_MIN_MS` / `_MAX_MS` | 350–1000 ms | 350–1000 ms |
| Silence caps | `VOXCARE_EOS_MAX_SILENCE_MS` / `_HOLDING_MS` | 5000 / 12000 ms | 5000 / 12000 ms |

The OpenRouter budget and deadline are larger because the route measures 1–2 s per judgment (with occasional stalls past 1.5 s), and the first judgment can only start after VAD quiet plus ASR finality. These values are starting assumptions, not calibrated targets.

## Controlled cases and evaluation

`evaluation/jev_cases.json` defines six paired case families: a complete short answer, an unfinished clause, hesitation then continuation, a complete sentence followed by announced extra content, side-talk vs addressed speech, and backchannel vs a real interruption. The tests in `tests/test_turns.py` and `tests/test_turn_control.py` check controller outcomes for them with injected judgments — which proves controller behaviour, not judge quality.

After real runs, record one JSON object per case and mode with matching `condition_id` and `transcript_sha256`, and compare:

```sh
uv run python scripts/evaluate_turn_cases.py --input observed-results.jsonl
```

With no input the script reports that no accuracy or latency result is available.

## Evidence

Small-N functional checks, not benchmarks.

**Judge calibration** (`scripts/live_mercury_check.py`, OpenRouter Mercury Decide, thresholds 0.72 / 0.50):

| Transcript | `turn_complete` | `holding_floor` | Latency | Outcome |
| --- | ---: | ---: | ---: | --- |
| "Web." | 0.9964 | 0.0001 | 1570 ms | COMMIT (expected) |
| "Web, and" | 0.0002 | 0.0124 | 935 ms | HOLD (expected) |
| "Let me think" | 0.0003 | 0.9999 | 1894 ms | HOLD (expected) |
| "It is ORD-1002" (after the hesitation) | 0.9992 | 0.0000 | 1276 ms | COMMIT (expected) |
| "I have a few questions." | 0.0067 | 0.0086 | 936 ms | HOLD (expected) |
| "First, can you check the delivery date?" | 0.0474 | 0.0008 | 1025 ms | HOLD — conservative on announced continuations |

**Voice end to end** (`scripts/live_voice_e2e.py`: Piper-synthesised speech → Vosk → Mercury → gateway LLM → Piper): a plain policy question commits (`turn_complete` ≈ 0.99) and is answered with spoken audio; "Let me think." holds and the continuation commits; the assistant's own audio fed back as mic input commits no turn; with semantic barge-in a backchannel keeps the reply playing and "Stop! I need to change my order right now." interrupts it with a `taking_floor` verdict. Mis-recognitions by the small Vosk model (e.g. "what does the return window for my tv") can legitimately be judged incomplete; rerun before treating one failure as a regression.

## Known limits

- The small Vosk model (`vosk-model-small-en-us-0.15`) hears spelled order codes as words; the LLM is instructed to normalise them, which works in typed checks but has not been measured on audio.
- Thresholds come from a reference implementation and the measurements above; they need tuning on real calls.
- The judge cannot hear prosody; trailing intonation and breathing are invisible to it.
