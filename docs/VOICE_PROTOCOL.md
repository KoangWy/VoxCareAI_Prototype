# Voice protocol

Authoritative contract for the browser ↔ server voice loop. `src/voxcare/voice/protocol.py`
is the machine-readable copy of every name below; `tests/test_voice_barge_in.py` fails if
the server code emits a name that is missing there or from this file, and
`tests/test_protocol_contract.py` fails if `static/voice-protocol.js` has no client
decision for a server message type.

Transport: one WebSocket per voice session,
`GET /ws/sessions/{session_id}` (HTTP route that upgrades). Audio frames are
raw binary PCM16 mono at 16 kHz. Control messages are JSON text frames in
either direction. Unknown `type` values MUST be ignored (never throw).

`protocol` version: the `session` hello carries `"protocol": 2`.
Protocol 2 adds, per TTS chunk, a JSON `audio_chunk` envelope ahead of each
binary PCM batch, and the server→client `interrupt` message. Protocol 1
clients (no envelope handling) keep working: they ignore the extra JSON
message and play every binary frame.

Typed turns (`POST /api/sessions/{id}/turns`) bypass the turn judge by
design: caller-typed text has no microphone timing, so there is nothing to
judge. Only the WebSocket voice path is judged.

## 1. Server → client

| `type` | fields | meaning |
|---|---|---|
| `session` | `session` (SessionView), `providers` (`/api/config` shape), `protocol: 2`, `barge_in` (`"off"`\|`"semantic"`) | hello after connect + accept; the client adopts the barge-in mode from it. |
| `voice_started` | `provider`, `locale: "en-US"` | ASR attached; client may stream audio. |
| `transcript_partial` | `session_id`, `turn_id`, `transcript_version`, `sequence`, `text`, `is_final`, `speech_final` | ASR interim or non-endpointed text. |
| `transcript_final` | same as partial | endpointed text (`speech_final=true`). |
| `turn_control` | `session_id`, `turn_id`, `turn_epoch`, `transcript_version`, `decision_request_id`, `hold_episode_id`, `state` (`COMMIT`\|`HOLD`\|`DEGRADED`\|`FALLBACK_RESOLVED`), `reason`, `jeval_noncommit_count`, `hold_elapsed_ms`, `hold_deadline_ms` (null until the deadline arms), `turn_mode`, `scores` (`turn_complete`, `holding_floor`, `addressed_to_agent`, `taking_floor` or null), `should_interrupt`, `timing` (T0–T6, see §4) | turn verdict for one transcript version. |
| `recovery_required` | `session_id`, `turn_id`, `turn_epoch`, `transcript_version`, `hold_episode_id`, `choices: ["manual_finish","caller_retry","handoff_request"]` | episode degraded; caller must pick. |
| `turn` | `result` (typed-turn payload + `timing`) | committed turn dispatched to the dialogue provider. |
| `audio_chunk` | `turn_id`, `audio_chunk_id`, `chunk_index` (0-based), `byte_length`, `protocol: 2` | precedes exactly one binary PCM batch of this chunk. New in protocol 2. |
| binary PCM16 | — | one TTS batch; belongs to the most recent `audio_chunk` envelope. Frames without a preceding envelope are legacy protocol 1 output — play them. |
| `audio_start` | `sample_rate: 16000`, `turn_id`, `audio_chunk_id` | TTS stream started for the turn. |
| `audio_end` | `turn_id`, `audio_chunk_id` | TTS stream finished for the chunk. Apply only when both ids match the current audio; otherwise ignore as stale. |
| `interrupt` | `session_id`, `turn_id`, `turn_epoch`, `hold_episode_id`, `reason: "taking_floor"`, `taking_floor` | semantic barge-in only: stop playback now, keep the mic open. Never sent when `VOXCARE_BARGE_IN=off` (default). |
| `retry_ready` | `turn_id`, `transcript_version`, `hold_episode_id` | caller-retry accepted; the degraded words are discarded — speak the turn again. |
| `handoff_request_recorded` | `message` | support request recorded; no live transfer (demo). |
| `provider_unavailable` | `provider` (`asr`\|`tts`\|`turn_decision`\|`conversation`), `error` (code below) | provider missing/failed; never treated as success. `turn_decision` errors: `TYPESAFE_API_KEY_NOT_CONFIGURED`, `OPENROUTER_API_KEY_NOT_CONFIGURED`, `TURN_MODE_INVALID`. `conversation` errors: `CONVERSATION_TIMEOUT` (judged voice turn's dialogue call exceeded the 30 s host bound after COMMIT; no `turn` follows for that episode). |
| `playback_ack_recorded` | `turn_id`, `ttfa_ms` (null unless clocks align), `ttfa_status` | echo of a `playback_started` ack. `acoustic_confirmed` is always false in this demo, so TTFA is structurally unavailable (see §4). |
| `playback_failed_recorded` | `turn_id` | echo of a `playback_failed` report. |
| `voice_stopping` | — | ack of client `stop`. |
| `session_reset` | `session` (new SessionView) | terminal for this socket; open a new socket for the new session id. |
| `error` | `code` (§3), sometimes `expected` | recoverable control error; the socket stays open, except the two close-code cases in §3. |

## 2. Client → server

| `type` | fields | meaning |
|---|---|---|
| `start` | `sample_rate: 16000`, `channels: 1`, `encoding: "pcm_s16le"`, optional `client_clock: {name, resolution_ms}` | attach ASR/TTS. Anything else → `UNSUPPORTED_AUDIO_FORMAT`, socket stays open. |
| binary PCM16 | — | mic audio; requires `start` first (`SEND_START_FIRST`). >64 KiB in one frame → `AUDIO_FRAME_TOO_LARGE` + close 1009. >24 MiB per session → `SESSION_AUDIO_LIMIT_REACHED` + close 1008. |
| `vad_speech_start` | `segment_id`, `client_monotonic_ms`, optional `agent_speaking` (bool: assistant audio audible client-side) | local-VAD onset. Resets the non-commit budget (a degraded episode is replaced by a fresh one) and cancels in-flight judgments. With `VOXCARE_BARGE_IN=semantic` and `agent_speaking: true`, the onset does NOT stop audio — even if the server has already sent the whole reply: the server holds (`barge_in_held`) and judges `taking_floor` on the following words; only `interrupt` stops playback. Default `off`: any onset retires the reply and in-flight work immediately. |
| `vad_quiet` | `segment_id`, `client_monotonic_ms`, `speech_end_monotonic_ms`, `silence_ms` (number, fractional ok — server rounds) | local-VAD quiet. Schedules a judgment; later judgments see `silence_ms` plus the server time elapsed since this message. |
| `manual_finish` | `transcript_version`, `turn_epoch`, `hold_episode_id` | commit the heard transcript now (caller-controlled end of turn, only from DEGRADED). Version mismatch → `STALE_RECOVERY_ACTION`; not quiet → `WAIT_FOR_LOCAL_VAD_QUIET`; no degraded episode → `FALLBACK_REQUIRES_DEGRADED_STATE`. |
| `caller_retry` | same as `manual_finish` | discard the degraded episode and its transcript, open a fresh episode (same epoch). Same errors. |
| `handoff_request` | same as `manual_finish` + optional `reason` | record a support request and resolve the hold episode. When no hold episode is active the HTTP `POST /api/sessions/{id}/handoff` path is equivalent. |
| `playback_started` | `turn_id`, `audio_chunk_id`, `client_monotonic_ms`, `clock_name`, `acoustic_confirmed: false` | browser output started (never an acoustic claim). Ids must match the active audio or the ack is recorded `stale_playback_ack` and dropped. |
| `playback_failed` | `turn_id`, `error_code` | browser output failed. |
| `stop` | — | finalize ASR; server replies `voice_stopping`. |
| `reset` | — | new session; server replies `session_reset`, then the socket ends. |

## 3. Error codes and close codes

`error` codes: `SEND_START_FIRST`, `AUDIO_FRAME_TOO_LARGE`,
`SESSION_AUDIO_LIMIT_REACHED`, `INVALID_CONTROL_MESSAGE`,
`UNSUPPORTED_AUDIO_FORMAT`, `STALE_RECOVERY_ACTION`, `SESSION_EXPIRED`,
`WAIT_FOR_LOCAL_VAD_QUIET`, `FALLBACK_REQUIRES_DEGRADED_STATE`,
`UNKNOWN_CONTROL_MESSAGE`.

WebSocket close codes sent by the server (an `error` message goes first):

| code | when |
|---|---|
| 4404 | unknown/inactive session id at connect. |
| 1013 | ASR provider not configured (`start` cannot attach). |
| 1009 | single audio frame > 64 KiB. |
| 1008 | session audio budget (> 24 MiB) exceeded. |

HTTP `POST /api/sessions` fails with 500
`{"detail": "TURN_CONFIG_INVALID: <field> <rule>"}` when a `VOXCARE_*`
turn variable is invalid (see `docs/TURN_CONTROL.md`). `GET /api/config`
never throws on bad turn config: it reports
`providers.turn_decision = {"name": "Turn configuration error",
"mode": <raw>, "configured": false, "error": <code>}` — never
"Application baseline".

## 4. Timing fields (T0–T6, best-effort observations)

`turn_control.timing` and `turn.result.timing` carry the same six slots:
T0 client speech end (local VAD), T1 ASR final observed (server monotonic),
T2 turn decision available (server monotonic), T3 provider first token
(unavailable — not instrumented), T4 text sent to TTS, T5 TTS first chunk,
T6 browser playback ack. Each slot is
`{"status": "available"|"pending"|"unavailable", ...}` plus `ms` or
`client_monotonic_ms` when available, and a `reason` otherwise.
`ttfa_ms` is always null: the client sends `acoustic_confirmed: false`
and T0/T6 share no aligned clock, so no time-to-first-audio claim is made.

## 5. Barge-in modes (`VOXCARE_BARGE_IN`, default `off`)

- `off`: any `vad_speech_start` cancels in-flight TTS and deadlines at
  once; the browser additionally drops onsets during playback as speaker
  echo (half-duplex gate).
- `semantic`: speech onset during assistant playback holds audio and judges
  `taking_floor` (threshold `VOXCARE_TAKING_FLOOR_THRESHOLD`, default 0.5)
  with `assistant_current_speech` in the judge state. Backchannel
  (`taking_floor` below bar) keeps audio, reports HOLD `backchannel_hold`, and
  is then discarded — it never becomes a turn or a recovery prompt. A real
  floor grab sends `interrupt`, cancels TTS, and the words get one normal
  judgment as the caller's new turn. Plain words do not interrupt: only a
  `taking_floor` verdict at/above the bar does.
