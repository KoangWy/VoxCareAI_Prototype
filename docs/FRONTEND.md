# Browser frontend

No build step: `static/index.html` loads `static/app.js` as a native ES module, and every other file in `static/` is a plain ES module served from `/static/`. There are no runtime dependencies and no CDN scripts (the stylesheet imports Google Fonts only). Audio is 16 kHz mono PCM16 in both directions.

The client implements the client half of [VOICE_PROTOCOL.md](VOICE_PROTOCOL.md). The `session` hello negotiates the protocol (`protocol: 2` enables `audio_chunk` envelopes and `interrupt`; anything else falls back to v1, where envelope-less binary frames simply play) and the barge-in mode; the observation log shows `Protocol v2 · barge-in <mode>`.

## Modules

| File | Owns |
| --- | --- |
| `app.js` | DOM wiring only: renders effects, owns the AudioContext / WebSocket / MediaStream handles, capture and playback glue, VAD tuning drawer |
| `voice-protocol.js` | Protocol surface: `PROTOCOL_VERSION`, `HANDLER_OWNER` (a client decision for every server message type), pure per-type reducers that never throw, handoff routing |
| `vad.js` | Adaptive SNR voice-activity gate (`LocalVad`), tuning bounds, echo gate, flag-aware onset routing (`routeVadOnset`), small shared helpers |
| `audio-playback.js` | Playback state: `audio_start` replacement, stale `audio_end` ignore, per-batch envelope validation (`expectChunk`, `consumeChunkForBinary`), generation guard |
| `audio-format.js` | Streaming 16 kHz resampler (anti-alias low-pass, state kept across frames), 20 ms PCM batcher, PCM16 decode |
| `mic-capture.js` | Mic start/stop state machine with an in-flight token |
| `mic-capture-node.js` | AudioWorklet capture with ScriptProcessor fallback, friendly microphone errors, granted-constraint logging |
| `pcm-worklet.js` | AudioWorklet processor: forwards raw Float32 frames to the main thread |
| `ws-client.js` | Close-code labels, auto-reconnect predicate, bounded backoff, socket lifecycle |

`tests/test_protocol_contract.py` parses `HANDLER_OWNER` out of `voice-protocol.js` and checks it against `src/voxcare/voice/protocol.py` in both directions: a new server message type fails the test until the client records a decision (`reducer` = handled in `reduceMessage`, `app` = streaming glue in `app.js`).

## Capture

Capture requests `echoCancellation`, `noiseSuppression`, and `autoGainControl` (mono) and logs which constraints the browser granted. AudioWorklet frames (128 samples at the device rate) go through one streaming resampler per capture session — filter state and read position carry across frames, so no sample is dropped at frame boundaries — and are sent as fixed 20 ms (640-byte) binary frames. Microphone failures show friendly messages (denied / no device / busy) and never leave the UI parked in "Listening".

## Voice activity detection

`LocalVad` tracks the room noise floor, nominates an onset only after ~110 ms of sustained above-gate energy, tolerates short dips inside a word, drops impulsive bursts (cough, click) locally, ignores onsets for a short refractory window after assistant playback starts, and reports `vad_quiet` after 650 ms of sustained quiet. The knobs (gate factors, hold, dip grace, refractory, quiet) are adjustable in the **Voice detection → Tune** drawer, shown at their current values, and persisted in `localStorage`.

VAD events are observations for the server's turn control. They never submit a turn, finish one, or request support on their own; only the visible recovery buttons send `manual_finish`, `caller_retry`, or `handoff_request`, and the microphone stays live while the caller chooses.

## Barge-in

Driven by the mode from the session hello:

- `off` (default, half-duplex): an onset while a reply plays, or within the 600 ms echo tail after it, is treated as speaker feedback and dropped (`Ignored VAD onset` in the log).
- `semantic`: the echo tail is still dropped, but an onset while a reply plays is sent with `agent_speaking: true` and playback continues until the server rules. An `interrupt` stops playback immediately and keeps the microphone open; a backchannel never interrupts.

## Playback

Each binary TTS batch must match the pending `audio_chunk` envelope (turn id, chunk id, increasing `chunk_index`); replays and foreign batches are dropped as ghosts and traced. A `byte_length` mismatch still plays and is traced. A new `audio_start` stops the previous reply's sources first; a stale `audio_end` is ignored; frames from a superseded generation never play.

## Recovery and lifecycle

The page keeps one generation counter per session. Reset increments it before stopping capture or requesting a new session, so callbacks from the old session are ignored. `STALE_RECOVERY_ACTION` and `WAIT_FOR_LOCAL_VAD_QUIET` rejections restore the recovery buttons and move `transcriptVersion` forward (never backward). The support button uses the WebSocket `handoff_request` while a hold episode is active and the HTTP handoff endpoint otherwise.

Socket drops show the close code and reason, reconnect with bounded exponential backoff (500 ms → 8 s, 5 attempts) and a fresh `start`, and otherwise offer one-click **Resume voice**. Closes 1008 (session audio budget) and 1009 (oversize frame) are terminal server decisions with friendly actions. Rapid mic toggles are serialized: the button is disabled during a transition and the loser of a race releases, never adopts, its stream, context, and socket.

## Timing honesty

The browser records its `performance.now` clock name and resolution; browser timestamps are never subtracted from server times. Playback acknowledgements report `acoustic_confirmed: false` because audio entering the output path does not prove the caller heard it, so the page keeps TTFA "unavailable". Policy evidence is shown with its fictional corpus, version, and clause ids; a support request is described as recorded, never as a transfer.

## Checks

```sh
node --test 'tests/frontend/*.test.mjs'   # headless tests: VAD, playback, mic races, protocol, socket/audio, barge-in, static
node scripts/check_static.mjs             # DOM ids and imports resolve; no innerHTML / CDN scripts / secrets
uv run pytest -q tests/test_protocol_contract.py
```

## Manual acceptance checklist

Run in Chrome and Safari, in a quiet room and then a noisy one, with DevTools open. "Log" means the page's observation log.

1. **Load** — `/` loads without console errors; the mic button is enabled only when providers are configured; the log shows `Voice protocol negotiated · v2 · barge-in off`; the Tune drawer opens and its sliders persist across a reload.
2. **Rapid toggles** — click the mic five times within ~300 ms: one transition at a time, no stuck "Starting…", no leaked microphone indicator.
3. **Capture** — start the mic: "Listening…" within 2 s and `Microphone capture attached · worklet` in the log; denying permission shows the friendly "denied" message and returns to idle.
4. **VAD** — one onset per utterance, no split on fricatives, a single cough or clap opens no turn, a quiet TV in the background does not nominate; `vad_quiet` ~650 ms after you stop.
5. **Order by voice** — say "Where is order O R D one zero zero two?": the transcript appears, the turn commits, and the answer names ORD-1002 and its ETA.
6. **Half-duplex** (`off`) — speak over a reply: the onset is logged as ignored echo and the reply keeps playing; after the echo tail, speech opens a new turn.
7. **Semantic barge-in** (`VOXCARE_BARGE_IN=semantic`, restart, reload) — "uh-huh" over a reply keeps it playing; "wait, stop" logs `Interrupted by caller · taking_floor`, stops audio at once, and keeps the mic open.
8. **Stop mid-reply** — pressing the mic during a reply stops audio immediately; restarting and speaking opens a normal new turn.
9. **Overlapping replies** — two quick typed requests: only the second is audible, with no tail of the first.
10. **Recovery** — let a turn degrade (start a sentence and stop mid-way): the three choices appear; **Try again** clears the transcript; double-clicking **Finish** restores the buttons instead of dead-ending.
11. **Support** — with choices visible, **Request support** resolves the episode over the socket; with none visible it records over HTTP; neither claims a transfer.
12. **Connection loss** — stop the server mid-call: the close reason and "Reconnecting…" appear; restarting the server reconnects, or **Resume voice** reconnects after the retries run out.
13. **Honesty** — TTFA stays "unavailable" everywhere, and every playback acknowledgement says acoustics are unconfirmed.
