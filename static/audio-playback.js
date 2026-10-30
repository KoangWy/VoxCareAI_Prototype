// Browser playback queue: one assistant reply at a time, keyed by
// (turn_id, audio_chunk_id). Pure state machine — no DOM, no WebAudio — so
// the correlation rules are unit-testable in node. The DOM/WebAudio glue
// (AudioContext, BufferSource nodes) lives in app.js and calls into this.
//
// Rules:
//  - A new `audio_start` REPLACES any prior stream: the previous turn/chunk
//    is marked superseded and the caller must stop its sources.
//  - A stale `audio_end` (ids do not match the current stream) is ignored
//    and reported as stale instead of touching current state.
//  - Protocol 2 frames carry a turn/chunk envelope: `expectChunk()` validates
//    the stashed `audio_chunk` message against the live stream and reports
//    whether the next binary batch should play. A mismatch (cancelled turn
//    whose bytes are still in flight) means DROP the frame — the caller
//    must not schedule it (ghost-audio guard).
//  - Protocol 1 fallback: when `activeAudio.expectsChunk` is false (legacy
//    envelope-less frames), play every frame while the generation matches.
//    `shouldPlayFrame(frameGeneration, activeGeneration)` keeps the
//    generation gate contract the callers already use.

export function createPlaybackState() {
  return {
    activeAudio: null, // { turnId, audioChunkId, acknowledged, scheduled, pendingSources, streamEnded, expectsChunk, lastChunkIndex }
    echoUntilMs: 0,
  };
}

/**
 * Handle a server `audio_start`. Returns `{ replaced }` where `replaced` is
 * the superseded stream descriptor (or null) so the caller can stop its
 * sources. `echoTailMs` extends the echo gate from `nowMs`.
 */
export function onAudioStart(state, message, { echoTailMs = 600, nowMs = 0 } = {}) {
  const next = {
    turnId: message.turn_id ?? null,
    audioChunkId: message.audio_chunk_id ?? null,
    acknowledged: false,
    scheduled: false,
    pendingSources: 0,
    streamEnded: false,
    expectsChunk: false, // becomes true once the first audio_chunk envelope lands
    lastChunkIndex: -1,
  };
  const replaced = state.activeAudio;
  state.activeAudio = next;
  if (Number.isFinite(nowMs) && Number.isFinite(echoTailMs)) {
    state.echoUntilMs = Math.max(state.echoUntilMs, nowMs + echoTailMs);
  }
  return { replaced: replaced ?? null, next };
}

/**
 * Handle a server `audio_chunk` envelope (protocol 2, ahead of exactly one
 * binary batch). Returns `{ play, reason }`: `play` is true only when the
 * envelope belongs to the current stream, in which case the next binary
 * frame is expected to carry `byteLength` bytes. Anything else is a ghost
 * (cancelled turn in flight) — report the reason and do NOT schedule.
 */
export function expectChunk(state, envelope, { strictByteLength = false } = {}) {
  const current = state.activeAudio;
  if (!current) return { play: false, reason: 'no-active-audio' };
  const turnMatch = envelope.turnId === undefined || envelope.turnId === current.turnId;
  const chunkMatch = envelope.audioChunkId === undefined || envelope.audioChunkId === current.audioChunkId;
  if (!turnMatch || !chunkMatch) return { play: false, reason: 'stale-chunk-ids' };
  if (Number.isInteger(envelope.chunkIndex) && envelope.chunkIndex <= current.lastChunkIndex) {
    return { play: false, reason: 'stale-chunk-index' };
  }
  current.lastChunkIndex = Number.isInteger(envelope.chunkIndex) ? envelope.chunkIndex : current.lastChunkIndex;
  current.expectsChunk = true;
  const expectedBytes = Number.isInteger(envelope.byteLength) && envelope.byteLength >= 0
    ? envelope.byteLength
    : null;
  if (strictByteLength && expectedBytes === null) return { play: false, reason: 'chunk-missing-byte-length' };
  return { play: true, reason: null, expectedBytes };
}

/**
 * Handle a server `audio_end`. Stale ends (id mismatch) are ignored.
 * Returns `{ applied: boolean, stale: boolean }`.
 */
export function onAudioEnd(state, message, { echoTailMs = 600, nowMs = 0 } = {}) {
  if (Number.isFinite(nowMs) && Number.isFinite(echoTailMs)) {
    state.echoUntilMs = Math.max(state.echoUntilMs, nowMs + echoTailMs);
  }
  const current = state.activeAudio;
  if (!current) return { applied: false, stale: true };
  const turnMatch = message.turn_id === undefined || message.turn_id === current.turnId;
  const chunkMatch = message.audio_chunk_id === undefined || message.audio_chunk_id === current.audioChunkId;
  if (!turnMatch || !chunkMatch) return { applied: false, stale: true };
  current.streamEnded = true;
  if (current.pendingSources === 0) state.activeAudio = null;
  return { applied: true, stale: false };
}

/**
 * Consume a stashed `audio_chunk` envelope for the next binary PCM batch.
 * Returns `{ play, reason, expectedBytes }`: play only when the envelope
 * belongs to the live stream AND the generation matches. On `play: true`
 * the caller schedules the frame; anything else is a ghost — drop it and
 * trace `reason`. Clears the pending envelope either way.
 */
export function consumeChunkForBinary(state, voice, frameGeneration, activeGeneration) {
  const pending = voice.pendingChunk;
  voice.pendingChunk = null;
  if (frameGeneration !== activeGeneration) return { play: false, reason: 'stale-generation' };
  if (!state.activeAudio) return { play: false, reason: 'no-active-audio' };
  if (!pending) {
    // Protocol 1 fallback: legacy frames carry no envelope. Play them while
    // the stream and generation are live.
    return { play: true, reason: null };
  }
  return expectChunk(state, pending);
}
/**
 * Whether a binary PCM frame for `frameGeneration` may be scheduled under
 * the currently live generation. Frames from a superseded generation (a
 * cancelled turn whose bytes are still in flight) must be dropped.
 */
export function shouldPlayFrame(frameGeneration, activeGeneration) {
  return frameGeneration === activeGeneration;
}

/** Drop every source and clear the active stream. Returns stopped sources. */
export function stopPlayback(state, sources, { echoTailMs = 600, nowMs = 0 } = {}) {
  const stopped = [];
  for (const source of sources) {
    try {
      source.stop();
      stopped.push(source);
    } catch {
      /* already ended */
    }
  }
  sources.clear();
  state.activeAudio = null;
  if (Number.isFinite(nowMs) && Number.isFinite(echoTailMs)) {
    state.echoUntilMs = Math.max(state.echoUntilMs, nowMs + echoTailMs);
  }
  return stopped;
}


export function markSourceEnded(state, sources, source, { echoTailMs = 600, nowMs = 0, turnId } = {}) {
  sources.delete(source);
  if (Number.isFinite(nowMs) && Number.isFinite(echoTailMs)) {
    state.echoUntilMs = Math.max(state.echoUntilMs, nowMs + echoTailMs);
  }
  const current = state.activeAudio;
  if (current && current.turnId === turnId) {
    current.pendingSources = Math.max(0, current.pendingSources - 1);
    if (current.streamEnded && current.pendingSources === 0) state.activeAudio = null;
  }
}
