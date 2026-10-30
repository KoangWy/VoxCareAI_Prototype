import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createPlaybackState, onAudioStart, onAudioEnd, stopPlayback, shouldPlayFrame,
  markSourceEnded, expectChunk, consumeChunkForBinary,
} from '../../static/audio-playback.js';
import { createVoiceState } from '../../static/voice-protocol.js';
function liveSource() {
  return { stopped: false, stop() { this.stopped = true; } };
}

function deadSource() {
  return { stop() { throw new Error('already ended'); } };
}

test('stale audio_end with a mismatched turn id is ignored', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T2', audio_chunk_id: 'C2' }, { nowMs: 1000 });
  const result = onAudioEnd(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 1100 });
  assert.deepEqual(result, { applied: false, stale: true });
  assert.equal(state.activeAudio.turnId, 'T2');
  assert.equal(state.activeAudio.streamEnded, false);
});

test('stale audio_end with a mismatched chunk id is ignored', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T1', audio_chunk_id: 'C2' }, { nowMs: 1000 });
  const result = onAudioEnd(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 1100 });
  assert.equal(result.stale, true);
  assert.ok(state.activeAudio);
});

test('matching audio_end marks the stream ended and clears it when idle', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 1000 });
  const result = onAudioEnd(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 2000 });
  assert.deepEqual(result, { applied: true, stale: false });
  assert.equal(state.activeAudio, null);
});

test('audio_end waits for pending sources before clearing', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 1000 });
  state.activeAudio.pendingSources = 1;
  onAudioEnd(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 2000 });
  assert.equal(state.activeAudio.streamEnded, true);
  const sources = new Set();
  const source = liveSource();
  sources.add(source);
  markSourceEnded(state, sources, source, { nowMs: 2100, turnId: 'T1' });
  assert.equal(state.activeAudio, null);
});

test('new audio_start replaces the prior stream and reports it for stopping', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 1000 });
  state.activeAudio.pendingSources = 2;
  const { replaced, next } = onAudioStart(state, { turn_id: 'T2', audio_chunk_id: 'C2' }, { nowMs: 2000 });
  assert.equal(replaced.turnId, 'T1');
  assert.equal(next.turnId, 'T2');
  assert.equal(next.pendingSources, 0);
  assert.equal(state.activeAudio.turnId, 'T2');
});

test('audio_end for the replaced turn can no longer clear the new stream', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 1000 });
  onAudioStart(state, { turn_id: 'T2', audio_chunk_id: 'C2' }, { nowMs: 2000 });
  const result = onAudioEnd(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 2100 });
  assert.equal(result.stale, true);
  assert.equal(state.activeAudio.turnId, 'T2');
});

test('stopPlayback stops every source, tolerates dead ones, and clears state', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 1000 });
  const sources = new Set([liveSource(), deadSource()]);
  const stopped = stopPlayback(state, sources, { nowMs: 1500 });
  assert.equal(stopped.length, 1);
  assert.equal(sources.size, 0);
  assert.equal(state.activeAudio, null);
  assert.ok(state.echoUntilMs >= 1500);
});

test('frames from a superseded generation must be dropped (ghost-audio guard)', () => {
  assert.equal(shouldPlayFrame(3, 4), false);
  assert.equal(shouldPlayFrame(4, 4), true);
});

test('source end for another turn leaves the current stream untouched', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T2', audio_chunk_id: 'C2' }, { nowMs: 1000 });
  state.activeAudio.pendingSources = 1;
  markSourceEnded(state, new Set(), liveSource(), { nowMs: 1100, turnId: 'T1' });
  assert.equal(state.activeAudio.pendingSources, 1);
});

test('audio_chunk envelope accepted for the live stream, rejected for a ghost turn', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 1000 });
  assert.equal(
    expectChunk(state, { turnId: 'T1', audioChunkId: 'C1', chunkIndex: 0, byteLength: 320 }).play,
    true,
  );
  const ghost = expectChunk(state, { turnId: 'T0', audioChunkId: 'C0', chunkIndex: 0, byteLength: 320 });
  assert.equal(ghost.play, false);
  assert.equal(ghost.reason, 'stale-chunk-ids');
  assert.equal(state.activeAudio.turnId, 'T1');
});

test('replayed chunk index is a ghost: drop it without touching the stream', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 1000 });
  expectChunk(state, { turnId: 'T1', audioChunkId: 'C1', chunkIndex: 0, byteLength: 320 });
  const replay = expectChunk(state, { turnId: 'T1', audioChunkId: 'C1', chunkIndex: 0, byteLength: 320 });
  assert.equal(replay.play, false);
  assert.equal(replay.reason, 'stale-chunk-index');
  assert.equal(state.activeAudio.turnId, 'T1');
});

test('binary batch plays only when generation is live and the envelope matches', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 1000 });
  const voice = createVoiceState();
  // Protocol 2: stashed envelope matches → play.
  voice.pendingChunk = { turnId: 'T1', audioChunkId: 'C1', chunkIndex: 0, byteLength: 320 };
  const live = consumeChunkForBinary(state, voice, 4, 4);
  assert.equal(live.play, true);
  assert.equal(voice.pendingChunk, null);
  // Protocol 2: envelope for a cancelled turn → ghost, drop it.
  const state2 = createPlaybackState();
  onAudioStart(state2, { turn_id: 'T2', audio_chunk_id: 'C2' }, { nowMs: 2000 });
  const voice2 = createVoiceState();
  voice2.pendingChunk = { turnId: 'T1', audioChunkId: 'C1', chunkIndex: 5, byteLength: 320 };
  const ghost = consumeChunkForBinary(state2, voice2, 4, 4);
  assert.equal(ghost.play, false);
  assert.equal(ghost.reason, 'stale-chunk-ids');
  // Protocol 2: stale generation → drop even with a matching envelope.
  const stale = consumeChunkForBinary(state, createVoiceState(), 3, 4);
  assert.equal(stale.play, false);
  assert.equal(stale.reason, 'stale-generation');
});

test('protocol 1 fallback: envelope-less binary frames keep playing (legacy output)', () => {
  const state = createPlaybackState();
  onAudioStart(state, { turn_id: 'T1', audio_chunk_id: 'C1' }, { nowMs: 1000 });
  const voice = createVoiceState(); // pendingChunk null: legacy server batch
  assert.equal(consumeChunkForBinary(state, voice, 4, 4).play, true);
});
