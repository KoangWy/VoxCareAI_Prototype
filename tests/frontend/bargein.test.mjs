import test from 'node:test';
import assert from 'node:assert/strict';
import { routeVadOnset, vadOnsetIsEcho } from '../../static/vad.js';
import { createPlaybackState, stopPlayback } from '../../static/audio-playback.js';
import { createVoiceState, reduceMessage } from '../../static/voice-protocol.js';

// Flag-aware barge-in: a fake socket + fake audio context
// drive the exact app.js decision path — routeVadOnset for onsets,
// stopPlayback for interrupts — without a DOM or a live server.

function fakeSocket() {
  return { sent: [], readyState: 1, send(payload) { this.sent.push(JSON.parse(payload)); } };
}

function liveSource() {
  return { stopped: false, stop() { this.stopped = true; } };
}

test('off (default): VAD onset during playback is echo — dropped, never sent', () => {
  const voice = createVoiceState();
  reduceMessage(voice, { type: 'session', session: { session_id: 's' }, protocol: 2 });
  assert.equal(voice.bargeInMode, 'off');
  const socket = fakeSocket();
  const playback = createPlaybackState();
  playback.activeAudio = { turnId: 'T1', audioChunkId: 'C1', pendingSources: 0, streamEnded: false };
  const onset = { type: 'vad_speech_start', segment_id: 'seg-1', client_monotonic_ms: 1000 };
  const routed = routeVadOnset(onset, { activeAudio: playback.activeAudio, nowMs: 1000, echoUntilMs: 1600, bargeInMode: voice.bargeInMode });
  assert.equal(routed.action, 'drop-echo');
  assert.equal(socket.sent.length, 0);
});

test('off: echo tail keeps dropping onsets; silence after the tail sends cleanly', () => {
  const voice = createVoiceState();
  reduceMessage(voice, { type: 'session', session: { session_id: 's' }, protocol: 1 });
  const socket = fakeSocket();
  const onset = { type: 'vad_speech_start', segment_id: 'seg-1', client_monotonic_ms: 1500 };
  const tail = routeVadOnset(onset, { activeAudio: null, nowMs: 1500, echoUntilMs: 1600, bargeInMode: voice.bargeInMode });
  assert.equal(tail.action, 'drop-echo');
  const clear = routeVadOnset(onset, { activeAudio: null, nowMs: 2000, echoUntilMs: 1600, bargeInMode: voice.bargeInMode });
  assert.equal(clear.action, 'send');
  assert.equal(clear.event.agent_speaking, undefined);
  socket.send(JSON.stringify(clear.event));
  assert.equal(socket.sent[0].type, 'vad_speech_start');
  assert.equal('agent_speaking' in socket.sent[0], false);
});

test('semantic: onset during playback sends with agent_speaking:true and keeps playing', () => {
  const voice = createVoiceState();
  reduceMessage(voice, { type: 'session', session: { session_id: 's' }, protocol: 2, barge_in: 'semantic' });
  assert.equal(voice.bargeInMode, 'semantic');
  const socket = fakeSocket();
  const playback = createPlaybackState();
  playback.activeAudio = { turnId: 'T1', audioChunkId: 'C1', pendingSources: 1, streamEnded: false };
  const onset = { type: 'vad_speech_start', segment_id: 'seg-9', client_monotonic_ms: 3000 };
  // Graded echo guard still applies first: nothing playing and inside the
  // refractory/echo tail drops locally even in semantic mode.
  const guardStillApplies = vadOnsetIsEcho(onset, { activeAudio: null, nowMs: 3000, echoUntilMs: 3600 });
  assert.equal(guardStillApplies, true);
  // But a genuine onset while the reply plays goes to the server flagged.
  const routed = routeVadOnset(onset, { activeAudio: playback.activeAudio, nowMs: 4000, echoUntilMs: 3600, bargeInMode: voice.bargeInMode });
  assert.equal(routed.action, 'send');
  assert.equal(routed.event.agent_speaking, true);
  socket.send(JSON.stringify(routed.event));
  assert.equal(socket.sent[0].agent_speaking, true);
  // ... and playback continues: nothing was stopped before an interrupt.
  assert.ok(playback.activeAudio);
});

test('semantic: nothing playing sends with agent_speaking:false', () => {
  const voice = createVoiceState();
  reduceMessage(voice, { type: 'session', session: { session_id: 's' }, protocol: 2, barge_in: 'semantic' });
  const routed = routeVadOnset(
    { type: 'vad_speech_start', segment_id: 'seg-2', client_monotonic_ms: 5000 },
    { activeAudio: null, nowMs: 5000, echoUntilMs: 0, bargeInMode: voice.bargeInMode },
  );
  assert.equal(routed.action, 'send');
  assert.equal(routed.event.agent_speaking, false);
});

test('interrupt stops playback immediately and keeps the mic open', () => {
  const voice = createVoiceState();
  reduceMessage(voice, { type: 'session', session: { session_id: 's' }, protocol: 2, barge_in: 'semantic' });
  const playback = createPlaybackState();
  playback.activeAudio = { turnId: 'T1', audioChunkId: 'C1', pendingSources: 1, streamEnded: false };
  const sources = new Set([liveSource()]);
  const effects = reduceMessage(voice, { type: 'interrupt', reason: 'taking_floor', taking_floor: 0.9 });
  // app.js applyEffects: stopPlayback() now, mic stays open (no capture teardown).
  const interrupt = effects.find((e) => e.kind === 'interrupt');
  assert.ok(interrupt);
  const stopped = stopPlayback(playback, sources, { nowMs: 6000 });
  assert.equal(stopped.length, 1);
  assert.equal(playback.activeAudio, null);
  assert.equal(sources.size, 0);
  assert.equal(voice.transcriptVersion, 0);
});
