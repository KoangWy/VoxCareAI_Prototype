import test from 'node:test';
import assert from 'node:assert/strict';
import {
  LocalVad, callbackIsCurrent, startMessage, submitSupportRequest, supportRequestEnabled,
  ttfaLabel, TTFA_UNAVAILABLE, voiceAvailability, vadOnsetIsEcho,
} from '../../static/vad.js';

test('local VAD reports speech onset and quiet observation without deciding an end of turn', () => {
  const vad = new LocalVad({ quietMs: 650, speechHoldMs: 40, startFrames: 2 });
  const loud = new Float32Array(128).fill(0.1);
  const quiet = new Float32Array(128);
  assert.equal(vad.process(loud, 10).length, 0);
  assert.equal(vad.process(loud, 20).length, 0);
  assert.equal(vad.process(loud, 40).length, 0);
  const onset = vad.process(loud, 60);
  assert.equal(onset[0].type, 'vad_speech_start');
  assert.equal(vad.process(quiet, 100).length, 0);
  const quietEvent = vad.process(quiet, 750);
  assert.equal(quietEvent[0].type, 'vad_quiet');
  assert.equal(vad.process(quiet, 900).length, 0);
});

test('voice unavailable state remains explicit after session reset rendering', () => {
  const providers = {
    asr: { name: 'Deepgram Nova-2', configured: false },
    conversation: { name: 'OpenAI gpt-6-luna', configured: false },
    tts: { name: 'ElevenLabs Flash 2.5', configured: false },
  };
  const state = voiceAvailability(providers);
  assert.equal(state.available, false);
  assert.equal(state.label, 'Voice providers unavailable');
  assert.match(state.help, /Configure Deepgram Nova-2/);
});

test('support request is enabled by an active session without provider or websocket state', async () => {
  assert.equal(supportRequestEnabled({ session_id: 'session-1' }), true);
  assert.equal(supportRequestEnabled(null), false);
  let request;
  const result = await submitSupportRequest({
    sessionId: 'session-1', reason: '  Need a person  ', isCurrent: () => true,
    fetchImpl: async (url, options) => {
      request = { url, options };
      return { ok: true, json: async () => ({ outcome: 'HANDOFF_REQUEST_RECORDED', status: 'REQUEST_RECORDED' }) };
    },
  });
  assert.equal(request.url, '/api/sessions/session-1/handoff');
  assert.equal(request.options.method, 'POST');
  assert.deepEqual(JSON.parse(request.options.body), { reason: 'Need a person' });
  assert.equal(result.status, 'RECORDED');
});

test('support acknowledgement from the old session is discarded after reset', async () => {
  let resolveFetch;
  let active = true;
  const pending = submitSupportRequest({
    sessionId: 'session-old', reason: 'Need support', isCurrent: () => active,
    fetchImpl: () => new Promise((resolve) => { resolveFetch = resolve; }),
  });
  active = false;
  resolveFetch({ ok: true, json: async () => ({ outcome: 'HANDOFF_REQUEST_RECORDED' }) });
  assert.deepEqual(await pending, { status: 'STALE_DISCARDED' });
});

test('real speech after a quiet observation emits a trusted onset for a fresh segment', () => {
  const vad = new LocalVad({ quietMs: 100, speechHoldMs: 40, startFrames: 1 });
  const loud = new Float32Array(64).fill(0.1);
  const quiet = new Float32Array(64);
  assert.equal(vad.process(loud, 0).length, 0);
  assert.equal(vad.process(loud, 20).length, 0);
  const opened = vad.process(loud, 50);
  assert.equal(opened[0].type, 'vad_speech_start');
  vad.process(quiet, 60);
  assert.equal(vad.process(quiet, 170)[0].type, 'vad_quiet');
  const nextSpeech = vad.process(loud, 220)[0];
  assert.equal(nextSpeech.type, 'vad_speech_start');
  assert.notEqual(nextSpeech.segment_id, opened[0].segment_id);
});

test('reset generation and correlation metadata discard stale callbacks', () => {
  const current = { generation: 4, activeGeneration: 4, sessionId: 'new', turnEpoch: 3, transcriptVersion: 8 };
  assert.equal(callbackIsCurrent({ session_id: 'new', turn_epoch: 3, transcript_version: 8 }, current), true);
  assert.equal(callbackIsCurrent({ session_id: 'old', turn_epoch: 3, transcript_version: 8 }, current), false);
  assert.equal(callbackIsCurrent({ session_id: 'new', turn_epoch: 2, transcript_version: 8 }, current), false);
  assert.equal(callbackIsCurrent({}, { ...current, generation: 3 }), false);
});

test('TTFA stays unavailable without aligned acoustic confirmation', () => {
  assert.equal(ttfaLabel({ t0: 10, t6: 500, clockAligned: true }), TTFA_UNAVAILABLE);
  assert.equal(ttfaLabel({ t0: 10, t6: 500, clockAligned: true, acousticConfirmed: true }), '490 ms (measured)');
});

test('audio start identifies session and monotonic clock resolution', () => {
  assert.deepEqual(startMessage('s1'), {
    type: 'start', session_id: 's1', sample_rate: 16_000, channels: 1, encoding: 'pcm_s16le',
    client_clock: { name: 'performance.now', resolution_ms: null },
  });
});

test('VAD onset while assistant audio plays or in the echo tail is echo, not caller speech', () => {
  const onset = { type: 'vad_speech_start', rms: 0.05 };
  assert.equal(vadOnsetIsEcho(onset, { activeAudio: { turnId: 'T1' }, nowMs: 1000 }), true);
  assert.equal(vadOnsetIsEcho(onset, { activeAudio: null, nowMs: 1000, echoUntilMs: 1500 }), true);
  assert.equal(vadOnsetIsEcho(onset, { activeAudio: null, nowMs: 1500, echoUntilMs: 1500 }), false);
  assert.equal(vadOnsetIsEcho(onset, { activeAudio: null, nowMs: 2000, echoUntilMs: 1500 }), false);
  assert.equal(vadOnsetIsEcho({ type: 'vad_quiet' }, { activeAudio: { turnId: 'T1' }, nowMs: 1000 }), false);
  assert.equal(vadOnsetIsEcho(null, { activeAudio: { turnId: 'T1' }, nowMs: 1000 }), false);
});
