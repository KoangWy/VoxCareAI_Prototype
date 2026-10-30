import test from 'node:test';
import assert from 'node:assert/strict';
import { LocalVad, isImpulsiveBurst, sanitizeVadConfig, VAD_DEFAULTS } from '../../static/vad.js';

const FRAME = 128;
const STEP_MS = 10;

function clip(vad, segments, { untilMs = 4000, baseAmp = 0.001 } = {}) {
  // segments: [[startMs, endMs, amp]] over a base room tone. Later segments
  // win on overlap, so a speech burst can sit on top of steady noise.
  const events = [];
  for (let t = 0; t < untilMs; t += STEP_MS) {
    let amp = baseAmp;
    for (const [s, e, a] of segments) {
      if (t >= s && t < e) amp = a;
    }
    events.push(...vad.process(new Float32Array(FRAME).fill(amp), t));
  }
  return events;
}

const types = (events) => events.map((e) => e.type);

test('adaptive gate: digital silence never nominates an onset', () => {
  const vad = new LocalVad();
  assert.deepEqual(clip(vad, []), []);
});

test('adaptive gate: steady room noise below the SNR gate never nominates', () => {
  const vad = new LocalVad();
  // 0.02 RMS room noise for 3 s: the floor adapts (~0.02) and the gate
  // (floor x 3) stays above it.
  const events = clip(new LocalVad(), [[0, 3000, 0.02]], { untilMs: 3000 });
  assert.deepEqual(events, []);
  assert.ok(vad.gate() >= VAD_DEFAULTS.absMinRms);
});

test('adaptive gate: speech burst nominates an onset, then observes quiet', () => {
  const vad = new LocalVad();
  const events = clip(vad, [[1000, 1500, 0.15]]);
  assert.ok(events.some((e) => e.type === 'vad_speech_start'), 'expected an onset');
  assert.ok(events.some((e) => e.type === 'vad_quiet'), 'expected a quiet observation');
  const onset = events.find((e) => e.type === 'vad_speech_start');
  // Sustained-energy hold: onset comes ~110 ms after the burst starts.
  assert.ok(onset.client_monotonic_ms - 1000 >= 100, `onset at ${onset.client_monotonic_ms}`);
  assert.ok(Number.isFinite(onset.snr_db) && onset.snr_db > 0);
  const quiet = events.find((e) => e.type === 'vad_quiet');
  assert.ok(quiet.silence_ms >= 650);
  assert.equal(quiet.segment_id, onset.segment_id);
});

test('adaptive gate: quiet-room speech above the absolute floor nominates', () => {
  const vad = new LocalVad();
  const events = clip(vad, [[800, 1400, 0.05]]);
  assert.deepEqual(types(events), ['vad_speech_start', 'vad_quiet']);
});

test('adaptive gate: speech in a noisy room nominates via SNR, not level', () => {
  const vad = new LocalVad();
  // 0.02 noise floor, 0.12 speech: a fixed 0.018 RMS gate would have been
  // permanently open; the SNR gate stays shut until the burst.
  const events = clip(vad, [[2000, 2600, 0.12]], { untilMs: 4000, baseAmp: 0.02 });
  const onsets = events.filter((e) => e.type === 'vad_speech_start');
  assert.equal(onsets.length, 1);
  assert.ok(onsets[0].client_monotonic_ms >= 2000);
});

test('adaptive gate: short cough impulse is rejected, no onset', () => {
  const vad = new LocalVad();
  assert.deepEqual(clip(vad, [[500, 530, 0.3]]), []);
});

test('adaptive gate: longer cough burst is still rejected as impulsive', () => {
  const vad = new LocalVad();
  assert.deepEqual(clip(vad, [[500, 580, 0.3]]), []);
});

test('adaptive gate: fricative-length dip keeps one segment alive', () => {
  const vad = new LocalVad();
  const events = clip(vad, [[500, 1500, 0.12], [900, 1020, 0.002]]);
  const onsets = events.filter((e) => e.type === 'vad_speech_start');
  assert.equal(onsets.length, 1, `expected one onset, got ${JSON.stringify(types(events))}`);
  assert.equal(events.filter((e) => e.type === 'vad_quiet').length, 1);
});

test('adaptive gate: refractory window ignores onsets right after playback starts', () => {
  const vad = new LocalVad();
  vad.setAgentSpeaking(true, 1000);
  const during = [];
  for (let t = 1000; t < 1350; t += STEP_MS) {
    during.push(...vad.process(new Float32Array(FRAME).fill(0.15), t));
  }
  assert.deepEqual(during, []);
});

test('adaptive gate: sustained speech after the refractory window still nominates', () => {
  const vad = new LocalVad();
  vad.setAgentSpeaking(true, 1000);
  const events = [];
  for (let t = 1000; t < 2500; t += STEP_MS) {
    events.push(...vad.process(new Float32Array(FRAME).fill(0.15), t));
  }
  assert.equal(events.filter((e) => e.type === 'vad_speech_start').length, 1);
});

test('isImpulsiveBurst matches the reference shape rule', () => {
  assert.equal(isImpulsiveBurst({ peakPosition: 0.1, decayRatio: 0.2, continuity: 0.4 }), true);
  assert.equal(isImpulsiveBurst({ peakPosition: 0.6, decayRatio: 0.9, continuity: 0.95 }), false);
  assert.equal(isImpulsiveBurst({ peakPosition: 0.1, decayRatio: 0.9, continuity: 0.9 }), false);
});

test('sanitizeVadConfig clamps tuning to safe bounds and drops unknowns', () => {
  const clean = sanitizeVadConfig({ speechHoldMs: 50, quietMs: 99999, bogus: 1, dipGraceMs: NaN });
  assert.equal(clean.speechHoldMs, 50);
  assert.equal(clean.quietMs, 1500);
  assert.ok(!('bogus' in clean));
  assert.ok(!('dipGraceMs' in clean));
});
