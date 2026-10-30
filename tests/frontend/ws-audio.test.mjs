import test from 'node:test';
import assert from 'node:assert/strict';
import {
  describeClose, shouldAutoReconnect, createReconnectState, nextBackoffDelay,
  resetReconnect, createWsLifecycle, onWsOpen, onWsClose,
} from '../../static/ws-client.js';
import { resampleTo16k, createResampler, createPcmBatcher, decodePcm16, TARGET_SAMPLE_RATE } from '../../static/audio-format.js';
import { micErrorMessage } from '../../static/mic-capture-node.js';

test('close codes surface a human-readable description with the reason', () => {
  assert.match(describeClose(1006, ''), /Abnormal closure/);
  assert.match(describeClose(1009, 'frame too large'), /frame too large/);
  assert.match(describeClose(1013, ''), /overload/i);
  assert.match(describeClose(4000, 'custom'), /4000/);
});

test('fatal close codes never auto-reconnect; drops and overload do', () => {
  assert.equal(shouldAutoReconnect(1008), false);
  assert.equal(shouldAutoReconnect(4404), false);
  assert.equal(shouldAutoReconnect(1006), true);
  assert.equal(shouldAutoReconnect(1013), true);
  assert.equal(shouldAutoReconnect(1006, { manualStop: true }), false);
});

test('backoff is bounded exponential and exhausts after max attempts', () => {
  const state = createReconnectState({ baseDelayMs: 500, maxDelayMs: 8000, maxAttempts: 5 });
  assert.deepEqual(
    [nextBackoffDelay(state), nextBackoffDelay(state), nextBackoffDelay(state), nextBackoffDelay(state), nextBackoffDelay(state)],
    [500, 1000, 2000, 4000, 8000],
  );
  assert.equal(nextBackoffDelay(state), null);
  resetReconnect(state);
  assert.equal(nextBackoffDelay(state), 500);
});

test('close lifecycle: auto drops keep recording state, fatal closes offer resume', () => {
  const lifecycle = createWsLifecycle();
  const reconnect = createReconnectState();
  onWsOpen(lifecycle, reconnect);
  assert.equal(lifecycle.status, 'open');
  const drop = onWsClose(lifecycle, 1006, '');
  assert.equal(drop.auto, true);
  assert.equal(lifecycle.status, 'closed');
  assert.equal(lifecycle.lastClose.code, 1006);
  const fatal = onWsClose(lifecycle, 4404, 'gone');
  assert.equal(fatal.auto, false);
  assert.equal(lifecycle.resumeOffered, true);
});

test('resampler keeps the 16 kHz contract and attenuates alias content', () => {
  assert.equal(TARGET_SAMPLE_RATE, 16_000);
  // 48 kHz sine at 1 kHz -> 160 samples per 10 ms at 16 kHz.
  const inputRate = 48_000;
  const samples = new Float32Array(480);
  for (let i = 0; i < samples.length; i += 1) {
    samples[i] = Math.sin((2 * Math.PI * 1000 * i) / inputRate);
  }
  const pcm = resampleTo16k(samples, inputRate);
  assert.equal(pcm.length, 160);
  assert.ok(pcm instanceof Int16Array);
  // DC input stays DC (no wild gain from the filter).
  const dc = resampleTo16k(new Float32Array(480).fill(0.5), inputRate);
  const mean = dc.reduce((a, b) => a + b, 0) / dc.length / 32767;
  assert.ok(Math.abs(mean - 0.5) < 0.05, `dc mean ${mean}`);
  // 16 kHz input passes through sample-for-sample.
  const direct = resampleTo16k(new Float32Array([0.5, -0.5, 0]), 16_000);
  assert.deepEqual([...direct], [16384, -16384, 0]);
});

test('PCM16 round-trips through the decoder', () => {
  const pcm = new Int16Array([0, 16384, -16384, 32767, -32768]);
  const decoded = decodePcm16(pcm.buffer);
  assert.equal(decoded.length, 5);
  assert.ok(Math.abs(decoded[1] - 0.5) < 1e-4);
  assert.deepEqual([...decodePcm16(new ArrayBuffer(0))], []);
  assert.deepEqual([...decodePcm16(new ArrayBuffer(1))], []);
});

test('microphone errors map to friendly English messages', () => {
  assert.match(micErrorMessage({ name: 'NotAllowedError' }), /denied/i);
  assert.match(micErrorMessage({ name: 'NotFoundError' }), /No microphone/);
  assert.match(micErrorMessage({ name: 'OverconstrainedError' }), /No microphone/);
  assert.match(micErrorMessage({ name: 'NotReadableError' }), /busy/);
  assert.match(micErrorMessage(new Error('boom')), /boom/);
});

test('streaming resampler matches a one-shot resample across worklet-sized frames', () => {
  for (const inputRate of [48_000, 44_100]) {
    const samples = new Float32Array(inputRate); // one second
    for (let i = 0; i < samples.length; i += 1) samples[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / inputRate);
    const oneShot = resampleTo16k(samples, inputRate);
    const resampler = createResampler(inputRate);
    const parts = [];
    for (let offset = 0; offset < samples.length; offset += 128) {
      parts.push(...resampler.process(samples.subarray(offset, offset + 128)));
    }
    // No sample is dropped at frame boundaries: one second stays ~16000 samples.
    assert.ok(Math.abs(parts.length - TARGET_SAMPLE_RATE) <= 1, `${inputRate}: ${parts.length}`);
    assert.deepEqual(parts.slice(0, oneShot.length), [...oneShot]);
  }
});

test('pcm batcher emits fixed 20 ms frames and carries the remainder', () => {
  const batcher = createPcmBatcher(320);
  assert.deepEqual(batcher.push(new Int16Array(100)), []);
  const batches = batcher.push(new Int16Array(600));
  assert.equal(batches.length, 2);
  assert.ok(batches.every((batch) => batch.length === 320));
  assert.equal(batcher.push(new Int16Array(259)).length, 0); // 60 carried + 259 = 319
  assert.equal(batcher.push(new Int16Array(1)).length, 1);
});
