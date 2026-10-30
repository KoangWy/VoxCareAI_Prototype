// Shared sample-rate contract helpers: 16 kHz mono PCM16 both directions.
// A gentle single-pole low-pass runs before decimation (plain box averaging
// would alias high-frequency energy down into the speech band).

export const TARGET_SAMPLE_RATE = 16_000;

/**
 * Streaming resampler for Float32 mono capture frames at `inputRate` to
 * 16 kHz Int16 PCM. The anti-alias filter state and the fractional read
 * position carry across `process()` calls, so splitting a signal into
 * AudioWorklet-sized frames (128 samples) neither drops samples nor clicks.
 */
export function createResampler(inputRate) {
  const rate = Number.isFinite(inputRate) && inputRate > 0 ? inputRate : TARGET_SAMPLE_RATE;
  const ratio = rate / TARGET_SAMPLE_RATE;
  const cutoff = 0.45 * (TARGET_SAMPLE_RATE / 2) / (rate / 2);
  const alpha = 1 - Math.exp(-2 * Math.PI * Math.min(0.45, Math.max(0.01, cutoff)));
  // Positions are counted in 1/16000 s (input samples step by 16000, output
  // samples by the input rate) so integer rates never accumulate float error.
  let y = 0;
  let consumed = 0;
  let nextTarget = rate;
  return {
    inputRate: rate,
    process(samples) {
      if (!samples || !samples.length) return new Int16Array(0);
      if (rate === TARGET_SAMPLE_RATE) {
        const pcm = new Int16Array(samples.length);
        for (let i = 0; i < samples.length; i += 1) pcm[i] = floatToInt16(samples[i]);
        return pcm;
      }
      const out = new Int16Array(Math.ceil(samples.length / ratio) + 1);
      let count = 0;
      for (let i = 0; i < samples.length; i += 1) {
        y += alpha * (samples[i] - y);
        consumed += TARGET_SAMPLE_RATE;
        while (consumed >= nextTarget) {
          out[count] = floatToInt16(y);
          count += 1;
          nextTarget += rate;
        }
      }
      // Rebase so the counters stay small however long the session runs.
      nextTarget -= consumed;
      consumed = 0;
      return out.subarray(0, count);
    },
  };
}

/**
 * One-shot resample of a whole Float32 buffer to 16 kHz Int16 PCM (a fresh
 * `createResampler` per call). Pure function — unit-tested in node.
 */
export function resampleTo16k(samples, inputRate) {
  if (!samples || !samples.length || !Number.isFinite(inputRate) || inputRate <= 0) return new Int16Array(0);
  return createResampler(inputRate).process(samples);
}

/**
 * Collects resampled PCM into fixed-size batches (default 320 samples = 20 ms
 * at 16 kHz) so the socket carries a steady frame size instead of one tiny
 * message per AudioWorklet callback.
 */
export function createPcmBatcher(samplesPerBatch = 320) {
  let buffer = new Int16Array(samplesPerBatch);
  let filled = 0;
  return {
    push(pcm) {
      const batches = [];
      let offset = 0;
      while (offset < pcm.length) {
        const take = Math.min(samplesPerBatch - filled, pcm.length - offset);
        buffer.set(pcm.subarray(offset, offset + take), filled);
        filled += take;
        offset += take;
        if (filled === samplesPerBatch) {
          batches.push(buffer);
          buffer = new Int16Array(samplesPerBatch);
          filled = 0;
        }
      }
      return batches;
    },
    reset() {
      filled = 0;
    },
  };
}

export function floatToInt16(sample) {
  const s = Math.max(-1, Math.min(1, sample));
  return s < 0 ? Math.max(-32768, Math.round(s * 32768)) : Math.min(32767, Math.round(s * 32767));
}

/** Decode a server PCM16 ArrayBuffer into Float32 samples in [-1, 1]. */
export function decodePcm16(data) {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  if (!bytes || bytes.byteLength < 2) return new Float32Array(0);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const sampleCount = Math.floor(bytes.byteLength / 2);
  const out = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i += 1) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}
