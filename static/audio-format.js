// Shared sample-rate contract helpers: 16 kHz mono PCM16 both directions.
// Capture is band-limited with a windowed-sinc FIR before decimation: a weak
// filter lets 8-24 kHz energy (sibilants, fan or electrical whine) fold back
// into the 0-8 kHz speech band that the recogniser listens to.

export const TARGET_SAMPLE_RATE = 16_000;

// Anti-alias design, in Hz at the 16 kHz output: the -6 dB point and the
// transition width. Passband to ~6.6 kHz, stopband from ~8.2 kHz (Blackman
// window, about -74 dB), so nothing above 8.2 kHz lands in the speech band.
const CUTOFF_HZ = 7_400;
const TRANSITION_HZ = 1_600;
const MAX_PHASES = 2_048;

function gcd(a, b) {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y) [x, y] = [y, x % y];
  return x;
}

function sinc(x) {
  if (x === 0) return 1;
  const px = Math.PI * x;
  return Math.sin(px) / px;
}

/**
 * One normalised kernel per fractional phase. Tap `m` (0..2H) weighs the
 * input sample `m` positions before the newest one used; `frac` is where the
 * output instant falls between input samples.
 */
function buildKernels(rate) {
  const scale = Math.min(rate, TARGET_SAMPLE_RATE) / TARGET_SAMPLE_RATE;
  const fc = (CUTOFF_HZ * scale) / rate; // cycles per input sample
  const half = Math.ceil((2.75 * rate) / (TRANSITION_HZ * scale)); // Blackman: N ~ 5.5 / transition
  const exactPhases = TARGET_SAMPLE_RATE / gcd(rate, TARGET_SAMPLE_RATE);
  const phases = Math.min(exactPhases, MAX_PHASES);
  const kernels = [];
  for (let p = 0; p < phases; p += 1) {
    const frac = p / phases;
    const taps = new Float64Array(2 * half + 1);
    let sum = 0;
    for (let m = 0; m <= 2 * half; m += 1) {
      const t = m - half + frac; // distance from the kernel centre, in input samples
      const w = 0.42 + 0.5 * Math.cos((Math.PI * t) / (half + 1)) + 0.08 * Math.cos((2 * Math.PI * t) / (half + 1));
      taps[m] = 2 * fc * sinc(2 * fc * t) * w;
      sum += taps[m];
    }
    for (let m = 0; m <= 2 * half; m += 1) taps[m] /= sum; // unity DC gain at every phase
    kernels.push(taps);
  }
  return { half, phases, kernels };
}

/**
 * Streaming resampler for Float32 mono capture frames at `inputRate` to
 * 16 kHz Int16 PCM. Output `k` is emitted on the same schedule as plain
 * decimation (after input sample ceil((k+1)·rate/16000) - 1), delayed by
 * the filter's half-length (~1.7 ms). Filter history and position carry
 * across `process()` calls, so AudioWorklet-sized frames (128 samples)
 * give exactly the same output as one call over the whole signal.
 */
export function createResampler(inputRate) {
  const rate = Number.isFinite(inputRate) && inputRate > 0 ? inputRate : TARGET_SAMPLE_RATE;
  if (rate === TARGET_SAMPLE_RATE) {
    return {
      inputRate: rate,
      process(samples) {
        if (!samples || !samples.length) return new Int16Array(0);
        const pcm = new Int16Array(samples.length);
        for (let i = 0; i < samples.length; i += 1) pcm[i] = floatToInt16(samples[i]);
        return pcm;
      },
    };
  }
  const { half, phases, kernels } = buildKernels(rate);
  const span = 2 * half + 1;
  let history = null; // the last `span` input samples
  let received = 0; // input samples seen so far
  let k = 0; // next output index
  return {
    inputRate: rate,
    process(samples) {
      if (!samples || !samples.length) return new Int16Array(0);
      if (history === null) history = new Float32Array(span).fill(samples[0]); // edge-extend the start
      const work = new Float32Array(span + samples.length);
      work.set(history, 0);
      work.set(samples, span);
      const base = received - span; // absolute input index of work[0]
      received += samples.length;
      const out = new Int16Array(Math.ceil((samples.length * TARGET_SAMPLE_RATE) / rate) + 2);
      let count = 0;
      for (;;) {
        const num = (k + 1) * rate;
        const newest = Math.ceil(num / TARGET_SAMPLE_RATE) - 1; // input index that releases output k
        if (newest >= received) break;
        const frac = (num % TARGET_SAMPLE_RATE) / TARGET_SAMPLE_RATE;
        const taps = kernels[Math.round(frac * phases) % phases];
        const end = Math.floor(num / TARGET_SAMPLE_RATE) - 1 - base; // newest tap, in work[]
        let acc = 0;
        for (let m = 0; m < span; m += 1) acc += taps[m] * work[end - m];
        out[count] = floatToInt16(acc);
        count += 1;
        k += 1;
      }
      history = work.slice(work.length - span);
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
 * Design uplink frame (guide §3.2 ASR_CHUNK_BYTES / ASR_CHUNK_MS): 256 samples
 * = 512 bytes of PCM16 = 16 ms at 16 kHz.
 */
export const UPLINK_FRAME_SAMPLES = 256;

/**
 * Collects resampled PCM into fixed-size batches (default 256 samples = 512
 * bytes = 16 ms at 16 kHz) so the socket carries a steady frame size instead of
 * one tiny message per AudioWorklet callback.
 */
export function createPcmBatcher(samplesPerBatch = UPLINK_FRAME_SAMPLES) {
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
