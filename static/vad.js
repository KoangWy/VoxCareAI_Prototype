// Adaptive SNR voice-activity gate ("open fast, decide late" — client side).
//
// Behavioural contract (shared by the browser capture path and the node tests):
//  - Tracks a slow-moving room noise floor from frame RMS; the onset gate is
//    SNR-based (floor x factor), never an absolute level, with an absolute
//    minimum floor so a dead-silent room cannot nominate on rustle.
//  - Once frame energy crosses the gate it must stay above it for
//    `speechHoldMs` before an onset is nominated (sustained-energy hold).
//    Brief dips (<= `dipGraceMs`) inside the streak do not reset it; longer
//    drops reset the streak. Fricative-length dips (~90-180 ms at 8 kHz frame
//    cadence) therefore keep one segment alive instead of splitting it.
//  - Onsets whose window looks like a single impulsive burst (cough spike,
//    click, mic bump: peaks early, dies fast, gappy continuity) are dropped
//    locally when `impulsiveFilter` is on — the sound already died out, so
//    no audio before the onset is clipped.
//  - While the assistant is speaking the gate factor rises and a short
//    refractory window after playback start ignores onsets (speaker echo).
//  - `vad_quiet` is an observation only: emitted after `quietMs` of
//    sustained sub-gate energy. It never ends a turn by itself.
//
// Frame input: Float32Array (or any numeric array) of time-domain samples in
// [-1, 1] plus a monotonic timestamp in ms, matching `performance.now()`.
// The gate is sample-rate agnostic: callers pass one frame per capture
// callback (or per synthetic step in tests).

export const VAD_DEFAULTS = Object.freeze({
  // Absolute minimum RMS floor: a dead-silent room cannot nominate on rustle.
  absMinRms: 0.006,
  // Gate = max(absMinRms, noiseFloor * factor); higher while agent speaks.
  snrGateFactorIdle: 3,
  snrGateFactorAgent: 5,
  // Sustained energy required before nominating an onset.
  speechHoldMs: 110,
  // Brief dips inside an onset streak that do not reset it.
  dipGraceMs: 60,
  // Drop obvious single impulsive bursts (cough/click/thump) locally.
  impulsiveFilter: true,
  // Ignore onsets this long after the agent starts talking.
  agentRefractoryMs: 350,
  // Quiet observation: silence required before `vad_quiet` fires.
  quietMs: 650,
  // Re-arm the onset streak only after this many consecutive above-gate
  // frames while idle (guards against a single loud sample re-opening).
  startFrames: 2,
  // Noise-floor learning rates: fast attack downward, very slow upward.
  // The floor never learns upward while an onset streak is open (the caller
  // is probably talking).
  floorAttack: 0.3,
  floorRelease: 0.002,
  floorMin: 0.0005,
  floorStart: 0.01,
});

export const VAD_TUNABLE_KEYS = Object.freeze([
  'snrGateFactorIdle',
  'snrGateFactorAgent',
  'speechHoldMs',
  'dipGraceMs',
  'agentRefractoryMs',
  'quietMs',
]);

export const VAD_TUNING_BOUNDS = Object.freeze({
  snrGateFactorIdle: [1.5, 8],
  snrGateFactorAgent: [2, 10],
  speechHoldMs: [40, 400],
  dipGraceMs: [0, 200],
  agentRefractoryMs: [0, 1000],
  quietMs: [250, 1500],
});

export function sanitizeVadConfig(patch = {}) {
  const clean = {};
  for (const key of VAD_TUNABLE_KEYS) {
    const value = patch[key];
    if (!Number.isFinite(value)) continue;
    const [lo, hi] = VAD_TUNING_BOUNDS[key];
    clean[key] = Math.min(hi, Math.max(lo, value));
  }
  return clean;
}

function frameRms(samples) {
  let energy = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const v = samples[i];
    energy += v * v;
  }
  return Math.sqrt(energy / samples.length);
}

/** Obvious single impulsive burst: peaks early, dies out, gappy. */
export function isImpulsiveBurst({ peakPosition, decayRatio, continuity }) {
  return peakPosition < 0.35 && decayRatio < 0.3 && continuity < 0.6;
}

function summariseOnset(frames, floor) {
  const n = frames.length;
  const rms = frames.map((f) => f.rms);
  const meanRms = rms.reduce((a, b) => a + b, 0) / n;
  let peakRms = 0;
  let peakIdx = 0;
  rms.forEach((r, i) => {
    if (r > peakRms) {
      peakRms = r;
      peakIdx = i;
    }
  });
  const gateHits = rms.filter((r) => r >= Math.max(VAD_DEFAULTS.absMinRms, floor * VAD_DEFAULTS.snrGateFactorIdle)).length;
  const third = Math.max(1, Math.floor(n / 3));
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const first = mean(rms.slice(0, third));
  const last = mean(rms.slice(n - third));
  return {
    durationMs: frames[n - 1].t - frames[0].t || 1,
    noiseFloorRms: floor,
    meanRms,
    peakRms,
    snrDb: 20 * Math.log10(Math.max(meanRms, 1e-6) / Math.max(floor, 1e-6)),
    continuity: gateHits / n,
    peakPosition: n > 1 ? peakIdx / (n - 1) : 0,
    decayRatio: first > 0 ? last / first : 1,
  };
}

function uuid() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  return `vad-${Date.now().toString(36)}-${Math.floor(Math.random() * 1e9).toString(36)}`;
}

export class LocalVad {
  constructor(options = {}) {
    this.options = { ...VAD_DEFAULTS, ...sanitizeVadConfig(options) };
    // Non-tunable structural options pass through verbatim.
    for (const key of ['absMinRms', 'startFrames', 'impulsiveFilter', 'floorAttack', 'floorRelease', 'floorMin', 'floorStart', 'quietMs']) {
      if (options[key] !== undefined) this.options[key] = options[key];
    }
    this.reset();
  }

  update(patch = {}) {
    Object.assign(this.options, sanitizeVadConfig(patch));
  }

  reset() {
    this.speaking = false;
    this.noiseFloor = this.options.floorStart;
    // Onset-streak state (idle -> onset nomination).
    this.aboveSince = null;
    this.lastAboveAt = 0;
    this.streak = [];
    this.idleAboveFrames = 0;
    // Quiet-observation state (speaking -> vad_quiet).
    this.quietStartedAt = null;
    this.quietSent = false;
    this.segmentId = null;
    this.lastSpeechAt = null;
    // Agent-speaking echo state.
    this.agentSpeaking = false;
    this.refractoryUntil = 0;
  }

  setAgentSpeaking(speaking, nowMs) {
    this.agentSpeaking = Boolean(speaking);
    if (this.agentSpeaking && Number.isFinite(nowMs)) {
      this.refractoryUntil = nowMs + this.options.agentRefractoryMs;
      this.aboveSince = null;
      this.streak = [];
      this.idleAboveFrames = 0;
    }
  }

  gateFactor() {
    return this.agentSpeaking ? this.options.snrGateFactorAgent : this.options.snrGateFactorIdle;
  }

  gate() {
    return Math.max(this.options.absMinRms, this.noiseFloor * this.gateFactor());
  }

  process(samples, nowMs) {
    if (!samples?.length || !Number.isFinite(nowMs)) return [];
    const rms = frameRms(samples);

    // Noise floor: fast attack downward, very slow upward. Never learn the
    // floor upward while an onset streak is open or speech is active (the
    // caller is probably talking).
    if (rms < this.noiseFloor) {
      const a = this.options.floorAttack;
      this.noiseFloor = this.noiseFloor * (1 - a) + rms * a;
    } else if (this.aboveSince === null && !this.speaking) {
      const r = this.options.floorRelease;
      this.noiseFloor += (rms - this.noiseFloor) * r;
    }
    this.noiseFloor = Math.max(this.noiseFloor, this.options.floorMin);

    const gate = this.gate();
    if (this.speaking) return this.#processSpeaking(rms, nowMs);
    return this.#processIdle(rms, nowMs, gate);
  }

  #processIdle(rms, nowMs, gate) {
    if (nowMs < this.refractoryUntil) {
      this.aboveSince = null;
      this.streak = [];
      this.idleAboveFrames = 0;
      return [];
    }
    if (rms >= gate) {
      this.idleAboveFrames += 1;
      if (this.idleAboveFrames < this.options.startFrames) return [];
      if (this.aboveSince === null) {
        this.aboveSince = nowMs;
        this.streak = [];
      }
      this.lastAboveAt = nowMs;
      this.streak.push({ t: nowMs, rms });
    } else if (this.aboveSince !== null) {
      if (nowMs - this.lastAboveAt <= this.options.dipGraceMs) {
        this.streak.push({ t: nowMs, rms }); // brief dip inside a streak — keep it
      } else {
        this.aboveSince = null;
        this.streak = [];
        this.idleAboveFrames = 0;
        return [];
      }
    } else {
      this.idleAboveFrames = 0;
      return [];
    }

    if (this.aboveSince !== null && nowMs - this.aboveSince >= this.options.speechHoldMs && this.streak.length >= 3) {
      const features = summariseOnset(this.streak, this.noiseFloor);
      const impulsive = this.options.impulsiveFilter && isImpulsiveBurst(features);
      this.aboveSince = null;
      this.streak = [];
      this.idleAboveFrames = 0;
      if (impulsive) return []; // dropped locally; keep scanning
      this.speaking = true;
      this.segmentId = uuid();
      this.quietStartedAt = null;
      this.quietSent = false;
      this.lastSpeechAt = nowMs;
      return [{ type: 'vad_speech_start', segment_id: this.segmentId, client_monotonic_ms: nowMs, rms, snr_db: features.snrDb }];
    }
    return [];
  }

  #processSpeaking(rms, nowMs) {
    const gate = this.gate();
    if (rms < gate) {
      if (this.quietStartedAt === null) this.quietStartedAt = nowMs;
      const silenceMs = Math.max(0, nowMs - this.quietStartedAt);
      if (!this.quietSent && silenceMs >= this.options.quietMs) {
        this.quietSent = true;
        return [{
          type: 'vad_quiet', segment_id: this.segmentId, client_monotonic_ms: nowMs,
          speech_end_monotonic_ms: this.lastSpeechAt, silence_ms: Math.round(silenceMs),
        }];
      }
      return [];
    }
    // Above gate: speech continues. A re-onset after a reported quiet opens
    // a fresh segment; anything shorter is the same utterance continuing.
    this.lastSpeechAt = nowMs;
    if (this.quietSent) {
      this.segmentId = uuid();
      this.quietStartedAt = null;
      this.quietSent = false;
      return [{ type: 'vad_speech_start', segment_id: this.segmentId, client_monotonic_ms: nowMs, rms }];
    }
    this.quietStartedAt = null;
    return [];
  }
}

/**
 * Flag-aware onset routing. Pure: decides whether a
 * local VAD onset is sent to the server or dropped as speaker echo, given
 * the negotiated barge-in mode from the session hello.
 *
 * - `off` (default, half-duplex gate): an onset while the reply is playing
 *   (or inside the echo tail) is speaker feedback — drop it; the caller
 *   resets the VAD streak. Anything else sends unchanged (no flag).
 * - `semantic`: the short echo tail after audio ENDS still drops locally,
 *   but an onset while audio is PLAYING sends with `agent_speaking: true`
 *   instead of being dropped — the server holds the audio and judges
 *   taking_floor; playback continues until `interrupt`. An onset with
 *   nothing playing sends with `agent_speaking: false`.
 * Returns `{ action: 'send', event }` or `{ action: 'drop-echo',
 * traceLabel, traceDetail }`.
 */
export function routeVadOnset(onsetEvent, { activeAudio, nowMs, echoUntilMs = 0, bargeInMode = 'off' } = {}) {
  if (!onsetEvent || onsetEvent.type !== 'vad_speech_start') return { action: 'drop-not-onset' };
  if (activeAudio && bargeInMode === 'semantic') {
    return { action: 'send', event: { ...onsetEvent, agent_speaking: true } };
  }
  const echo = vadOnsetIsEcho(onsetEvent, { activeAudio, nowMs, echoUntilMs });
  if (echo) {
    return {
      action: 'drop-echo',
      traceLabel: 'Ignored VAD onset',
      traceDetail: 'assistant playback echo gate — onset treated as speaker feedback',
    };
  }
  if (bargeInMode === 'semantic') {
    return { action: 'send', event: { ...onsetEvent, agent_speaking: Boolean(activeAudio) } };
  }
  return { action: 'send', event: onsetEvent };
}

export function vadOnsetIsEcho(onsetEvent, { activeAudio, nowMs, echoUntilMs = 0 } = {}) {
  // The demo is half-duplex: while the assistant reply is playing (or within
  // the short echo tail after it), a VAD onset is treated as speaker-to-mic
  // echo of the bot's own voice, not as caller speech.
  if (!onsetEvent || onsetEvent.type !== 'vad_speech_start') return false;
  if (activeAudio) return true;
  return Number.isFinite(nowMs) && Number.isFinite(echoUntilMs) && nowMs < echoUntilMs;
}
export function callbackIsCurrent(message, { generation, activeGeneration, sessionId, turnEpoch, transcriptVersion } = {}) {
  if (generation !== activeGeneration) return false;
  if (message.session_id && sessionId && message.session_id !== sessionId) return false;
  if (Number.isInteger(message.turn_epoch) && Number.isInteger(turnEpoch) && message.turn_epoch < turnEpoch) return false;
  if (Number.isInteger(message.transcript_version) && Number.isInteger(transcriptVersion) && message.transcript_version < transcriptVersion) return false;
  return true;
}

export const TTFA_UNAVAILABLE = 'Unavailable — no acoustically verified playback measurement';

export function ttfaLabel({ t0, t6, clockAligned = false, acousticConfirmed = false } = {}) {
  if (!Number.isFinite(t0) || !Number.isFinite(t6) || !clockAligned || !acousticConfirmed || t6 < t0) {
    return TTFA_UNAVAILABLE;
  }
  return `${Math.round(t6 - t0)} ms (measured)`;
}

export function startMessage(sessionId, audio = {}) {
  return {
    type: 'start',
    session_id: sessionId,
    sample_rate: audio.sampleRate ?? 16_000,
    channels: 1,
    encoding: 'pcm_s16le',
    client_clock: { name: 'performance.now', resolution_ms: audio.resolutionMs ?? null },
  };
}

export function voiceAvailability(providers = {}) {
  const required = ['asr', 'conversation', 'tts'];
  const missing = required.filter((name) => !providers[name]?.configured).map((name) => providers[name]?.name || name);
  return {
    available: missing.length === 0,
    missing,
    label: missing.length ? 'Voice providers unavailable' : 'Start speaking',
    help: missing.length
      ? `Configure ${missing.join(', ')} in the server .env file to enable live voice.`
      : 'Speak in English. The transcript updates as speech is recognized.',
  };
}

export function supportRequestEnabled(session) {
  return Boolean(session && typeof session.session_id === 'string' && session.session_id.length > 0);
}

export async function submitSupportRequest({ fetchImpl, sessionId, reason, isCurrent }) {
  const response = await fetchImpl(`/api/sessions/${encodeURIComponent(sessionId)}/handoff`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason: String(reason || '').trim().slice(0, 200) }),
  });
  if (!isCurrent()) return { status: 'STALE_DISCARDED' };
  const result = await response.json();
  if (!isCurrent()) return { status: 'STALE_DISCARDED' };
  if (!response.ok) throw new Error(result.detail || 'The support request could not be recorded.');
  if (result.outcome !== 'HANDOFF_REQUEST_RECORDED') throw new Error('The support request did not return a recorded outcome.');
  return { status: 'RECORDED', result };
}
