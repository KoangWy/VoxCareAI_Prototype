import {
  LocalVad, VAD_DEFAULTS, VAD_TUNABLE_KEYS, VAD_TUNING_BOUNDS, sanitizeVadConfig,
  callbackIsCurrent, startMessage, submitSupportRequest, supportRequestEnabled,
  ttfaLabel, voiceAvailability, routeVadOnset,
} from './vad.js';
import {
  createPlaybackState, onAudioStart, onAudioEnd, stopPlayback as stopPlaybackState,
  shouldPlayFrame, markSourceEnded, consumeChunkForBinary,
} from './audio-playback.js';
import {
  createMicState, micTransitionAllowed, beginStart, beginStop,
  isTokenCurrent, finishStart, finishStop, abortTransition,
} from './mic-capture.js';
import {
  createReconnectState, nextBackoffDelay, resetReconnect,
  createWsLifecycle, onWsOpen, onWsClose,
} from './ws-client.js';
import {
  createVoiceState, reduceMessage, routeHandoff, handoffRequestPayload,
} from './voice-protocol.js';
import { attachCapture, micErrorMessage, grantedConstraints } from './mic-capture-node.js';
import { createResampler, createPcmBatcher, decodePcm16 } from './audio-format.js';

const $ = (id) => document.getElementById(id);
const ui = {
  pill: $('session-pill'), state: $('conversation-state'), mic: $('mic-button'), micLabel: $('mic-label'),
  micHelp: $('mic-help'), transcript: $('transcript-box'), answerCard: $('answer-card'), answer: $('answer-text'),
  evidence: $('evidence-list'), error: $('error-banner'), providers: $('provider-list'), sessionId: $('session-id'),
  protocol: $('protocol-mode'), form: $('text-form'), input: $('message-input'), reset: $('reset-button'), actions: $('recovery-actions'),
  finish: $('finish-button'), retry: $('retry-button'), support: $('support-button'), supportReason: $('support-reason'),
  supportStatus: $('support-status'),
  trace: $('trace-list'), ttfa: $('ttfa-value'), playback: $('playback-state'),
  settings: $('vad-settings'), settingsToggle: $('vad-settings-toggle'),
};

const PLAYBACK_ECHO_TAIL_MS = 600;
export const MIC_CONSTRAINTS = Object.freeze({
  channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
});

let session = null;
let config = null;
let socket = null;
let mediaStream = null;
let audioContext = null;
let captureHandle = null;
let micResampler = null;
let micBatcher = null;
let isRecording = false;
let asrRunning = false;
let generation = 0;
let manualStop = false;
const mic = createMicState();
const playback = createPlaybackState();
const reconnect = createReconnectState();
const wsLife = createWsLifecycle();
const voice = createVoiceState();
let vad = new LocalVad(loadVadTuning());
const playbackSources = new Set();
let reconnectTimer = null;
const oneShotTimers = new Set();

function later(fn, delayMs) {
  const id = setTimeout(() => {
    oneShotTimers.delete(id);
    fn();
  }, delayMs);
  oneShotTimers.add(id);
  return id;
}

function clearAllTimers() {
  for (const id of oneShotTimers) clearTimeout(id);
  oneShotTimers.clear();
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

// --- VAD tuning drawer (persisted in localStorage) ---
const VAD_TUNING_KEY = 'voxcare.vad-tuning.v1';

function loadVadTuning() {
  try {
    const raw = localStorage.getItem(VAD_TUNING_KEY);
    if (!raw) return {};
    return sanitizeVadConfig(JSON.parse(raw));
  } catch {
    return {};
  }
}

function saveVadTuning() {
  try {
    const tuned = {};
    for (const key of VAD_TUNABLE_KEYS) tuned[key] = vad.options[key];
    localStorage.setItem(VAD_TUNING_KEY, JSON.stringify(tuned));
  } catch {
    /* private mode / quota — tuning stays in memory */
  }
}

function sliderPosition(key, value) {
  const [lo, hi] = VAD_TUNING_BOUNDS[key];
  return String(Math.round(((value - lo) / (hi - lo)) * 100));
}

const VAD_TUNING_LABELS = {
  snrGateFactorIdle: 'Gate factor (idle)',
  snrGateFactorAgent: 'Gate factor (assistant speaking)',
  speechHoldMs: 'Onset hold (ms)',
  dipGraceMs: 'Dip grace (ms)',
  agentRefractoryMs: 'Echo refractory (ms)',
  quietMs: 'Quiet observation (ms)',
};

function buildVadSettings() {
  if (!ui.settings) return;
  ui.settings.replaceChildren();
  const title = document.createElement('p');
  title.className = 'eyebrow';
  title.textContent = 'VOICE DETECTION TUNING';
  ui.settings.append(title);
  for (const key of VAD_TUNABLE_KEYS) {
    const row = document.createElement('label');
    row.className = 'tune-row';
    const name = document.createElement('span');
    name.textContent = `${VAD_TUNING_LABELS[key]}: `;
    const value = document.createElement('output');
    value.textContent = String(vad.options[key]);
    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = '100';
    input.value = sliderPosition(key, vad.options[key]);
    input.setAttribute('aria-label', VAD_TUNING_LABELS[key]);
    input.dataset.tuneKey = key;
    input.addEventListener('input', () => {
      const bounds = VAD_TUNING_BOUNDS[key];
      const next = bounds[0] + (Number(input.value) / 100) * (bounds[1] - bounds[0]);
      vad.update({ [key]: Math.round(next * 100) / 100 });
      value.textContent = String(vad.options[key]);
      saveVadTuning();
      addTrace('VAD tuning changed', `${key} = ${vad.options[key]} (defaults ${VAD_DEFAULTS[key]})`);
    });
    row.append(name, value, input);
    ui.settings.append(row);
  }
  const reset = document.createElement('button');
  reset.type = 'button';
  reset.className = 'button secondary';
  reset.textContent = 'Reset tuning to defaults';
  reset.addEventListener('click', () => {
    vad = new LocalVad();
    saveVadTuning();
    buildVadSettings();
    addTrace('VAD tuning reset', 'factory defaults restored');
  });
  ui.settings.append(reset);
}

// --- small DOM helpers ---
function showError(message, kind = 'recoverable') {
  ui.error.textContent = message;
  ui.error.dataset.kind = kind;
  ui.error.hidden = false;
}

function clearError() {
  ui.error.textContent = '';
  ui.error.hidden = true;
  delete ui.error.dataset.kind;
}

function setState(text, live = false) {
  ui.state.textContent = text;
  ui.state.classList.toggle('live', live);
}

function addTrace(label, detail = '') {
  const item = document.createElement('li');
  const title = document.createElement('strong');
  title.textContent = label;
  const value = document.createElement('span');
  value.textContent = detail;
  item.append(title, value);
  ui.trace.prepend(item);
  while (ui.trace.children.length > 24) ui.trace.lastElementChild.remove();
}

function currentGuard(capturedGeneration, capturedSessionId) {
  return capturedGeneration === generation && session?.session_id === capturedSessionId;
}

function eventIsCurrent(message, capturedGeneration, capturedSessionId) {
  return callbackIsCurrent(message, {
    generation: capturedGeneration,
    activeGeneration: generation,
    sessionId: capturedSessionId,
    turnEpoch: voice.turnEpoch,
    transcriptVersion: voice.transcriptVersion,
  });
}

function setMicBusy(busy, label) {
  ui.mic.disabled = busy ? true : ui.mic.disabled;
  if (label) ui.mic.setAttribute('aria-label', label);
}

// --- providers / turn rendering (textContent only, never innerHTML) ---
function renderProviders(providers = {}) {
  const entries = [providers.asr, providers.conversation, providers.conversation_fallback, providers.tts].filter(Boolean);
  ui.providers.replaceChildren(...entries.map((provider) => {
    const item = document.createElement('li');
    const dot = document.createElement('span');
    dot.className = `provider-state${provider.configured ? ' ready' : ''}`;
    dot.setAttribute('aria-hidden', 'true');
    const name = document.createElement('span');
    name.className = 'provider-name';
    name.textContent = provider.name;
    const detail = document.createElement('span');
    detail.className = 'provider-detail';
    detail.textContent = provider.configured ? `${provider.model || 'configured'} · ready to connect` : 'Credential not configured';
    const label = document.createElement('span');
    label.append(name, detail);
    item.append(dot, label);
    return item;
  }));

  const availability = voiceAvailability(providers);
  const missing = availability.missing.join(', ');
  ui.mic.disabled = !availability.available || mic.transitioning;
  if (!availability.available) {
    ui.micLabel.textContent = availability.label;
    ui.micHelp.textContent = availability.help;
    ui.form.querySelector('button[type="submit"]').disabled = !providers.conversation?.configured;
    if (!providers.conversation?.configured) ui.input.placeholder = 'Conversation provider not configured';
    showError(`Live microphone flow is unavailable until these direct providers are configured: ${missing}. No simulated answer will be shown.`, 'configuration');
  } else {
    ui.micLabel.textContent = isRecording ? 'Listening…' : 'Start speaking';
    ui.micHelp.textContent = isRecording
      ? 'Speak naturally. Silence is observed, never treated as an automatic end of turn.'
      : availability.help;
    ui.form.querySelector('button[type="submit"]').disabled = !providers.conversation?.configured;
    if (providers.conversation?.configured) ui.input.placeholder = 'Try: Where is order ORD-1002?';
    if (ui.error.dataset.kind === 'configuration') clearError();
  }
}

function renderEvidence(result) {
  ui.evidence.replaceChildren();
  const evidence = Array.isArray(result.evidence) ? result.evidence : [];
  if (evidence.length) {
    const notice = document.createElement('p');
    notice.className = 'fictional-tag';
    notice.textContent = result.is_fictional || evidence.some((item) => item.is_fictional)
      ? `Fictional demo policy · ${result.corpus_id || evidence[0].corpus_id || 'corpus'} · ${result.clause_version || evidence[0].clause_version || 'version not provided'}`
      : `Evidence returned · ${result.clause_version || 'version not provided'}`;
    ui.evidence.append(notice);
    for (const item of evidence) {
      const article = document.createElement('article');
      article.className = 'evidence-item';
      const heading = document.createElement('strong');
      heading.textContent = `${item.policy_id || 'Policy'} · ${item.clause_ref || 'clause'}`;
      const category = document.createElement('span');
      category.textContent = `${item.category || 'Policy'}${item.applicability_rules ? ` · applies under ${JSON.stringify(item.applicability_rules)}` : ''}`;
      const content = document.createElement('p');
      content.textContent = item.content_en || item.title || '';
      article.append(heading, category, content);
      ui.evidence.append(article);
    }
  } else if (result.policy_outcome || result.outcome === 'INSUFFICIENT_EVIDENCE' || result.outcome === 'NEEDS_CONTEXT') {
    const empty = document.createElement('p');
    empty.className = 'insufficient-evidence';
    empty.textContent = 'No applicable policy evidence was returned. This response does not verify eligibility, fees, refunds, or warranty coverage.';
    ui.evidence.append(empty);
    if (Array.isArray(result.missing_context) && result.missing_context.length) {
      const missing = document.createElement('p');
      missing.textContent = `More context needed: ${result.missing_context.join(', ')}.`;
      ui.evidence.append(missing);
    }
  }
}

function renderTurn(result) {
  if (!result) return;
  if (result.user_text) {
    ui.transcript.textContent = result.user_text;
    ui.transcript.classList.remove('placeholder');
  }
  ui.answerCard.hidden = true;
  ui.evidence.replaceChildren();
  if (result.assistant_text) {
    ui.answer.textContent = result.assistant_text;
    ui.answerCard.hidden = false;
  }
  renderEvidence(result);
  const messages = {
    DEPENDENCY_UNAVAILABLE: 'A required provider is unavailable. The request was not answered; check configuration or retry.',
    TOOL_ERROR: 'A read-only service failed. No successful lookup was reported; you can retry or request support.',
    SERVICE_TIMEOUT: 'The read service timed out after its bounded retry. No order or policy result was assumed.',
    OUT_OF_SCOPE: 'That order is outside this simulated session. No order details were returned.',
    AUTH_SCOPE_VIOLATION: 'This request is outside the current demo session. No details were returned.',
    INSUFFICIENT_EVIDENCE: 'No applicable policy evidence was available for this question.',
    NEEDS_CONTEXT: 'More product or brand details are needed before checking applicable policy information.',
    HANDOFF_REQUEST_RECORDED: 'Your request for support was recorded for this demo. No live agent transfer occurred.',
  };
  if (messages[result.outcome]) showError(messages[result.outcome]);
  else if (result.outcome === 'STALE_DISCARDED') return;
  else if (ui.error.dataset.kind !== 'configuration') clearError();
  setState(String(result.outcome || 'response').replaceAll('_', ' ').toLowerCase());
  if (result.trace_id) addTrace('Turn result', `${result.outcome} · ${result.trace_id}`);
  for (const event of result.events || []) addTrace(event.event_type || event.type || 'Event', event.outcome || event.reason || 'recorded');
}

// --- transport ---
function sendControl(payload) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    showError('The voice connection is not active. Start the microphone and reconnect before choosing a turn action.');
    return false;
  }
  socket.send(JSON.stringify(payload));
  return true;
}

// --- playback (WebAudio glue over audio-playback.js state) ---
function stopPlayback() {
  stopPlaybackState(playback, playbackSources, { echoTailMs: PLAYBACK_ECHO_TAIL_MS, nowMs: performance.now() });
  vad.setAgentSpeaking(false, performance.now());
}

function acknowledgePlayback(turnId, audioChunkId, capturedGeneration, capturedSessionId, atMs) {
  if (!currentGuard(capturedGeneration, capturedSessionId) || !turnId || !audioChunkId || !socket || socket.readyState !== WebSocket.OPEN) return;
  if (playback.activeAudio?.turnId === turnId) playback.activeAudio.acknowledged = true;
  sendControl({
    type: 'playback_started', turn_id: turnId, audio_chunk_id: audioChunkId,
    client_monotonic_ms: atMs, clock_name: 'performance.now', acoustic_confirmed: false,
  });
  ui.playback.textContent = 'Browser output began · acoustic audibility unconfirmed';
  addTrace('Browser output started', `turn ${turnId} · chunk ${audioChunkId} · acoustics unconfirmed`);
}

function waitForOutputStart(source, scheduledContextTime, metadata, capturedGeneration, capturedSessionId) {
  let reported = false;
  const report = (monotonicMs) => {
    if (reported) return;
    reported = true;
    acknowledgePlayback(metadata.turnId, metadata.audioChunkId, capturedGeneration, capturedSessionId, monotonicMs);
  };
  if (typeof audioContext?.getOutputTimestamp === 'function') {
    const poll = () => {
      if (reported || !currentGuard(capturedGeneration, capturedSessionId)) return;
      if ((audioContext?.currentTime || 0) > scheduledContextTime + (source.buffer?.duration || 1) + 1) return;
      let output = null;
      try {
        output = audioContext.getOutputTimestamp();
      } catch {
        later(() => report(performance.now()), 0);
        return;
      }
      if (output && output.contextTime >= scheduledContextTime && output.performanceTime > 0) report(output.performanceTime);
      else requestAnimationFrame(poll);
    };
    requestAnimationFrame(poll);
  } else {
    const waitMs = Math.max(0, (scheduledContextTime - (audioContext?.currentTime || 0)) * 1000);
    later(() => report(performance.now()), waitMs);
  }
  source.addEventListener('ended', () => {
    markSourceEnded(playback, playbackSources, source, {
      echoTailMs: PLAYBACK_ECHO_TAIL_MS, nowMs: performance.now(), turnId: metadata.turnId,
    });
  }, { once: true });
}

// Playback scheduling cursor (AudioContext time); reset per audio_start.
let playbackTime = 0;

function playPcm16(data, capturedGeneration, capturedSessionId) {
  if (!shouldPlayFrame(capturedGeneration, generation)) return; // cancelled-turn bytes in flight
  // Protocol 2: the binary batch is only valid when a matching audio_chunk
  // envelope preceded it. Missing envelope = protocol 1 legacy output — play
  // it. Mismatched ids = ghost bytes after a cancel — drop and trace.
  const verdict = consumeChunkForBinary(playback, voice, capturedGeneration, generation);
  if (!verdict.play) {
    addTrace('Ghost audio dropped', verdict.reason || 'chunk mismatch');
    return;
  }
  if (!audioContext || data.byteLength < 2 || !playback.activeAudio) return;
  try {
    if (Number.isInteger(verdict.expectedBytes) && verdict.expectedBytes !== data.byteLength) {
      addTrace('Audio chunk size differs', `envelope ${verdict.expectedBytes} B · frame ${data.byteLength} B · playing anyway`);
    }
    const samples = decodePcm16(data);
    if (!samples.length) return;
    const buffer = audioContext.createBuffer(1, samples.length, 16_000);
    buffer.getChannelData(0).set(samples);
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    const scheduledAt = Math.max(audioContext.currentTime, playbackTime);
    source.start(scheduledAt);
    playbackTime = scheduledAt + buffer.duration;
    playbackSources.add(source);
    playback.activeAudio.pendingSources += 1;
    const played = { turnId: playback.activeAudio.turnId, audioChunkId: playback.activeAudio.audioChunkId };
    source.addEventListener('ended', () => {
      markSourceEnded(playback, playbackSources, source, {
        echoTailMs: PLAYBACK_ECHO_TAIL_MS, nowMs: performance.now(), turnId: played.turnId,
      });
    }, { once: true });
    if (!playback.activeAudio.scheduled) {
      playback.activeAudio.scheduled = true;
      vad.setAgentSpeaking(true, performance.now());
      waitForOutputStart(source, scheduledAt, played, capturedGeneration, capturedSessionId);
    }
  } catch {
    ui.playback.textContent = 'Playback failed before browser output started';
    if (playback.activeAudio?.turnId) sendControl({ type: 'playback_failed', turn_id: playback.activeAudio.turnId, error_code: 'BROWSER_AUDIO_ERROR' });
    showError('The answer text is available, but browser audio playback failed. No audible playback was confirmed.');
  }
}
// --- message handling: pure reducers decide, DOM applies ---
function applyEffects(effects) {
  for (const effect of effects) {
    switch (effect.kind) {
      case 'session-negotiated':
        ui.protocol.hidden = false;
        ui.protocol.textContent = `Protocol v${effect.protocol} · barge-in ${effect.bargeIn}`;
        ui.protocol.setAttribute('aria-label', `Voice protocol version ${effect.protocol}, barge-in mode ${effect.bargeIn}`);
        addTrace('Voice protocol negotiated', `v${effect.protocol} · barge-in ${effect.bargeIn}`);
        break;
      case 'turn-control':
        setState(`${effect.state || 'HOLD'} · ${effect.reason || ''}`.trim(), effect.state === 'HOLD' && isRecording);
        addTrace(`Turn ${effect.state || 'control'}`, `${effect.reason || 'decision'} · epoch ${voice.turnEpoch} · transcript v${voice.transcriptVersion}`);
        break;
      case 'deadline-armed':
        addTrace('Hold deadline armed', `${effect.holdDeadlineMs} ms absolute deadline`);
        break;
      case 'interrupt':
        // Semantic barge-in: the server judged a real floor grab while a
        // reply was playing. Stop playback NOW (ghost guard: the
        // generation bump also drops in-flight binary), keep the mic open.
        stopPlayback();
        setState('Interrupted — listening');
        addTrace('Interrupted by caller', `${effect.reason || 'taking_floor'} · taking_floor=${effect.takingFloor ?? 'unknown'} · mic stays open`);
        break;
      case 'show-error':
        showError(effect.message);
        break;
      case 'show-actions':
        ui.actions.hidden = false;
        ui.finish.disabled = false;
        ui.retry.disabled = false;
        ui.support.disabled = false;
        setState('Choose how to continue', false);
        ui.micHelp.textContent = 'The turn is paused for your choice. Your microphone and transcript remain active.';
        addTrace('Recovery choices available', 'finish · retry · request support');
        break;
      case 'render-turn':
        renderTurn(effect.result);
        ui.actions.hidden = true;
        if (effect.result?.turn_decision?.state) setState(`Turn ${effect.result.turn_decision.state.toLowerCase()}`);
        break;
      case 'handoff-recorded':
        showError('Your request for support was recorded for this demo. No live transfer or agent assignment occurred.');
        setState('Support request recorded');
        ui.actions.hidden = true;
        addTrace('Handoff request recorded', effect.reason || 'No live transfer occurred');
        break;
      case 'retry-ready':
        ui.transcript.textContent = 'Your transcript will appear here.';
        ui.transcript.classList.add('placeholder');
        ui.actions.hidden = true;
        setState('Retry ready', isRecording);
        addTrace('Retry started', `transcript v${voice.transcriptVersion}`);
        break;
      case 'provider-unavailable':
        showError(`${String(effect.provider || 'Provider').toUpperCase()} is unavailable (${effect.error || 'UNAVAILABLE'}). The request was not treated as successful.`);
        setState(`${effect.provider || 'Provider'} unavailable`);
        addTrace('Provider unavailable', `${effect.provider || 'unknown'} · ${effect.error || 'no code'}`);
        break;
      case 'playback-failed':
        ui.playback.textContent = 'Playback failed · acoustic output unconfirmed';
        showError('The answer text remains available, but audio playback failed.');
        addTrace('Playback failed', effect.reason || 'browser failure');
        break;
      case 'playback-ack':
        // Honest by construction: ttfa is the server-labelled client-clock
        // measurement, never an acoustic claim.
        ui.playback.textContent = 'Browser playback acknowledgement recorded · acoustic audibility unconfirmed';
        ui.ttfa.textContent = ttfaLabel();
        if (effect.ttfaStatus && effect.ttfaStatus !== 'unavailable' && Number.isFinite(effect.ttfaMs)) {
          addTrace('Playback acknowledgement recorded', `server ttfa ${effect.ttfaMs} ms (client clock, acoustics unconfirmed) · ${effect.ttfaStatus}`);
        } else {
          addTrace('Playback acknowledgement recorded', 'TTFA unavailable without aligned T0 and acoustic T6');
        }
        break;
      case 'voice-stopping':
        setState('Voice session stopping');
        addTrace('Voice session stopping', 'server acknowledged stop');
        break;
      case 'session-reset':
        setState('Session reset by server');
        addTrace('Session reset', 'server issued a replacement session');
        break;
      case 'restore-actions':
        ui.actions.hidden = false;
        ui.finish.disabled = false;
        ui.retry.disabled = false;
        ui.support.disabled = false;
        setState('Choose how to continue', false);
        addTrace('Recovery choices restored', `${effect.code} · transcript v${voice.transcriptVersion} resynced`);
        break;
      case 'voice-error':
        setState('Recoverable voice error');
        addTrace('Voice error', effect.code || 'unknown');
        break;
      case 'trace':
        addTrace(effect.label, effect.detail);
        break;
      default:
        break;
    }
  }
}

function handleMessage(message, capturedGeneration, capturedSessionId) {
  if (!eventIsCurrent(message, capturedGeneration, capturedSessionId)) return;
  if (message && typeof message.type === 'string') {
    switch (message.type) {
      case 'session':
        // Protocol negotiation runs through the reducer so the node harness
        // can assert the fallback; apply its effects like any other message.
        if (message.session?.session_id !== capturedSessionId) return;
        break;
      case 'voice_started':
        onVoiceStarted(capturedGeneration, capturedSessionId);
        return;
      case 'transcript_partial':
      case 'transcript_final':
        ui.transcript.textContent = message.text || '';
        ui.transcript.classList.remove('placeholder');
        if (message.type === 'transcript_final') addTrace('ASR final observed', `turn ${message.turn_id || 'unlinked'} · version ${voice.transcriptVersion}`);
        return;
      case 'audio_start': {
        const { replaced } = onAudioStart(playback, message, { echoTailMs: PLAYBACK_ECHO_TAIL_MS, nowMs: performance.now() });
        if (replaced) {
          // A new reply replaces the prior one — stop its sources now.
          for (const source of playbackSources) {
            try { source.stop(); } catch { /* already ended */ }
          }
          playbackSources.clear();
          addTrace('Prior reply audio replaced', `turn ${replaced.turnId || 'unknown'} stopped for turn ${message.turn_id || 'unknown'}`);
        }
        playbackTime = audioContext?.currentTime || 0;
        vad.setAgentSpeaking(true, performance.now());
        ui.playback.textContent = 'Server audio stream started · playback not yet confirmed';
        addTrace('Server audio started', `turn ${message.turn_id || 'unknown'} · chunk ${message.audio_chunk_id || 'unidentified'}`);
        return;
      }
      case 'audio_end': {
        const { applied, stale } = onAudioEnd(playback, message, { echoTailMs: PLAYBACK_ECHO_TAIL_MS, nowMs: performance.now() });
        ui.playback.textContent = playback.activeAudio?.acknowledged
          ? 'Browser output began · acoustic audibility unconfirmed'
          : 'Server audio finished · browser playback was not confirmed';
        ui.ttfa.textContent = ttfaLabel();
        if (stale) addTrace('Stale audio end ignored', `turn ${message.turn_id || 'unknown'} · chunk ${message.audio_chunk_id || 'unidentified'} is not current`);
        else if (applied) addTrace('Server audio stream ended', 'TTFA unavailable without aligned T0 and acoustic T6');
        return;
      }
      default:
        break;
    }
  }
  let effects = [];
  try {
    effects = reduceMessage(voice, message);
  } catch {
    // The voice-message handler never throws on unknown types (acceptance):
    // record and continue.
    addTrace('Ignored voice message', typeof message?.type === 'string' ? message.type : 'untyped');
    return;
  }
  applyEffects(effects);
}

// --- capture (AudioWorklet with ScriptProcessor fallback) ---
function onCaptureFrames(frames, capturedGeneration, capturedSessionId) {
  if (!currentGuard(capturedGeneration, capturedSessionId) || !socket || socket.readyState !== WebSocket.OPEN || !asrRunning) return;
  const nowMs = performance.now();
  const frame = frames[0];
  if (frame) {
    for (const vadEvent of vad.process(frame, nowMs)) {
      // Flag-aware barge-in: off = half-duplex gate (echo
      // dropped); semantic = graded guard stays, but a real onset during
      // playback sends with agent_speaking:true and keeps playing until
      // the server's interrupt verdict.
      const routed = routeVadOnset(vadEvent, { activeAudio: playback.activeAudio, nowMs, echoUntilMs: playback.echoUntilMs, bargeInMode: voice.bargeInMode });
      if (routed.action === 'send') sendControl(routed.event);
      else if (routed.action === 'drop-echo') {
        vad.reset();
        addTrace(routed.traceLabel, `${routed.traceDetail} · barge-in ${voice.bargeInMode}`);
        break;
      }
    }
  }
  if (!micResampler || !micBatcher) return;
  for (const chunk of frames) {
    for (const batch of micBatcher.push(micResampler.process(chunk))) {
      if (socket.readyState === WebSocket.OPEN) socket.send(batch.buffer);
    }
  }
}

async function attachAudioProcessor(capturedGeneration, capturedSessionId) {
  if (!mediaStream || !audioContext || captureHandle) return;
  micResampler = createResampler(audioContext.sampleRate);
  micBatcher = createPcmBatcher();
  const handle = await attachCapture({
    audioContext,
    mediaStream,
    onFrames: (frames) => onCaptureFrames(frames, capturedGeneration, capturedSessionId),
  });
  if (!isTokenCurrent(mic, mic.token) && !isRecording) {
    handle.disconnect();
    return;
  }
  captureHandle = handle;
  addTrace('Microphone capture attached', `${handle.kind} · 16 kHz mono PCM`);
  if (handle.kind === 'script') {
    addTrace('Capture fallback', 'ScriptProcessor fallback in use (AudioWorklet unavailable)');
  }
  if (handle.workletError) {
    addTrace('AudioWorklet unavailable', 'fell back to ScriptProcessor capture');
  }
}

function onVoiceStarted(capturedGeneration, capturedSessionId) {
  asrRunning = true;
  isRecording = true;
  setState('Listening', true);
  ui.mic.classList.add('recording');
  ui.mic.setAttribute('aria-label', 'Stop microphone capture');
  ui.micLabel.textContent = 'Listening…';
  ui.micHelp.textContent = 'Speak naturally. Pause is observed, not used to automatically finish your turn.';
  attachAudioProcessor(capturedGeneration, capturedSessionId).catch((error) => {
    // A throw here must surface — never leave "Listening" with no audio.
    if (!currentGuard(capturedGeneration, capturedSessionId)) return;
    setState('Microphone unavailable');
    ui.mic.classList.remove('recording');
    ui.mic.setAttribute('aria-label', 'Start microphone');
    showError(micErrorMessage(error));
    addTrace('Capture failed', micErrorMessage(error));
  });
  addTrace('Microphone capture started', '16 kHz mono PCM');
}

// --- socket lifecycle ---
function connectVoiceSocket(capturedGeneration, capturedSessionId) {
  return new Promise((resolve, reject) => {
    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    let ws;
    try {
      ws = new WebSocket(`${wsProtocol}//${location.host}/ws/sessions/${encodeURIComponent(capturedSessionId)}`);
    } catch (error) {
      reject(error instanceof Error ? error : new Error('Speech connection failed.'));
      return;
    }
    if (socket && socket.readyState < WebSocket.CLOSING) {
      try { socket.close(); } catch { /* superseded */ }
    }
    socket = ws;
    ws.binaryType = 'arraybuffer';
    let settled = false;
    ws.onmessage = (event) => {
      if (!currentGuard(capturedGeneration, capturedSessionId) || socket !== ws) return;
      if (event.data instanceof ArrayBuffer) playPcm16(event.data, capturedGeneration, capturedSessionId);
      else {
        try {
          const parsed = JSON.parse(event.data);
          // Protocol 2 audio_chunk envelopes stash the pending chunk; the
          // reducer records them, and the next binary batch is validated by
          // consumeChunkForBinary() in playPcm16.
          handleMessage(parsed, capturedGeneration, capturedSessionId);
        } catch { /* ignore malformed frames */ }
      }
    };
    ws.onerror = () => {
      if (!settled) {
        settled = true;
        if (currentGuard(capturedGeneration, capturedSessionId)) showError('Could not connect to the speech service. Check its configuration and retry.');
        reject(new Error('Speech connection failed.'));
      }
    };
    ws.onclose = (event) => {
      if (!settled) {
        settled = true;
        reject(new Error('Speech connection failed.'));
      }
      if (!currentGuard(capturedGeneration, capturedSessionId) || socket !== ws) return;
      handleSocketClose(event, capturedGeneration, capturedSessionId);
    };
    ws.onopen = () => {
      if (!currentGuard(capturedGeneration, capturedSessionId)) { ws.close(); return; }
      onWsOpen(wsLife, reconnect);
      ws.send(JSON.stringify(startMessage(capturedSessionId, { resolutionMs: measureClockResolution() })));
      if (!settled) {
        settled = true;
        resolve();
      }
    };
  });
}

function handleSocketClose(event, capturedGeneration, capturedSessionId) {
  const code = event?.code ?? 1005;
  const reason = event?.reason ?? '';
  const { auto, description } = onWsClose(wsLife, code, reason);
  asrRunning = false;
  socket = null;
  addTrace('Voice connection closed', `${description} · code ${code}`);
  if (code === 1008 || code === 1009) {
    // Protocol §3 close codes: audio-limit breaches end the socket after a
    // named error. Surface the friendly action, not a bare code.
    setState('Voice connection closed');
    showError(`Voice connection closed (${description}).`);
    if (code === 1008) {
      ui.micLabel.textContent = 'Reset session to continue';
      ui.mic.setAttribute('aria-label', 'Reset session: the session audio budget was reached');
    } else {
      ui.micLabel.textContent = 'Resume voice';
      ui.mic.setAttribute('aria-label', 'Resume voice connection');
    }
    wsLife.resumeOffered = true;
    return;
  }
  if (!isRecording) {
    setState('Voice connection closed');
    return;
  }
  if (auto && !manualStop) {
    const delay = nextBackoffDelay(reconnect);
    if (delay !== null) {
      setState('Reconnecting voice…');
      showError(`Voice connection dropped (${description}). Reconnecting…`);
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        if (!currentGuard(capturedGeneration, capturedSessionId)) return;
        resumeVoice(capturedGeneration, capturedSessionId);
      }, delay);
      oneShotTimers.add(reconnectTimer);
      return;
    }
  }
  // Manual resume: keep the mic UI live so one click restarts capture.
  setState('Voice connection closed');
  showError(`Voice connection closed (${description}). Press Resume voice to reconnect.`);
  ui.micLabel.textContent = 'Resume voice';
  ui.mic.setAttribute('aria-label', 'Resume voice connection');
  wsLife.resumeOffered = true;
}

async function resumeVoice(capturedGeneration, capturedSessionId) {
  if (!currentGuard(capturedGeneration, capturedSessionId)) return;
  generation += 1; // fresh generation + fresh start, so stale frames cannot play
  const freshGeneration = generation;
  clearError();
  try {
    await connectVoiceSocket(freshGeneration, capturedSessionId);
    if (!currentGuard(freshGeneration, capturedSessionId)) return;
    asrRunning = true;
    setState('Listening', true);
    addTrace('Voice reconnected', 'fresh start message sent');
  } catch {
    if (!currentGuard(freshGeneration, capturedSessionId)) return;
    setState('Voice connection closed');
    showError('Voice reconnection failed. Press Resume voice to retry.');
    ui.micLabel.textContent = 'Resume voice';
    wsLife.resumeOffered = true;
  }
}

function measureClockResolution() {
  let previous = performance.now();
  let minimum = Number.POSITIVE_INFINITY;
  for (let index = 0; index < 256; index += 1) {
    const current = performance.now();
    const delta = current - previous;
    if (delta > 0) minimum = Math.min(minimum, delta);
    previous = current;
  }
  return Number.isFinite(minimum) ? Number(minimum.toFixed(6)) : null;
}

// --- mic start/stop state machine ---
async function startMicrophone() {
  if (!micTransitionAllowed(mic)) return; // rapid-toggle race guard
  clearError();
  const capturedSessionId = session?.session_id;
  if (!capturedSessionId) return;
  const token = beginStart(mic);
  if (token === null) return;
  const capturedGeneration = generation;
  manualStop = false;
  ui.mic.disabled = true; // disabled during transition
  ui.mic.setAttribute('aria-label', 'Starting microphone…');
  let stream = null;
  let context = null;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser does not provide microphone capture. Use a current browser over localhost or HTTPS.');
    stream = await navigator.mediaDevices.getUserMedia({ audio: MIC_CONSTRAINTS });
    if (!isTokenCurrent(mic, token) || !currentGuard(capturedGeneration, capturedSessionId)) {
      stream.getTracks().forEach((track) => track.stop());
      abortTransition(mic, token);
      renderProviders(config?.providers || {});
      return;
    }
    const granted = grantedConstraints(stream);
    addTrace('Microphone constraints granted', ['echoCancellation', 'noiseSuppression', 'autoGainControl'].map((k) => `${k}=${granted[k] ?? 'unknown'}`).join(' · '));
    mediaStream = stream;
    stream = null;
    context = new AudioContext();
    await context.resume();
    if (!isTokenCurrent(mic, token) || !currentGuard(capturedGeneration, capturedSessionId)) {
      if (context.state !== 'closed') await context.close().catch(() => {});
      if (mediaStream) { mediaStream.getTracks().forEach((track) => track.stop()); mediaStream = null; }
      abortTransition(mic, token);
      renderProviders(config?.providers || {});
      return;
    }
    audioContext = context;
    context = null;
    vad.reset();
    if (!socket || socket.readyState !== WebSocket.OPEN) await connectVoiceSocket(capturedGeneration, capturedSessionId);
    else socket.send(JSON.stringify(startMessage(capturedSessionId, { resolutionMs: measureClockResolution() })));
    if (!isTokenCurrent(mic, token) || !currentGuard(capturedGeneration, capturedSessionId) || !socket || socket.readyState !== WebSocket.OPEN) {
      abortTransition(mic, token);
      renderProviders(config?.providers || {});
      return;
    }
    if (!finishStart(mic, token)) return; // superseded — stop owns the UI now
    isRecording = true;
    renderProviders(config?.providers || {});
  } catch (error) {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    if (context && context.state !== 'closed') await context.close().catch(() => {});
    if (isTokenCurrent(mic, token)) {
      await stopMicrophoneInternal(token, false);
      if (currentGuard(capturedGeneration, capturedSessionId)) {
        showError(micErrorMessage(error));
        setState('Microphone unavailable');
      }
    }
  } finally {
    if (isTokenCurrent(mic, token) && mic.transitioning) abortTransition(mic, token);
    renderProviders(config?.providers || {});
  }
}

async function stopMicrophoneInternal(token, notifyServer = true) {
  isRecording = false;
  asrRunning = false;
  ui.mic.classList.remove('recording');
  ui.mic.setAttribute('aria-label', 'Start microphone');
  ui.micLabel.textContent = 'Start speaking';
  if (captureHandle) {
    try { captureHandle.disconnect(); } catch { /* already torn down */ }
    captureHandle = null;
  }
  micResampler = null;
  micBatcher = null;
  if (mediaStream) { mediaStream.getTracks().forEach((track) => track.stop()); mediaStream = null; }
  vad.reset();
  stopPlayback(); // stopping the mic tears down playback state too
  if (socket?.readyState === WebSocket.OPEN && notifyServer) {
    try { socket.send(JSON.stringify({ type: 'stop' })); } catch { /* closing */ }
  }
  if (audioContext && audioContext.state !== 'closed') await audioContext.close().catch(() => {});
  // Only null the context this transition owns: a fresh start may have
  // adopted a new context while this stop was awaiting close.
  if (isTokenCurrent(mic, token)) audioContext = null;
  if (ui.mic.disabled && !mic.transitioning) {
    renderProviders(config?.providers || {});
  } else {
    ui.micHelp.textContent = 'Silence never finishes a turn automatically. Use the explicit turn controls when available.';
    ui.micLabel.textContent = 'Start speaking';
  }
  setState('Microphone stopped');
}

async function stopMicrophone(notifyServer = true) {
  if (!micTransitionAllowed(mic)) return;
  const token = beginStop(mic);
  if (token === null) return;
  ui.mic.disabled = true;
  manualStop = true;
  clearAllTimers();
  try {
    await stopMicrophoneInternal(token, notifyServer);
    finishStop(mic, token);
  } finally {
    if (isTokenCurrent(mic, token) && mic.transitioning) abortTransition(mic, token);
    manualStop = false;
    renderProviders(config?.providers || {});
  }
}

// --- text + recovery + support ---
async function sendText(event) {
  event.preventDefault();
  const text = ui.input.value.trim();
  if (!text || !session) return;
  const submit = ui.form.querySelector('button[type="submit"]');
  if (submit?.disabled) return; // Enter must honour the disabled state
  const capturedGeneration = generation;
  const capturedSessionId = session.session_id;
  clearError();
  setState('Submitting diagnostic text…');
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(capturedSessionId)}/turns`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, source: 'manual' }),
    });
    const result = await response.json();
    if (!currentGuard(capturedGeneration, capturedSessionId)) return;
    if (!response.ok) throw new Error(result.detail || 'The request could not be completed.');
    ui.input.value = ''; // clear only after success
    renderTurn(result);
  } catch (error) {
    if (!currentGuard(capturedGeneration, capturedSessionId)) return;
    showError(error?.message || 'The request could not be completed.');
    setState('Unavailable');
  }
}

function sendRecoveryAction(type) {
  const payload = { type, transcript_version: voice.transcriptVersion, turn_epoch: voice.turnEpoch, hold_episode_id: voice.currentHoldEpisodeId };
  if (type === 'handoff_request') payload.reason = ui.supportReason.value || 'Caller requested human support';
  if (sendControl(payload)) {
    addTrace('Caller action', `${type} · transcript v${voice.transcriptVersion}`);
    ui.actions.hidden = true;
    setState(type === 'manual_finish' ? 'Finishing current turn' : type === 'caller_retry' ? 'Retry requested' : 'Support requested');
  }
}

function clearConversation() {
  ui.transcript.textContent = 'Your transcript will appear here.';
  ui.transcript.classList.add('placeholder');
  ui.answerCard.hidden = true;
  ui.evidence.replaceChildren();
  ui.input.value = '';
  ui.actions.hidden = true;
  ui.trace.replaceChildren();
  ui.playback.textContent = 'No browser playback observed';
  ui.ttfa.textContent = ttfaLabel();
  voice.currentTurnId = null;
  voice.currentHoldEpisodeId = null;
  voice.transcriptVersion = 0;
  voice.turnEpoch = 0;
  voice.actionsVisible = false;
}

async function resetSession() {
  if (resetSession.inFlight) return; // double-reset guard
  resetSession.inFlight = true;
  ui.reset.disabled = true;
  generation += 1;
  const resetGeneration = generation;
  const previousSessionId = session?.session_id;
  clearAllTimers();
  stopPlayback();
  manualStop = true;
  if (mic.transitioning) {
    mic.token += 1; // supersede any in-flight start/stop
    mic.transitioning = false;
    mic.status = 'idle';
  }
  await stopMicrophoneInternal(mic.token, false);
  mic.transitioning = false;
  mic.status = 'idle';
  if (socket && socket.readyState < WebSocket.CLOSING) {
    try { socket.close(); } catch { /* already closing */ }
  }
  socket = null;
  resetReconnect(reconnect);
  clearError();
  ui.support.disabled = true;
  ui.supportStatus.textContent = '';
  setState('Resetting session…');
  try {
    const response = previousSessionId
      ? await fetch(`/api/sessions/${encodeURIComponent(previousSessionId)}/reset`, { method: 'POST' })
      : await fetch('/api/sessions', { method: 'POST' });
    if (resetGeneration !== generation) return;
    if (!response.ok) throw new Error('Could not reset the demo session.');
    session = await response.json();
    ui.support.disabled = !supportRequestEnabled(session);
    ui.sessionId.textContent = session.session_id;
    ui.pill.textContent = 'Session active';
    ui.pill.className = 'pill online';
    clearConversation();
    setState('Ready');
    renderProviders(config?.providers || {});
    addTrace('Session reset', 'new session scope; previous callbacks invalidated');
  } catch (error) {
    if (resetGeneration !== generation) return;
    ui.support.disabled = !supportRequestEnabled(session);
    showError(error?.message || 'Could not reset the demo session.');
    setState('Reset unavailable');
  } finally {
    manualStop = false;
    resetSession.inFlight = false;
    ui.reset.disabled = false;
  }
}

async function requestSupport() {
  if (!supportRequestEnabled(session)) {
    ui.supportStatus.textContent = 'An active session is required before a support request can be recorded.';
    return;
  }
  // When a hold episode is active and the voice socket is live, resolve
  // the episode via WS handoff_request; otherwise use the HTTP handoff.
  const route = routeHandoff(voice, socket?.readyState === WebSocket.OPEN);
  if (route === 'ws') {
    sendRecoveryAction('handoff_request');
    ui.supportStatus.textContent = 'Support request sent on the voice connection.';
    return;
  }
  const capturedGeneration = generation;
  const capturedSessionId = session.session_id;
  ui.support.disabled = true;
  ui.supportStatus.textContent = 'Recording your request…';
  try {
    const response = await submitSupportRequest({
      fetchImpl: fetch,
      sessionId: capturedSessionId,
      reason: handoffRequestPayload(voice, ui.supportReason.value).reason,
      isCurrent: () => currentGuard(capturedGeneration, capturedSessionId),
    });
    if (response.status === 'STALE_DISCARDED' || !currentGuard(capturedGeneration, capturedSessionId)) return;
    const result = response.result;
    ui.supportStatus.textContent = result.message || 'Your support request was recorded for this demo. No live transfer or agent assignment occurred.';
    setState('Support request recorded');
    addTrace('Support request recorded', result.request_id || 'session request; no live transfer or agent assignment');
  } catch (error) {
    if (!currentGuard(capturedGeneration, capturedSessionId)) return;
    ui.supportStatus.textContent = error?.message || 'The support request could not be recorded. Please retry.';
  } finally {
    if (currentGuard(capturedGeneration, capturedSessionId)) ui.support.disabled = !supportRequestEnabled(session);
  }
}

let initializeAttempts = 0;

function initialize() {
  const initialGeneration = generation;
  initializeAttempts += 1;
  const attempt = initializeAttempts;
  Promise.all([fetch('/api/config'), fetch('/api/sessions', { method: 'POST' })])
    .then(async ([configResponse, sessionResponse]) => {
      if (!configResponse.ok || !sessionResponse.ok) throw new Error('The demo server did not start correctly.');
      const [nextConfig, nextSession] = await Promise.all([configResponse.json(), sessionResponse.json()]);
      if (initialGeneration !== generation) return;
      config = nextConfig;
      session = nextSession;
      initializeAttempts = 0;
      ui.support.disabled = !supportRequestEnabled(session);
      ui.sessionId.textContent = session.session_id;
      ui.pill.textContent = 'Session active';
      ui.pill.className = 'pill online';
      ui.ttfa.textContent = ttfaLabel();
      renderProviders(config.providers);
      setState('Ready');
    })
    .catch((error) => {
      if (initialGeneration !== generation) return;
      ui.pill.textContent = 'Server unavailable';
      ui.pill.className = 'pill offline';
      showError(error?.message || 'Could not connect to the demo server.');
      ui.micLabel.textContent = 'Voice demo unavailable';
      ui.micHelp.textContent = 'Start the local application server, then retry.';
      if (attempt < 3) later(initialize, 2000 * attempt); // initialize retry
    });
}

ui.mic.addEventListener('click', () => {
  if (!micTransitionAllowed(mic)) return;
  if (wsLife.resumeOffered && !isRecording) {
    wsLife.resumeOffered = false;
    if (session?.session_id) resumeVoice(generation, session.session_id);
    else startMicrophone();
  } else if (isRecording) stopMicrophone(true);
  else startMicrophone();
});
ui.form.addEventListener('submit', sendText);
ui.reset.addEventListener('click', resetSession);
ui.finish.addEventListener('click', () => sendRecoveryAction('manual_finish'));
ui.retry.addEventListener('click', () => sendRecoveryAction('caller_retry'));
ui.support.addEventListener('click', requestSupport);
if (ui.settingsToggle && ui.settings) {
  ui.settingsToggle.addEventListener('click', () => {
    ui.settings.hidden = !ui.settings.hidden;
    ui.settingsToggle.setAttribute('aria-expanded', String(!ui.settings.hidden));
  });
}
buildVadSettings();
initialize();
