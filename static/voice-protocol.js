// Voice-protocol message handling: pure per-type reducers over an explicit
// UI-state snapshot. The DOM lives in app.js; this module
// decides WHAT each server message means so the decisions are unit-testable
// and the handler provably never throws on unknown types.
//
// Server message types (docs/VOICE_PROTOCOL.md §1, src/voxcare/voice/protocol.py):
//   session, voice_started, transcript_partial, transcript_final,
//   turn_control, recovery_required, turn, audio_chunk,
//   audio_start, audio_end, interrupt, retry_ready,
//   handoff_request_recorded, provider_unavailable,
//   playback_ack_recorded, playback_failed_recorded,
//   voice_stopping, session_reset, error
// HANDLER_OWNER records the client decision for EVERY server name: 'reducer'
// means reduceMessage() below owns it, 'app' means the DOM glue in app.js owns
// it deliberately (streaming audio/transcript frames). A server name with no
// entry here fails tests/test_protocol_contract.py — decide first, then add.
// Retired names (turn_deadline, policy_result, playback_failed) are NOT known:
// deadlines arrive inside turn_control, results inside turn, and the server
// only ever echoes playback_failed_recorded.
//
// Error codes that must RESTORE recovery actions + resync transcriptVersion:
// STALE_RECOVERY_ACTION, WAIT_FOR_LOCAL_VAD_QUIET.

export const PROTOCOL_VERSION = 2;

export const HANDLER_OWNER = Object.freeze({
  session: 'reducer',
  voice_started: 'app',
  transcript_partial: 'app',
  transcript_final: 'app',
  turn_control: 'reducer',
  recovery_required: 'reducer',
  turn: 'reducer',
  audio_chunk: 'reducer',
  audio_start: 'app',
  audio_end: 'app',
  interrupt: 'reducer',
  retry_ready: 'reducer',
  handoff_request_recorded: 'reducer',
  provider_unavailable: 'reducer',
  playback_ack_recorded: 'reducer',
  playback_failed_recorded: 'reducer',
  voice_stopping: 'reducer',
  session_reset: 'reducer',
  error: 'reducer',
});

export const KNOWN_MESSAGE_TYPES = Object.freeze(Object.keys(HANDLER_OWNER));

export function isKnownMessageType(type) {
  return KNOWN_MESSAGE_TYPES.includes(type);
}

export const RESTORING_ERROR_CODES = Object.freeze(['STALE_RECOVERY_ACTION', 'WAIT_FOR_LOCAL_VAD_QUIET']);

export function createVoiceState() {
  return {
    currentTurnId: null,
    currentHoldEpisodeId: null,
    transcriptVersion: 0,
    turnEpoch: 0,
    actionsVisible: false,
    protocolVersion: null, // negotiated from the session hello (2, or 1 fallback)
    bargeInMode: 'off', // 'semantic' only when the hello advertises it
    pendingChunk: null, // last protocol-2 audio_chunk envelope, consumed by the next binary frame
  };
}

/**
 * Reduce one server message into a list of UI effects. Never throws on
 * unknown message types — they produce zero effects. Each effect is a plain
 * `{ kind, ... }` record the DOM layer applies; tests assert on effects.
 */
export function reduceMessage(state, message) {
  const effects = [];
  if (!message || typeof message.type !== 'string') return effects;
  if (Number.isInteger(message.turn_epoch)) state.turnEpoch = Math.max(state.turnEpoch, message.turn_epoch);
  if (Number.isInteger(message.transcript_version)) {
    state.transcriptVersion = Math.max(state.transcriptVersion, message.transcript_version);
  }
  switch (message.type) {
    case 'session': {
      // Protocol negotiation: protocol 2 adds audio_chunk envelopes
      // and interrupt; anything else is a graceful v1 fallback (envelope-less
      // binary frames keep playing as legacy output).
      state.protocolVersion = message.protocol === PROTOCOL_VERSION ? PROTOCOL_VERSION : 1;
      state.bargeInMode = message.barge_in === 'semantic' ? 'semantic' : 'off';
      effects.push({ kind: 'session-negotiated', protocol: state.protocolVersion, bargeIn: state.bargeInMode });
      break;
    }
    case 'turn_control':
      state.currentTurnId = message.turn_id || state.currentTurnId;
      if (message.state === 'FALLBACK_RESOLVED' || message.state === 'COMMIT') state.currentHoldEpisodeId = null;
      else state.currentHoldEpisodeId = message.hold_episode_id || state.currentHoldEpisodeId;
      effects.push({ kind: 'turn-control', state: message.state, reason: message.reason });
      if (Number.isInteger(message.hold_deadline_ms)) {
        // The old turn_deadline message is dead; the deadline now arrives here.
        effects.push({ kind: 'deadline-armed', holdDeadlineMs: message.hold_deadline_ms });
      }
      if (message.state === 'DEGRADED') effects.push({ kind: 'show-error', message: `Turn handling is degraded (${message.reason}). Your transcript is preserved; choose an explicit recovery action when ready.` });
      break;
    case 'recovery_required':
      state.currentHoldEpisodeId = message.hold_episode_id || state.currentHoldEpisodeId;
      state.actionsVisible = true;
      effects.push({ kind: 'show-actions' });
      break;
    case 'turn':
      state.actionsVisible = false;
      effects.push({ kind: 'render-turn', result: message.result || message });
      break;
    case 'audio_chunk':
      // Protocol 2 envelope ahead of exactly one binary PCM batch:
      // stash it so the next binary frame can prove it belongs to the live
      // stream. The DOM glue (expectChunk) validates it against activeAudio.
      state.pendingChunk = {
        turnId: message.turn_id ?? null,
        audioChunkId: message.audio_chunk_id ?? null,
        chunkIndex: message.chunk_index ?? null,
        byteLength: message.byte_length ?? null,
      };
      break;
    case 'interrupt':
      // Semantic barge-in only: the server judged taking_floor while a
      // reply was playing. The DOM layer stops playback NOW and keeps the mic
      // open — no other effect touches capture state.
      effects.push({ kind: 'interrupt', reason: message.reason || 'taking_floor', takingFloor: message.taking_floor ?? null });
      break;
    case 'handoff_request_recorded':
      state.actionsVisible = false;
      effects.push({ kind: 'handoff-recorded', reason: message.reason });
      break;
    case 'retry_ready':
      state.currentTurnId = message.turn_id || null;
      state.currentHoldEpisodeId = message.hold_episode_id || null;
      state.transcriptVersion = Number.isInteger(message.transcript_version)
        ? message.transcript_version
        : state.transcriptVersion + 1;
      state.actionsVisible = false;
      effects.push({ kind: 'retry-ready' });
      break;
    case 'provider_unavailable':
      effects.push({ kind: 'provider-unavailable', provider: message.provider, error: message.error });
      break;
    case 'playback_failed_recorded':
      effects.push({ kind: 'playback-failed', reason: message.reason });
      break;
    case 'playback_ack_recorded':
      // Honest by construction: ttfa is a client-clock measurement the server
      // labels, never an acoustic confirmation. Null unless clocks align.
      effects.push({ kind: 'playback-ack', ttfaMs: message.ttfa_ms ?? null, ttfaStatus: message.ttfa_status || 'unavailable' });
      break;
    case 'voice_stopping':
      effects.push({ kind: 'voice-stopping' });
      break;
    case 'session_reset':
      effects.push({ kind: 'session-reset', session: message.session || null });
      break;
    case 'error': {
      const code = message.code || 'UNKNOWN';
      if (RESTORING_ERROR_CODES.includes(code)) {
        // The rejection dead-ends the UI unless actions are restored and
        // the version is resynced from the server's authoritative copy. Never
        // move the local version backwards: a rejection echoes the version
        // the client already sent, which may predate a retry_ready bump.
        state.actionsVisible = true;
        const authoritative = Number.isInteger(message.transcript_version)
          ? message.transcript_version
          : (Number.isInteger(message.expected_transcript_version) ? message.expected_transcript_version : null);
        if (authoritative !== null) state.transcriptVersion = Math.max(state.transcriptVersion, authoritative);
        effects.push({ kind: 'restore-actions', code });
      } else {
        effects.push({ kind: 'show-error', message: `Voice session issue: ${code}. Retry the microphone or reset this session.` });
      }
      effects.push({ kind: 'voice-error', code });
      break;
    }
    default:
      if (message.event_type) effects.push({ kind: 'trace', label: message.event_type, detail: message.outcome || 'observed' });
      break;
  }
  return effects;
}

/**
 * Route the support button: when a hold episode is active the WS
 * `handoff_request` path resolves the episode server-side; otherwise fall
 * back to the HTTP handoff endpoint. Returns 'ws' or 'http'.
 */
export function routeHandoff(state, socketOpen) {
  if (state.currentHoldEpisodeId && socketOpen) return 'ws';
  return 'http';
}

export function handoffRequestPayload(state, reason) {
  return {
    type: 'handoff_request',
    transcript_version: state.transcriptVersion,
    turn_epoch: state.turnEpoch,
    hold_episode_id: state.currentHoldEpisodeId,
    reason: String(reason || '').trim().slice(0, 200) || 'Caller requested human support',
  };
}
