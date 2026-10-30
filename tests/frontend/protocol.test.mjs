import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createVoiceState, reduceMessage, routeHandoff, handoffRequestPayload,
  isKnownMessageType, RESTORING_ERROR_CODES, KNOWN_MESSAGE_TYPES, HANDLER_OWNER,
} from '../../static/voice-protocol.js';

test('STALE_RECOVERY_ACTION restores actions and resyncs the version', () => {
  const state = createVoiceState();
  state.transcriptVersion = 7;
  const effects = reduceMessage(state, { type: 'error', code: 'STALE_RECOVERY_ACTION', transcript_version: 9 });
  assert.equal(state.actionsVisible, true);
  assert.equal(state.transcriptVersion, 9);
  assert.ok(effects.some((e) => e.kind === 'restore-actions'));
});

test('WAIT_FOR_LOCAL_VAD_QUIET restores actions without moving the version back', () => {
  const state = createVoiceState();
  state.transcriptVersion = 9;
  const effects = reduceMessage(state, { type: 'error', code: 'WAIT_FOR_LOCAL_VAD_QUIET', transcript_version: 4 });
  assert.equal(state.actionsVisible, true);
  assert.equal(state.transcriptVersion, 9);
  assert.ok(effects.some((e) => e.kind === 'restore-actions'));
});

test('generic voice errors do not restore recovery actions', () => {
  const state = createVoiceState();
  const effects = reduceMessage(state, { type: 'error', code: 'AUDIO_FRAME_TOO_LARGE' });
  assert.equal(state.actionsVisible, false);
  assert.ok(effects.some((e) => e.kind === 'show-error'));
});

test('every restoring code is covered', () => {
  assert.deepEqual([...RESTORING_ERROR_CODES], ['STALE_RECOVERY_ACTION', 'WAIT_FOR_LOCAL_VAD_QUIET']);
});

test('handoff routes to WS when a hold episode is active and the socket is live', () => {
  const state = createVoiceState();
  state.currentHoldEpisodeId = 'hold-1';
  assert.equal(routeHandoff(state, true), 'ws');
  assert.equal(routeHandoff(state, false), 'http');
  assert.equal(routeHandoff(createVoiceState(), true), 'http');
});

test('handoff_request payload carries episode, version, and a bounded reason', () => {
  const state = createVoiceState();
  state.currentHoldEpisodeId = 'hold-9';
  state.transcriptVersion = 12;
  state.turnEpoch = 3;
  const payload = handoffRequestPayload(state, 'x'.repeat(500));
  assert.equal(payload.type, 'handoff_request');
  assert.equal(payload.hold_episode_id, 'hold-9');
  assert.equal(payload.transcript_version, 12);
  assert.equal(payload.turn_epoch, 3);
  assert.ok(payload.reason.length <= 200);
});

test('recovery_required exposes actions; turn result hides them', () => {
  const state = createVoiceState();
  reduceMessage(state, { type: 'recovery_required', hold_episode_id: 'h-1' });
  assert.equal(state.actionsVisible, true);
  assert.equal(state.currentHoldEpisodeId, 'h-1');
  reduceMessage(state, { type: 'turn', result: { outcome: 'ANSWERED' } });
  assert.equal(state.actionsVisible, false);
});

test('retry_ready resyncs the version, or bumps it when the server omits it', () => {
  const synced = createVoiceState();
  reduceMessage(synced, { type: 'retry_ready', turn_id: 't', transcript_version: 6, hold_episode_id: null });
  assert.equal(synced.transcriptVersion, 6);
  const bumped = createVoiceState();
  bumped.transcriptVersion = 6;
  reduceMessage(bumped, { type: 'retry_ready', turn_id: 't', hold_episode_id: null });
  assert.equal(bumped.transcriptVersion, 7);
});

test('handler never throws on unknown, untyped, null, or retired messages', () => {
  const state = createVoiceState();
  for (const message of [
    { type: 'some_future_type', foo: 1 },
    { type: 'turn_deadline', hold_deadline_ms: 9000 }, // dead: deadlines now arrive inside turn_control
    { type: 'policy_result', result: { outcome: 'ANSWERED' } }, // dead: results arrive as turn
    { type: 'playback_failed', reason: 'old' }, // dead: server echoes playback_failed_recorded
    { event_type: 'custom', outcome: 'x' },
    {},
    null,
    undefined,
    'audio_start',
    42,
  ]) {
    assert.doesNotThrow(() => reduceMessage(state, message));
  }
  // Unknown and retired types produce no state churn.
  const before = { ...state };
  reduceMessage(state, { type: 'some_future_type' });
  assert.deepEqual({ ...state }, before);
  reduceMessage(state, { type: 'turn_deadline', hold_deadline_ms: 9000 });
  assert.deepEqual({ ...state }, before);
  reduceMessage(state, { type: 'policy_result', result: {} });
  assert.deepEqual({ ...state }, before);
});

test('contract: every protocol-v2 server type has a client case (no dead names)', () => {
  // Authoritative server names (docs/VOICE_PROTOCOL.md §1): retirements
  // (turn_deadline, policy_result, playback_failed) must NOT be known —
  // the Python contract test enforces the same list both ways.
  const serverEmitted = [
    'session', 'voice_started', 'transcript_partial', 'transcript_final',
    'turn_control', 'recovery_required', 'turn', 'audio_chunk', 'audio_start',
    'audio_end', 'interrupt', 'provider_unavailable', 'playback_ack_recorded',
    'playback_failed_recorded', 'voice_stopping', 'session_reset', 'error',
    'retry_ready', 'handoff_request_recorded',
  ];
  for (const type of serverEmitted) {
    assert.equal(isKnownMessageType(type), true, `client has no case for server type ${type}`);
  }
  for (const dead of ['turn_deadline', 'policy_result', 'playback_failed']) {
    assert.equal(isKnownMessageType(dead), false, `dead client case ${dead} must stay retired`);
    assert.equal(HANDLER_OWNER[dead], undefined, `dead name ${dead} must have no handler owner`);
  }
});

test('session hello negotiates protocol 2 and falls back to v1', () => {
  const modern = createVoiceState();
  const effectsModern = reduceMessage(modern, { type: 'session', session: { session_id: 's' }, protocol: 2, barge_in: 'semantic' });
  assert.equal(modern.protocolVersion, 2);
  assert.equal(modern.bargeInMode, 'semantic');
  assert.ok(effectsModern.some((e) => e.kind === 'session-negotiated' && e.protocol === 2));
  const legacy = createVoiceState();
  const effectsLegacy = reduceMessage(legacy, { type: 'session', session: { session_id: 's' } });
  assert.equal(legacy.protocolVersion, 1);
  assert.equal(legacy.bargeInMode, 'off');
  assert.ok(effectsLegacy.some((e) => e.kind === 'session-negotiated' && e.protocol === 1));
});

test('interrupt effect stops playback without touching capture state', () => {
  const state = createVoiceState();
  state.transcriptVersion = 5;
  const effects = reduceMessage(state, { type: 'interrupt', reason: 'taking_floor', taking_floor: 0.9 });
  assert.ok(effects.some((e) => e.kind === 'interrupt' && e.reason === 'taking_floor'));
  assert.equal(state.transcriptVersion, 5);
  assert.equal(state.actionsVisible, false);
});

test('audio_chunk envelopes stash the pending chunk; deadline arrives via turn_control', () => {
  const state = createVoiceState();
  const envelope = reduceMessage(state, { type: 'audio_chunk', turn_id: 'T1', audio_chunk_id: 'C1', chunk_index: 0, byte_length: 320 });
  assert.deepEqual(envelope, []);
  assert.deepEqual(state.pendingChunk, { turnId: 'T1', audioChunkId: 'C1', chunkIndex: 0, byteLength: 320 });
  const turn = createVoiceState();
  const effects = reduceMessage(turn, { type: 'turn_control', turn_id: 't', state: 'HOLD', reason: 'confirming', hold_deadline_ms: 9000 });
  assert.ok(effects.some((e) => e.kind === 'deadline-armed' && e.holdDeadlineMs === 9000));
});

test('playback ack stays honest: server-labelled client-clock time, never acoustic', () => {
  const state = createVoiceState();
  const effects = reduceMessage(state, { type: 'playback_ack_recorded', turn_id: 'T1', ttfa_ms: 410.5, ttfa_status: 'measured_client_clock' });
  const ack = effects.find((e) => e.kind === 'playback-ack');
  assert.equal(ack.ttfaMs, 410.5);
  assert.equal(ack.ttfaStatus, 'measured_client_clock');
});
