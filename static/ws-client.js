// WebSocket lifecycle: close-code surfacing, bounded reconnect backoff with a
// fresh `start`, and a manual resume offer. Pure logic — the socket
// itself is created by the caller (app.js) and injected for tests.

export const RECOVERABLE_CLOSE_CODES = Object.freeze([1001, 1006, 1012, 1013, 1014]);
export const FATAL_CLOSE_CODES = Object.freeze([1008, 4401, 4404]);

export const CLOSE_CODE_LABELS = Object.freeze({
  1000: 'Normal closure',
  1001: 'Endpoint going away',
  1005: 'No status code (abrupt drop)',
  1006: 'Abnormal closure (connection lost)',
  1008: 'Server ended the call: the session audio limit was reached — press Reset session to start a new one',
  1009: 'Server ended the call: one audio frame was over the 64 KiB limit — check the microphone setup, then press Resume voice',
  1011: 'Server error',
  1012: 'Service restart',
  1013: 'Temporary overload (try again)',
  1014: 'Bad gateway',
  4401: 'Unauthorized voice session',
  4404: 'Voice session not found',
});

export function describeClose(code, reason) {
  const label = CLOSE_CODE_LABELS[code] || `Close code ${code}`;
  const trimmed = String(reason || '').trim().slice(0, 200);
  return trimmed ? `${label} — ${trimmed}` : label;
}

/** True when an automatic reconnect attempt is worth making. */
export function shouldAutoReconnect(code, { manualStop = false } = {}) {
  if (manualStop) return false;
  if (FATAL_CLOSE_CODES.includes(code)) return false;
  return true;
}

export function createReconnectState({ baseDelayMs = 500, maxDelayMs = 8000, maxAttempts = 5 } = {}) {
  return { attempts: 0, baseDelayMs, maxDelayMs, maxAttempts, pending: false };
}

/**
 * Bounded exponential backoff: min(base * 2^attempts, max), or null when the
 * attempt budget is exhausted (caller then offers manual resume only).
 */
export function nextBackoffDelay(state) {
  if (state.attempts >= state.maxAttempts) return null;
  const delay = Math.min(state.maxDelayMs, state.baseDelayMs * 2 ** state.attempts);
  state.attempts += 1;
  return delay;
}

export function resetReconnect(state) {
  state.attempts = 0;
  state.pending = false;
}

export function createWsLifecycle() {
  return {
    status: 'idle', // idle | connecting | open | closed
    lastClose: null, // { code, reason, atMs }
    resumeOffered: false,
  };
}

export function onWsOpen(lifecycle, reconnect) {
  lifecycle.status = 'open';
  lifecycle.resumeOffered = false;
  resetReconnect(reconnect);
}

export function onWsClose(lifecycle, code, reason, atMs = Date.now()) {
  lifecycle.status = 'closed';
  lifecycle.lastClose = { code, reason: String(reason || ''), atMs };
  const auto = shouldAutoReconnect(code);
  lifecycle.resumeOffered = !auto;
  return { auto, description: describeClose(code, reason) };
}
