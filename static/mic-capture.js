// Mic start/stop state machine with an in-flight token.
//
// Rapid mic toggles race shared globals (socket, AudioContext, MediaStream):
// the loser overwrites the winner's handles, leaking the mic stream and the
// context. Every transition takes a token; completions that no longer hold
// the current token must release — never adopt — the resources they opened.
// Pure logic (no navigator/DOM) so the race rules are unit-testable.

export function createMicState() {
  return {
    status: 'idle', // idle | starting | recording | stopping
    token: 0,
    transitioning: false,
  };
}

export function micTransitionAllowed(state) {
  return !state.transitioning;
}

/** Begin a start transition. Returns the token, or null when busy. */
export function beginStart(state) {
  if (state.transitioning) return null;
  state.transitioning = true;
  state.status = 'starting';
  state.token += 1;
  return state.token;
}

/** Begin a stop transition. Returns the token, or null when busy. */
export function beginStop(state) {
  if (state.transitioning) return null;
  state.transitioning = true;
  state.status = 'stopping';
  state.token += 1;
  return state.token;
}

export function isTokenCurrent(state, token) {
  return token === state.token;
}

export function finishStart(state, token) {
  if (token !== state.token) return false;
  state.transitioning = false;
  state.status = 'recording';
  return true;
}

export function finishStop(state, token) {
  if (token !== state.token) return false;
  state.transitioning = false;
  state.status = 'idle';
  return true;
}

/** Abort a transition that failed before adopting resources. */
export function abortTransition(state, token) {
  if (token !== state.token) return false;
  state.transitioning = false;
  state.status = 'idle';
  return true;
}

export function isRecordingState(state) {
  return state.status === 'recording';
}
