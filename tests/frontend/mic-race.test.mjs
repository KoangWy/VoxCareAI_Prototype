import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createMicState, micTransitionAllowed, beginStart, beginStop,
  isTokenCurrent, finishStart, finishStop, abortTransition,
} from '../../static/mic-capture.js';

// Simulates the app.js start/stop flow against mocked async media: the loser
// of a rapid toggle must release — never adopt — its resources.

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

test('mic start/stop race: loser releases its stream, winner keeps its own', async () => {
  const state = createMicState();
  const startToken = beginStart(state);
  assert.ok(startToken !== null);
  assert.equal(micTransitionAllowed(state), false);

  // Rapid toggle while starting: no second transition may begin.
  assert.equal(beginStop(state), null);
  assert.equal(beginStart(state), null);

  // The in-flight start is superseded (reset path bumps the token).
  state.token += 1;
  state.transitioning = false;
  state.status = 'idle';
  assert.equal(isTokenCurrent(state, startToken), false);

  // A fresh stop transition now proceeds with its own token.
  const stopToken = beginStop(state);
  assert.ok(stopToken !== null && stopToken !== startToken);
  assert.equal(finishStop(state, stopToken), true);
  assert.equal(state.status, 'idle');
});

test('stale finish calls cannot overwrite the current transition', () => {
  const state = createMicState();
  const first = beginStart(state);
  assert.equal(finishStart(state, first + 99), false);
  assert.equal(state.status, 'starting');
  assert.equal(finishStart(state, first), true);
  assert.equal(state.status, 'recording');
});

test('abort releases a failed transition back to idle', () => {
  const state = createMicState();
  const token = beginStart(state);
  assert.equal(abortTransition(state, token), true);
  assert.equal(state.status, 'idle');
  assert.equal(micTransitionAllowed(state), true);
  assert.equal(abortTransition(state, token - 1), false);
});

test('stop-while-starting serialises: stop waits for the transition flag', () => {
  const state = createMicState();
  beginStart(state);
  // UI guard: the mic button must be treated as busy during transition.
  assert.equal(micTransitionAllowed(state), false);
});

test('mocked rapid toggle: second start attempt during getUserMedia is refused', async () => {
  const state = createMicState();
  const gum = deferred();
  const token = beginStart(state);
  assert.ok(token);
  const loser = beginStart(state);
  assert.equal(loser, null, 'second toggle during in-flight getUserMedia must be refused');
  gum.resolve('stream-1');
  const stream = await gum.promise;
  assert.equal(stream, 'stream-1');
  assert.equal(finishStart(state, token), true);
});
