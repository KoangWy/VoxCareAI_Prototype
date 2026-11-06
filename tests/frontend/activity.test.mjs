import test from 'node:test';
import assert from 'node:assert/strict';
import { checkMark, createActivityView, entryView, orderedViews, renderActivity, stepView, toneFor } from '../../static/activity-log.js';

const TURN = {
  kind: 'turn', turn: 2, source: 'typed', caller_text: 'Please check order ORD-2002.', outcome: 'OUT_OF_SCOPE',
  summary: 'Refused: not on the verified account',
  steps: [
    { kind: 'route', text: 'No fixed rule matched: the conversation model proposed tool calls (1 call)', code: 'MODEL_PLAN' },
    {
      kind: 'tool', text: "lookup_order_details(order_code='ORD-2002')", name: 'lookup_order_details',
      checks: [
        { check: 'allowlist', result: 'pass', label: 'allowlist' },
        { check: 'customer_scope', result: 'fail', label: 'customer scope', detail: 'belongs to another customer; no details returned' },
      ],
      result: 'refused: belongs to another customer; no details returned', status: 'OUT_OF_SCOPE',
    },
    { kind: 'audit', text: 'AUTH_SCOPE_VIOLATION · Denied/DENIED · lookup_order_details', audit_id: 'cc5c50b2-fafd-4a5d-8bcf-c1933c33b467' },
    { kind: 'reply', text: 'For data privacy and security, I can only share details for orders on your verified account.' },
  ],
};

test('tones follow the server outcome, never the reply wording', () => {
  assert.equal(toneFor(TURN), 'refused');
  assert.equal(toneFor({ kind: 'turn', outcome: 'ANSWERED' }), 'ok');
  assert.equal(toneFor({ kind: 'turn', outcome: 'WRITE_REQUEST_RECORDED' }), 'pending');
  assert.equal(toneFor({ kind: 'turn', outcome: 'STALE_DISCARDED' }), 'muted');
  assert.equal(toneFor({ kind: 'session' }), 'info');
  assert.equal(toneFor({ kind: 'support' }), 'pending');
});

test('a tool step keeps every check, its result and the audit id', () => {
  const tool = stepView(TURN.steps[1]);
  assert.equal(tool.label, 'Tool call');
  assert.deepEqual(tool.checks.map((check) => `${check.mark} ${check.label}`), ['✓ allowlist', '✗ customer scope']);
  assert.equal(tool.checks[1].detail, 'belongs to another customer; no details returned');
  assert.equal(tool.result, 'refused: belongs to another customer; no details returned');
  const audit = stepView(TURN.steps[2]);
  assert.equal(audit.shortAuditId, 'cc5c50b2');
  assert.equal(audit.auditId, 'cc5c50b2-fafd-4a5d-8bcf-c1933c33b467');
  assert.equal(checkMark('note'), '•');
});

test('entries render newest first with the caller words and summary', () => {
  const views = orderedViews([{ kind: 'session', summary: 'Session started', steps: [] }, TURN]);
  assert.equal(views[0].title, 'Turn 2 · typed');
  assert.equal(views[0].caller, 'Please check order ORD-2002.');
  assert.equal(views[1].title, 'Session');
  assert.equal(entryView({ kind: 'turn', turn: 1, source: 'voice', steps: [] }, 0).title, 'Turn 1 · voice');
});

function fakeDocument() {
  const make = (tag) => {
    const node = {
      tag, className: '', textContent: '', title: '', open: false, children: [], listeners: {},
      append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = [...items]; },
      addEventListener(type, handler) { this.listeners[type] = handler; },
    };
    return node;
  };
  return { createElement: make };
}

function texts(node, out = []) {
  if (node.textContent) out.push(node.textContent);
  for (const child of node.children) texts(child, out);
  return out;
}

test('render opens the newest turn, keeps operator toggles, and uses text nodes only', () => {
  const doc = fakeDocument();
  const list = doc.createElement('ol');
  const view = createActivityView();
  const older = { ...TURN, turn: 1, outcome: 'ANSWERED', summary: 'Answered' };
  renderActivity(list, [older, TURN], view, doc);
  const [newest, previous] = list.children.map((item) => item.children[0]);
  assert.equal(newest.open, true);
  assert.equal(previous.open, false);
  const words = texts(list);
  assert.ok(words.includes('“Please check order ORD-2002.”'));
  assert.ok(words.includes('✗ customer scope'));
  assert.ok(words.includes('cc5c50b2'));

  previous.open = true;
  previous.listeners.toggle();
  renderActivity(list, [older, TURN], view, doc);
  assert.equal(list.children[1].children[0].open, true);

  renderActivity(list, [], view, doc);
  assert.match(list.children[0].textContent, /No activity yet/);
});
