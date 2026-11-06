// Session activity log (operator view). The entries come from the server
// (GET /api/sessions/{id}/activity): how each turn was handled, every tool call
// with the checks it passed or failed, the audit rows it wrote, state changes
// and the reply. Values arrive PII-masked. This module only turns them into DOM
// nodes (textContent only); it never infers anything the server did not record.

const STEP_LABELS = Object.freeze({
  input: 'Input',
  route: 'Handled by',
  note: 'Model note',
  tool: 'Tool call',
  write: 'Change',
  audit: 'Audit',
  state: 'State',
  reply: 'Reply',
});

const ENTRY_TITLES = Object.freeze({ session: 'Session', support: 'Support button', event: 'Event' });

const REFUSED = /OUT_OF_SCOPE|UNAUTHENTICATED|THROTTLED|REFUSED|FAILED|UNAVAILABLE|INSUFFICIENT_EVIDENCE|LIMIT_REACHED/;
const PENDING = /CONFIRMATION|NEEDS_|RECORDED|DIAGNOSIS|CLARIFICATION|OPD_FALLBACK|WITHDRAWN/;

export function toneFor(entry) {
  if (!entry || entry.kind !== 'turn') return entry?.kind === 'support' ? 'pending' : 'info';
  const outcome = String(entry.outcome || '');
  if (outcome === 'STALE_DISCARDED') return 'muted';
  if (REFUSED.test(outcome)) return 'refused';
  if (PENDING.test(outcome)) return 'pending';
  return 'ok';
}

export function checkMark(result) {
  if (result === 'pass') return '✓';
  if (result === 'fail') return '✗';
  return '•';
}

export function stepView(step) {
  const auditId = typeof step?.audit_id === 'string' ? step.audit_id : '';
  return {
    kind: step?.kind || 'note',
    label: STEP_LABELS[step?.kind] || 'Note',
    text: String(step?.text || ''),
    checks: (Array.isArray(step?.checks) ? step.checks : []).map((check) => ({
      label: String(check.label || check.check || ''),
      result: check.result === 'pass' || check.result === 'fail' ? check.result : 'note',
      mark: checkMark(check.result),
      detail: check.detail ? String(check.detail) : '',
    })),
    result: step?.kind === 'tool' && step.result ? String(step.result) : '',
    auditId,
    shortAuditId: auditId ? auditId.slice(0, 8) : '',
  };
}

export function entryView(entry, index) {
  const isTurn = entry?.kind === 'turn';
  return {
    key: `${index}:${entry?.kind}:${isTurn ? entry.turn : ''}`,
    title: isTurn ? `Turn ${entry.turn} · ${entry.source === 'voice' ? 'voice' : 'typed'}` : ENTRY_TITLES[entry?.kind] || 'Event',
    caller: isTurn ? String(entry.caller_text || '') : '',
    summary: String(entry?.summary || ''),
    tone: toneFor(entry),
    steps: (Array.isArray(entry?.steps) ? entry.steps : []).map(stepView),
  };
}

/** Newest first; the newest turn opens unless the operator toggled it. */
export function orderedViews(entries) {
  return (Array.isArray(entries) ? entries : []).map(entryView).reverse();
}

export function createActivityView() {
  return { toggled: new Map() };
}

function el(doc, tag, className, text) {
  const node = doc.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null && text !== '') node.textContent = text;
  return node;
}

function stepNode(doc, step) {
  const item = el(doc, 'li', `activity-step step-${step.kind}`);
  item.append(el(doc, 'span', 'step-kind', step.label));
  const body = el(doc, 'div', 'step-body');
  body.append(el(doc, step.kind === 'tool' ? 'code' : 'p', 'step-text', step.text));
  if (step.checks.length) {
    const checks = el(doc, 'ul', 'step-checks');
    for (const check of step.checks) {
      const chip = el(doc, 'li', `check check-${check.result}`, `${check.mark} ${check.label}`);
      if (check.detail) chip.title = check.detail;
      checks.append(chip);
    }
    body.append(checks);
  }
  if (step.result) body.append(el(doc, 'p', 'step-result', `→ ${step.result}`));
  if (step.auditId) {
    const id = el(doc, 'code', 'audit-id', step.shortAuditId);
    id.title = `audit_id ${step.auditId}`;
    body.append(id);
  }
  item.append(body);
  return item;
}

export function renderActivity(list, entries, view, doc = globalThis.document) {
  const views = orderedViews(entries);
  if (!views.length) {
    list.replaceChildren(el(doc, 'li', 'activity-empty', 'No activity yet. Ask a question to see how the turn is handled.'));
    return;
  }
  const latestTurn = views.find((item) => item.title.startsWith('Turn'))?.key;
  const nodes = views.map((item) => {
    const entryNode = el(doc, 'li', `activity-entry tone-${item.tone}`);
    if (!item.steps.length) {
      const line = el(doc, 'div', 'activity-line');
      line.append(el(doc, 'span', 'activity-title', item.title), el(doc, 'span', 'activity-summary', item.summary));
      entryNode.append(line);
      return entryNode;
    }
    const details = el(doc, 'details', 'activity-details');
    details.open = view?.toggled?.has(item.key) ? view.toggled.get(item.key) : item.key === latestTurn;
    details.addEventListener('toggle', () => view?.toggled?.set(item.key, details.open));
    const summary = el(doc, 'summary', 'activity-head');
    summary.append(el(doc, 'span', 'activity-title', item.title));
    if (item.caller) summary.append(el(doc, 'span', 'activity-caller', `“${item.caller}”`));
    summary.append(el(doc, 'span', 'activity-summary', item.summary));
    const steps = el(doc, 'ol', 'activity-steps');
    steps.append(...item.steps.map((step) => stepNode(doc, step)));
    details.append(summary, steps);
    entryNode.append(details);
    return entryNode;
  });
  list.replaceChildren(...nodes);
}
