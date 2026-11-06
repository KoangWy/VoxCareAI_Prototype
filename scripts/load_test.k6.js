// NFR-06 load test (k6): concurrent sessions at 10, 25 and 50 streams (guide §19.1).
//
//   k6 run -e BASE=http://127.0.0.1:8000 -e STREAMS=50 scripts/load_test.k6.js
//
// Each virtual user opens a session (POST /session), sends typed turns through the
// same session boundary the voice path uses, and opens the voice WebSocket with its
// token. Typed turns measure the dialogue/tool path under load; TTFA p95 ≤ 1200 ms
// under load needs real audio streams and the browser clock (scripts/nfr_latency_report.py).
// Run with VOXCARE_CONVERSATION_PROVIDER=keyword to load the stack without spending model credits.
import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

const BASE = __ENV.BASE || 'http://127.0.0.1:8000';
const STREAMS = Number(__ENV.STREAMS || 10);
const turnLatency = new Trend('turn_latency_ms', true);
const handshake = new Trend('wss_handshake_ms', true);

export const options = {
  scenarios: {
    calls: { executor: 'constant-vus', vus: STREAMS, duration: __ENV.DURATION || '2m' },
  },
  thresholds: { http_req_failed: ['rate<0.01'] },
};

const QUESTIONS = [
  'Where is order ORD-1002?',
  'What is the return window for a defective TV?',
  'Compare the Samsung and the LG 55-inch TV, and check if voucher VOXCOM_500K applies.',
  'What did I buy before?',
];

export default function () {
  const created = http.post(`${BASE}/session`, JSON.stringify({ verified: true }), { headers: { 'Content-Type': 'application/json' } });
  check(created, { 'session created': (r) => r.status === 200 });
  const session = created.json();
  const headers = { 'Content-Type': 'application/json', 'VoxCare-Session-Token': session.token };
  for (const text of QUESTIONS) {
    const started = Date.now();
    const response = http.post(`${BASE}/api/sessions/${session.session_id}/turns`, JSON.stringify({ text }), { headers });
    turnLatency.add(Date.now() - started);
    check(response, { 'turn answered': (r) => r.status === 200 && r.json('assistant_text') });
    sleep(1);
  }
  const wsUrl = `${BASE.replace('http', 'ws')}/ws/voice?token=${encodeURIComponent(session.token)}`;
  const opened = Date.now();
  ws.connect(wsUrl, {}, (socket) => {
    socket.on('message', () => { handshake.add(Date.now() - opened); socket.close(); });
    socket.setTimeout(() => socket.close(), 2000);
  });
}
