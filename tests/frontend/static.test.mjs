import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

test('static cross-check: DOM ids exist and imports resolve', () => {
  const result = spawnSync('node', [join(root, 'scripts', 'check_static.mjs')], { encoding: 'utf8' });
  assert.equal(result.status, 0, `check_static.mjs failed:\n${result.stdout}\n${result.stderr}`);
  assert.match(result.stdout, /check_static: OK/);
});
