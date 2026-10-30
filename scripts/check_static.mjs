// Static cross-check: every DOM id used by the browser modules must exist in
// index.html, and every imported name must resolve to an export of its target
// module. Runs under plain node (no DOM, no deps): `node scripts/check_static.mjs`.
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const html = readFileSync(join(root, 'static', 'index.html'), 'utf8');
const staticFiles = readdirSync(join(root, 'static')).filter((f) => f.endsWith('.js') || f.endsWith('.mjs'));

let failures = 0;
const fail = (message) => {
  failures += 1;
  console.error(`check_static: FAIL ${message}`);
};

// 1. DOM ids: collect id="..." from index.html, and $('...') / getElementById("...")
//    plus ui.* key usage from the modules.
const htmlIds = new Set([...html.matchAll(/ id="([^"]+)"/g)].map((m) => m[1]));
const usedIds = new Set();
for (const file of staticFiles) {
  const source = readFileSync(join(root, 'static', file), 'utf8');
  for (const m of source.matchAll(/\$\('([^']+)'\)/g)) usedIds.add(m[1]);
  for (const m of source.matchAll(/\$\("([^"]+)"\)/g)) usedIds.add(m[1]);
  for (const m of source.matchAll(/getElementById\('([^']+)'\)/g)) usedIds.add(m[1]);
  for (const m of source.matchAll(/getElementById\("([^"]+)"\)/g)) usedIds.add(m[1]);
}
for (const id of [...usedIds].sort()) {
  if (!htmlIds.has(id)) fail(`DOM id "${id}" used by static/*.js but missing from index.html`);
}

// 2. Import/export resolution: parse `import { a, b } from './x.js'` in every
//    static module and check each name against that file's exports
//    (`export (const|function|class|async function) name`, `export { ... }`,
//    and `export * from` re-exports).
function parseExports(source) {
  const names = new Set();
  for (const m of source.matchAll(/export\s+(?:async\s+function|function|const|let|var|class)\s+([A-Za-z_$][\w$]*)/g)) {
    names.add(m[1]);
  }
  for (const m of source.matchAll(/export\s*\{([^}]+)\}/g)) {
    for (const part of m[1].split(',')) {
      const alias = part.trim().split(/\s+as\s+/);
      names.add((alias[1] || alias[0]).trim());
    }
  }
  return names;
}

function parseStarReexports(source) {
  return [...source.matchAll(/export\s+\*\s+from\s+['"]([^'"]+)['"]/g)].map((m) => m[1]);
}

async function resolveExports(file, seen = new Set()) {
  if (seen.has(file)) return new Set();
  seen.add(file);
  const source = readFileSync(join(root, 'static', file), 'utf8');
  const names = parseExports(source);
  for (const target of parseStarReexports(file, source)) {
    void target;
  }
  for (const m of source.matchAll(/export\s+\*\s+from\s+['"]([^'"]+)['"]/g)) {
    const target = m[1].startsWith('./') ? m[1].slice(2) : m[1];
    for (const name of await resolveExports(target, seen)) names.add(name);
  }
  return names;
}

for (const file of staticFiles) {
  const source = readFileSync(join(root, 'static', file), 'utf8');
  for (const m of source.matchAll(/import\s*\{([^}]+)\}\s*from\s*['"]([^'"]+)['"]/g)) {
    if (!m[2].startsWith('.')) continue; // no bare/CDN imports allowed anyway (checked below)
    const target = m[2].startsWith('./') ? m[2].slice(2) : m[2];
    let available;
    try {
      available = await resolveExports(target);
    } catch {
      fail(`${file} imports from missing file ${m[2]}`);
      continue;
    }
    for (const part of m[1].split(',')) {
      const local = part.trim().split(/\s+as\s+/);
      const imported = local[0].trim();
      if (imported && !available.has(imported)) fail(`${file} imports "${imported}" from ${m[2]}, which does not export it`);
    }
  }
}

// 3. Constraints: no innerHTML with dynamic data, no new runtime deps/CDNs,
//    no bare imports, no secrets in the page.
for (const file of staticFiles) {
  const source = readFileSync(join(root, 'static', file), 'utf8');
  if (/\.innerHTML\s*=/.test(source)) fail(`${file} assigns innerHTML (use textContent/createElement)`);
  if (/from\s+['"]https?:\/\//.test(source)) fail(`${file} imports from a CDN URL`);
  if (/from\s+['"]([^.'"][^'"]*)['"]/.test(source)) fail(`${file} has a bare package import (no runtime deps allowed)`);
  if (/(api[_-]?key|secret|token)\s*[:=]\s*['"][A-Za-z0-9]{8,}/i.test(source)) {
    fail(`${file} may embed a secret value`);
  }
}

// 4. Every static module reachable from index.html's entry must parse —
//    exercised by importing the module graph headlessly.
try {
  await import('../static/app.js');
  console.log('check_static: app.js module graph imports OK (browser globals expected to fail at runtime only)');
} catch (error) {
  // app.js touches `document` at module top level ($() lookups), so importing
  // it in node must fail at the DOM access — anything else is a real break.
  const message = error && error.message ? error.message : String(error);
  if (!/document is not defined/.test(message)) fail(`app.js import failed unexpectedly: ${message}`);
}

if (failures) {
  console.error(`check_static: ${failures} failure(s)`);
  process.exit(1);
}
console.log(`check_static: OK — ${usedIds.size} DOM ids resolve, imports resolve, constraints hold`);
