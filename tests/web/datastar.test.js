// Check the UI's `data-*` attributes against the Datastar build that ships
// with it.
//
// Every gesture in the page is a string in an attribute: nothing in Python
// type-checks them, and a typo does not throw, it just silently does nothing.
// So this pulls the plugin names out of the vendored bundle, confirms the page
// only uses attributes that bundle implements, and compiles each expression
// after applying Datastar's own two rewrites.
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..', '..');
const raw = fs.readFileSync(path.join(root, 'src/hearth/web/index.html'), 'utf8');
// Comments in this file talk *about* data-attributes; only real ones count.
const html = raw.replace(/<!--[\s\S]*?-->/g, '');

// The browser decodes entities before Datastar ever sees an expression, so
// `&&` is written `&amp;&amp;` in the source and has to be decoded here too.
const decode = (s) => s
  .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"')
  .replace(/&#39;/g, "'").replace(/&amp;/g, '&');
const bundle = fs.readFileSync(path.join(root, 'src/hearth/web/datastar.js'), 'utf8');

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { console.log('  ok   ' + name); pass++; }
  else { console.log('  FAIL ' + name + (detail ? '\n    ' + detail : '')); fail++; }
}

// The bundle registers each attribute plugin as `{name:"show"...}`.
const plugins = new Set(
  [...bundle.matchAll(/\{name:"([a-z0-9-]+)"/g)].map((m) => m[1])
    .filter((n) => !n.startsWith('datastar-')));
check('bundle exposes its plugin list', plugins.size > 10, [...plugins].join(' '));
check('bundle is a pinned version', /^\/\/ Datastar v1\.\d+\.\d+/.test(bundle));

// Attributes that hold a Datastar expression rather than a plain name.
const EXPRESSIONS = new Set(['on', 'show', 'text', 'attr', 'class', 'style',
  'signals', 'computed', 'effect', 'init', 'on-interval', 'on-intersect',
  'on-signal-patch']);

const attrs = [...html.matchAll(/\sdata-([a-z-]+)((?::|__)[^=\s>]*)?(?:="([^"]*)")?/g)];
check('the page uses Datastar attributes', attrs.length > 10, `${attrs.length} found`);

const seen = new Set();
for (const [, name, , value] of attrs) {
  seen.add(name);
  if (!plugins.has(name)) { check(`data-${name} is a real plugin`, false); continue; }
  if (!EXPRESSIONS.has(name) || value === undefined) continue;

  // Datastar's own rewrites: `@action(` becomes a call into its action table,
  // and `$signal` becomes a lookup. It then wraps the trailing statement in a
  // `return`, which is what lets `data-signals` be a bare object literal.
  const js = decode(value)
    .replace(/@([A-Za-z_$][\w$]*)\(/g, '__action("$1",evt,')
    .replace(/\$([A-Za-z_][\w]*)/g, '__signal_$1');
  try {
    new Function('el', '$', '__action', 'evt', 'return (' + js + ');');
    check(`data-${name} expression compiles`, true);
  } catch (e) {
    check(`data-${name} expression compiles`, false, `${e.message}\n    ${value}`);
  }
}

// Attributes that only exist in the page as bare names still have to be real.
for (const name of ['bind', 'ref', 'indicator']) {
  check(`data-${name} is used and supported`, seen.has(name) && plugins.has(name));
}

// The frontend is meant to work with the network off.
check('no external script or style is loaded',
      !/<(script|link)[^>]+(src|href)="https?:/.test(raw));
check('the runtime is served from this app', raw.includes('src="/datastar.js"'));

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
