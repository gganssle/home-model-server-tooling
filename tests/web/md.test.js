const { md, esc } = require('./md.js');
let pass = 0, fail = 0;
function t(name, got, want) {
  if (got === want) { console.log('  ok   ' + name); pass++; }
  else { console.log('  FAIL ' + name + '\n    got:  ' + JSON.stringify(got) + '\n    want: ' + JSON.stringify(want)); fail++; }
}
function has(name, got, needle) {
  if (got.includes(needle)) { console.log('  ok   ' + name); pass++; }
  else { console.log('  FAIL ' + name + '\n    got: ' + JSON.stringify(got)); fail++; }
}
function hasnt(name, got, needle) {
  if (!got.includes(needle)) { console.log('  ok   ' + name); pass++; }
  else { console.log('  FAIL ' + name + '\n    got: ' + JSON.stringify(got)); fail++; }
}

console.log('\nescaping');
t('escapes html', esc('<img src=x onerror=1>'), '&lt;img src=x onerror=1&gt;');
hasnt('script tags never survive', md('<script>alert(1)</script>'), '<script>alert');
hasnt('img onerror never survives', md('<img src=x onerror="alert(1)">'), '<img src=x');

console.log('\ncode blocks');
has('fenced code becomes pre', md('```\nx = 1\n```'), '<pre><code>x = 1</code></pre>');
has('code inside a fence is escaped', md('```\n<b>hi</b>\n```'), '&lt;b&gt;hi&lt;/b&gt;');
hasnt('markdown inside a fence is left alone', md('```\n**not bold**\n```'), '<strong>');
has('inline code', md('use `foo()` here'), '<code>foo()</code>');

console.log('\nthe placeholder collision bug');
const prose = md('I have 3 apples and 7 pears');
has('bare numbers survive', prose, '3 apples');
hasnt('bare numbers are not spliced', prose, 'undefined');
const mixed = md('First 1 then:\n\n```\ncode\n```\n\nand 0 after');
has('code still renders alongside numbers', mixed, '<pre><code>code</code></pre>');
hasnt('no undefined leaks in', mixed, 'undefined');
has('leading number intact', mixed, 'First 1 then');
has('trailing number intact', mixed, 'and 0 after');

console.log('\ninline formatting');
has('bold', md('**bold**'), '<strong>bold</strong>');
has('italic', md('an *ital* word'), '<em>ital</em>');
has('links', md('[x](https://e.com)'), 'href="https://e.com"');
hasnt('javascript: urls are not linkified', md('[x](javascript:alert(1))'), '<a href');
has('headings', md('## Title'), '<h2>Title</h2>');
has('bullets', md('- one\n- two'), '<li>one</li>');
has('numbered lists', md('1. one\n2. two'), '<li>one</li>');

console.log('\nparagraphs');
has('paragraphs split on blank lines', md('a\n\nb'), '<p>a</p>');
has('single newlines become breaks', md('a\nb'), 'a<br>b');

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
