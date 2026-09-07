"""The Markdown subset and its escaping.

This used to run under node against the copy of the renderer that lived in the
web UI's <script> block. Datastar renders on the server, so the renderer and
its tests both moved to Python; the cases are the same ones, including the
regression that gave the placeholder its unusual name.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hearth.mdrender import attr, escape, render, safe_href  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def eq(name: str, got: str, want: str) -> None:
    check(name, got == want, f"\n    got:  {got!r}\n    want: {want!r}")


def has(name: str, got: str, needle: str) -> None:
    check(name, needle in got, f"\n    got: {got!r}")


def hasnt(name: str, got: str, needle: str) -> None:
    check(name, needle not in got, f"\n    got: {got!r}")


print("\nescaping")
eq("escapes html", escape("<img src=x onerror=1>"), "&lt;img src=x onerror=1&gt;")
eq("attribute escaping also handles quotes", attr('a "b" & <c>'),
   "a &quot;b&quot; &amp; &lt;c&gt;")
hasnt("script tags never survive", render("<script>alert(1)</script>"), "<script>alert")
hasnt("img onerror never survives", render('<img src=x onerror="alert(1)">'), "<img src=x")

print("\nsource links")
eq("http passes through", safe_href("http://e.com/a"), "http://e.com/a")
eq("https passes through", safe_href("https://e.com/a"), "https://e.com/a")
eq("javascript: is defused", safe_href("javascript:alert(1)"), "#")
eq("data: is defused", safe_href("data:text/html,<script>"), "#")
eq("a scheme-relative url is defused", safe_href("//evil.example"), "#")
eq("nothing at all is defused", safe_href(""), "#")
eq("None is defused", safe_href(None), "#")

print("\ncode blocks")
has("fenced code becomes pre", render("```\nx = 1\n```"), "<pre><code>x = 1</code></pre>")
has("code inside a fence is escaped", render("```\n<b>hi</b>\n```"), "&lt;b&gt;hi&lt;/b&gt;")
hasnt("markdown inside a fence is left alone", render("```\n**not bold**\n```"), "<strong>")
has("inline code", render("use `foo()` here"), "<code>foo()</code>")
has("newlines inside a fence survive", render("```\na\nb\n```"), "<code>a\nb</code>")

print("\nthe placeholder collision bug")
prose = render("I have 3 apples and 7 pears")
has("bare numbers survive", prose, "3 apples")
hasnt("bare numbers are not spliced", prose, "undefined")
mixed = render("First 1 then:\n\n```\ncode\n```\n\nand 0 after")
has("code still renders alongside numbers", mixed, "<pre><code>code</code></pre>")
has("leading number intact", mixed, "First 1 then")
has("trailing number intact", mixed, "and 0 after")
has("an out-of-range placeholder is left alone", render("@@HEARTHCODE9@@"), "@@HEARTHCODE9@@")

print("\ninline formatting")
has("bold", render("**bold**"), "<strong>bold</strong>")
has("italic", render("an *ital* word"), "<em>ital</em>")
has("links", render("[x](https://e.com)"), 'href="https://e.com"')
hasnt("javascript: urls are not linkified", render("[x](javascript:alert(1))"), "<a href")
has("headings", render("## Title"), "<h2>Title</h2>")
has("bullets", render("- one\n- two"), "<li>one</li>")
has("numbered lists", render("1. one\n2. two"), "<li>one</li>")

print("\nparagraphs")
has("paragraphs split on blank lines", render("a\n\nb"), "<p>a</p>")
has("single newlines become breaks", render("a\nb"), "a<br>b")

print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
sys.exit(1 if FAILED else 0)
