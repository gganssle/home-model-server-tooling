"""A small, deliberate subset of Markdown, rendered to HTML.

This is a line-for-line port of the renderer that used to live in the web UI's
`<script>` block. Datastar patches HTML rather than data, so the model's output
has to become HTML on this side of the wire now; the rules did not change, only
the language they are written in.

Deliberate about the subset: a local model emits headings, lists, bold, links
and fenced code, and very little else. Supporting exactly that is a few dozen
lines that can be read in one sitting, where a full CommonMark implementation
would be a dependency and a much larger place for an escaping bug to hide.

Escaping is the load-bearing part. Everything is escaped first and markup is
only ever *added* afterwards, so a model that emits `<script>` gets a visible
`&lt;script&gt;` and never a live tag.
"""
from __future__ import annotations

import re

# Fenced code is lifted out before any other rule runs, so nothing inside a
# fence is ever treated as prose. The placeholder has to be something prose
# will not contain: an earlier version keyed on bare integers, and a reply
# containing "I have 3 apples" spliced a code block into the middle of it.
_FENCE = re.compile(r"```(\w*)\n?([\s\S]*?)```")
_PLACEHOLDER = re.compile(r"@@HEARTHCODE(\d+)@@")

_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_ITALIC = re.compile(r"(^|[^*])\*([^*\n]+)\*")
# Only http(s). This is what keeps `[click](javascript:...)` from linkifying.
_LINK = re.compile(r"\[([^\]]+)\]\((https?:[^)\s]+)\)")
_H3 = re.compile(r"^###\s+(.*)$", re.M)
_H2 = re.compile(r"^##\s+(.*)$", re.M)
_H1 = re.compile(r"^#\s+(.*)$", re.M)
_BULLET = re.compile(r"^\s*[-*]\s+(.*)$", re.M)
_ORDERED = re.compile(r"^\s*\d+\.\s+(.*)$", re.M)
_LIST_RUN = re.compile(r"(<li>[\s\S]*?</li>)(?!\s*<li>)")
_PARA_BREAK = re.compile(r"\n{2,}")
_BLOCK_START = re.compile(r"^\s*<(h\d|ul|pre|li|blockquote)")


def escape(text: object) -> str:
    """Escape text for an HTML text node."""
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def attr(text: object) -> str:
    """Escape text for a double-quoted HTML attribute.

    Separate from `escape` because attributes need the quote handled too, and
    the fragments in `render.py` interpolate titles and filenames into them.
    """
    return escape(text).replace('"', "&quot;")


def safe_href(url: object) -> str:
    """Reduce a URL to something safe to put in an `href`, or to "#".

    Source URLs reach the page from a search provider and from stored history,
    neither of which is trusted. The Markdown renderer already refuses to
    linkify anything but http(s); a link built by hand in `render.py` needs the
    same rule, and a scheme-relative `//host` needs it too - it inherits the
    page's scheme and is a real destination.
    """
    return url if re.match(r"^https?://", str(url or ""), re.I) else "#"


def render(src: object) -> str:
    """Render the supported Markdown subset to HTML."""
    blocks: list[str] = []

    def stash(match: re.Match[str]) -> str:
        code = re.sub(r"\n\Z", "", match.group(2))
        blocks.append(f"<pre><code>{escape(code)}</code></pre>")
        return f"@@HEARTHCODE{len(blocks) - 1}@@"

    s = _FENCE.sub(stash, str(src))
    s = escape(s)
    s = _INLINE_CODE.sub(r"<code>\g<1></code>", s)
    s = _BOLD.sub(r"<strong>\g<1></strong>", s)
    s = _ITALIC.sub(r"\g<1><em>\g<2></em>", s)
    s = _LINK.sub(r'<a href="\g<2>" target="_blank" rel="noopener">\g<1></a>', s)
    s = _H3.sub(r"<h3>\g<1></h3>", s)
    s = _H2.sub(r"<h2>\g<1></h2>", s)
    s = _H1.sub(r"<h2>\g<1></h2>", s)
    s = _BULLET.sub(r"<li>\g<1></li>", s)
    s = _ORDERED.sub(r"<li>\g<1></li>", s)
    s = _LIST_RUN.sub(r"<ul>\g<1></ul>", s)
    s = "".join(
        part if _BLOCK_START.match(part) else "<p>" + part.replace("\n", "<br>") + "</p>"
        for part in _PARA_BREAK.split(s)
    )

    def restore(match: re.Match[str]) -> str:
        index = int(match.group(1))
        # A number the model wrote that happens to look like a placeholder must
        # survive untouched rather than turn into "undefined".
        return blocks[index] if index < len(blocks) else match.group(0)

    return _PLACEHOLDER.sub(restore, s)
