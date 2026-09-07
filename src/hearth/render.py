"""HTML fragments for the web UI.

Under Datastar the browser is not told *what happened*, it is handed *what the
page should now look like*. So every piece of the UI that used to be built by
JavaScript in the browser is built here instead, as a string, and shipped over
SSE by `webui.py`.

Two rules hold throughout:

  * Every fragment carries the `id` it will be matched against. Datastar's
    default patch mode is `outer`, which morphs an element in place by id, so
    a fragment is self-describing - `webui.py` rarely has to name a selector.
  * Anything that came from a model, a user, or a filename goes through
    `escape` or `attr` on the way in. This is the only place in the codebase
    that concatenates untrusted text into markup, which is why it is one file.
"""
from __future__ import annotations

from typing import Any, Iterable

from hearth.commands import WEB_COMMANDS
from hearth.mdrender import attr, escape, safe_href
from hearth.mdrender import render as markdown
from hearth.store import Message, Thread

DOT = " &middot; "


def _js(value: str) -> str:
    """Quote a string for use inside a Datastar expression attribute.

    The expression sits in a double-quoted HTML attribute, so the literal uses
    single quotes and the whole attribute is escaped by the caller.
    """
    escaped = value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    return "'" + escaped + "'"


def image_url(name: str) -> str:
    return "/api/images/" + attr(name)


# ---------------- sidebar ----------------


def thread_row(thread: Thread, active: str) -> str:
    title = thread.title or "Untitled"
    # Written as plain `&&`: `attr()` turns it into `&amp;&amp;`, and the
    # browser hands Datastar back the `&&` it expects. The confirm text is a
    # JS string literal nested inside the attribute.
    remove = (
        f"confirm({_js(f'Delete {title}?')}) && @delete('/ui/threads/{thread.id}')"
    )
    return (
        f'<div class="thread{" active" if thread.id == active else ""}"'
        f' data-on:click="@get(\'/ui/threads/{attr(thread.id)}\')">'
        f'<span class="title">{escape(title)}</span>'
        f'<button class="del" title="Delete" data-on:click__stop="{attr(remove)}">'
        "&times;</button>"
        "</div>"
    )


def thread_list(threads: Iterable[Thread], active: str = "", query: str = "") -> str:
    """The sidebar list, filtered server-side.

    Filtering moved off the client with everything else: the search box is
    bound to a signal, and typing in it re-requests this fragment.
    """
    needle = query.strip().lower()
    rows = [t for t in threads if not needle or needle in (t.title or "").lower()]
    body = "".join(thread_row(t, active) for t in rows)
    if not body:
        body = '<div class="thread-empty">no conversations</div>'
    return f'<div id="threads">{body}</div>'


def status(info: dict[str, Any] | None) -> str:
    """The connection dot and the model/memory line, as one two-element patch."""
    if info is None:
        return (
            '<span class="dot" id="dot"></span>'
            '<div id="sysinfo">server unreachable</div>'
        )
    loaded = [k for k in ("text", "image") if info.get(k, {}).get("loaded")]
    if loaded:
        line = f"{' + '.join(loaded)} loaded - {info['memory']['active_gb']} GB"
    else:
        line = "models idle"
    if info.get("busy_with"):
        line += f" - {info['busy_with']} running"
    return (
        '<span class="dot on" id="dot"></span>'
        f'<div id="sysinfo">{escape(line)}</div>'
    )


# ---------------- composer ----------------


def attachments(names: Iterable[str]) -> str:
    """Thumbnails for what is queued in the composer.

    The queue itself lives in the `atts` signal as a list of filenames the
    server has already stored, which is why removing one is a signal edit in
    the browser followed by a re-render here - and why "use as base" costs no
    upload at all.
    """
    thumbs = []
    for index, name in enumerate(names):
        thumbs.append(
            '<div class="thumb">'
            f'<img src="{image_url(name)}" alt="attachment">'
            '<button class="x" title="Remove"'
            f' data-on:click="$atts = $atts.filter((n, i) =&gt; i !== {index});'
            ' @post(\'/ui/attachments\')">&times;</button>'
            "</div>"
        )
    return f'<div id="attachments">{"".join(thumbs)}</div>'


def slash_menu() -> str:
    """The command list that drops out of the composer when you type "/".

    Datastar has no client-side templating on purpose, so the rows are real
    elements rendered here, each carrying its own `data-show` for whether it
    still matches what has been typed. `$_slashMatch` does the matching once,
    on the page, and every row just asks whether it is in the result.
    """
    rows = []
    for command in WEB_COMMANDS:
        pick = f"$draft = {_js(command.name + ' ')}; $_input.focus()"
        rows.append(
            f'<div class="slash-row"'
            f' data-show="$_slashMatch.includes({_js(command.name)})"'
            f' data-on:click="{attr(pick)}">'
            f'<code>{escape(command.usage)}</code>'
            f'<span>{escape(command.help)}</span>'
            "</div>"
        )
    return f'<div id="slash" data-show="$_slashMatch.length > 0">{"".join(rows)}</div>'


# ---------------- messages ----------------


def _stats(bits: Iterable[object]) -> str:
    """Join stat fragments with a separator, escaping the fragments only.

    The separator is an entity, so it has to be added after escaping rather
    than before - otherwise its own ampersand gets escaped too.
    """
    return DOT.join(escape(b) for b in bits)


def image_stats(meta: dict[str, Any]) -> str:
    bits = [
        f"seed {meta.get('seed')}",
        f"{meta.get('steps')} steps",
        f"{meta.get('width')}x{meta.get('height')}",
        f"{meta.get('elapsed_s')}s",
    ]
    if meta.get("from_image"):
        bits.insert(0, f"from {meta['from_image']} @ {meta.get('image_strength')}")
    return _stats(bits)


def text_stats(meta: dict[str, Any]) -> str:
    return _stats([
        f"{meta.get('tokens_per_second')} tok/s",
        f"{meta.get('generation_tokens')} tokens",
        f"{meta.get('elapsed_s')}s",
    ])


def think_block(text: str, *, open_: bool = False, id_: str = "") -> str:
    ident = f' id="{id_}"' if id_ else ""
    return (
        f'<details class="think"{" open" if open_ else ""}{ident}>'
        "<summary>reasoning</summary>"
        f'<div class="think-body">{escape(text)}</div>'
        "</details>"
    )


def use_as_base(name: str) -> str:
    """Queue a generated image as the base for the next one.

    Three signal writes and a re-render: no download, no re-upload, and the
    image never leaves the server's image directory.
    """
    expr = f"$atts = [{_js(name)}]; $imgmode = true; @post('/ui/attachments')"
    return (
        '<div class="imgactions"><button title="Generate a variation of this image"'
        f' data-on:click="{attr(expr)}">Use as base</button></div>'
    )


def sources_block(sources: Iterable[dict[str, Any]]) -> str:
    """The collapsible list of pages a retrieval read.

    Both the live stream and a reload build it from here, so a source that is
    safe to click during the answer is still safe to click a week later.
    """
    items = []
    for source in sources:
        url = source.get("url") or ""
        items.append(
            f'<li><a href="{attr(safe_href(url))}" target="_blank"'
            f' rel="noopener noreferrer">{escape(source.get("title") or url)}</a>'
            f'<span class="src-url">{escape(url)}</span></li>'
        )
    if not items:
        return ""
    count = f"{len(items)} source" + ("" if len(items) == 1 else "s")
    return (
        f'<details class="sources"><summary>{count}</summary>'
        f'<ol>{"".join(items)}</ol></details>'
    )


def message(m: Message) -> str:
    meta = m.meta or {}
    parts: list[str] = []

    # A retrieval is a note about the conversation rather than a turn in it, so
    # it gets one dim line instead of a speaker label and a rendered body.
    if m.role == "tool":
        return (
            f'<div class="msg tool" id="{attr(m.id)}">'
            f'<div class="body"><div>{escape(m.content)}</div></div></div>'
        )

    if meta.get("sources"):
        parts.append(sources_block(meta["sources"]))

    if meta.get("thinking"):
        parts.append(think_block(meta["thinking"]))

    attached = meta.get("images") or []
    if attached:
        imgs = "".join(
            f'<img src="{image_url(n)}" alt="attached image">' for n in attached
        )
        parts.append(f'<div class="atts">{imgs}</div>')

    if m.image:
        alt = attr(meta.get("prompt") or "generated image")
        parts.append(f'<img class="gen" src="{image_url(m.image)}" alt="{alt}">')
        if meta.get("seed") is not None:
            parts.append(f'<div class="stats">{image_stats(meta)}</div>')
        parts.append(use_as_base(m.image))

    if m.content:
        if m.role == "user":
            parts.append("<div>" + escape(m.content).replace("\n", "<br>") + "</div>")
        else:
            parts.append("<div>" + markdown(m.content) + "</div>")

    if meta.get("tokens_per_second"):
        parts.append(f'<div class="stats">{text_stats(meta)}</div>')
    # A failed turn is still stored, so the error it died with should still be
    # on screen after a reload rather than only while it happened.
    if meta.get("error"):
        parts.append(f'<div class="err">{escape(meta["error"])}</div>')

    role = "you" if m.role == "user" else "model"
    return (
        f'<div class="msg {attr(m.role)}" id="{attr(m.id)}">'
        f'<div class="role">{role}</div>'
        f'<div class="body">{"".join(parts)}</div>'
        "</div>"
    )


EMPTY = (
    '<div class="empty"><h1>hearth</h1>'
    "<p>Ask anything, or tick <b>Image mode</b> to draw.<br>"
    "You can also type <code>/image a red barn at dusk</code>.</p></div>"
)


def message_list(messages: Iterable[Message]) -> str:
    body = "".join(message(m) for m in messages)
    return f'<div id="mlist">{body or EMPTY}</div>'


# ---------------- the streaming placeholder ----------------

# The bubble a generation fills in. Every part that can change during a stream
# is its own id, so the server patches one slot at a time instead of rebuilding
# the bubble on every token. `gen` is reused for each turn and is replaced
# wholesale by the stored message when the turn finishes.
PENDING = (
    '<div class="msg assistant" id="gen">'
    '<div class="role">model</div>'
    '<div class="body">'
    '<div id="gen-sources"></div>'
    '<div id="gen-think"></div>'
    '<div class="notice" id="gen-notice">...</div>'
    '<div id="gen-progress"></div>'
    '<div id="gen-body"></div>'
    "</div></div>"
)


def gen_notice(text: str) -> str:
    if not text:
        return '<div class="notice" id="gen-notice" hidden></div>'
    return f'<div class="notice" id="gen-notice">{escape(text)}</div>'


def gen_progress(step: int, total: int) -> str:
    return f'<div id="gen-progress"><progress max="{total}" value="{step}"></progress></div>'


def gen_think(text: str) -> str:
    return f'<div id="gen-think">{think_block(text, open_=True)}</div>'


def gen_body(content: str) -> str:
    return f'<div id="gen-body">{markdown(content)}</div>'


def gen_sources(sources: Iterable[dict[str, Any]]) -> str:
    return f'<div id="gen-sources">{sources_block(sources)}</div>'


def gen_error(text: str) -> str:
    return f'<div id="gen-body"><div class="err">{escape(text)}</div></div>'
