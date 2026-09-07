"""The browser's half of the server, spoken in Datastar rather than JSON.

`server.py` exposes a JSON API because the CLI, the REPL and the OpenAI shim
all want one. The browser does not: under Datastar the page holds a handful of
signals and no application logic, and every visible change arrives as an HTML
fragment the server rendered. So these routes return markup, or a stream of
`datastar-patch-*` frames, and share the store, the model manager and the turn
pipeline with the JSON side through `Backend`.

Three shapes of response show up here, and the client picks its behaviour from
the Content-Type:

  * `text/html`   - one or more fragments, morphed into place by their `id`.
                    Used where nothing but the DOM changes (`/ui/threads`).
  * `text/event-stream` - patches, possibly signals too, possibly for a long
                    time. Used for anything that also moves a signal, and for
                    generation, where frames arrive for as long as the model
                    is talking.
  * 204          - nothing to say.

The signal bag travels with every request: in the query string for GET and
DELETE, in the JSON body otherwise. That is why the request models here are
lenient where `server.py`'s are strict - the client sends the whole bag by
design, and a model that forbade extras would reject every request the moment
a new signal was added to the page.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import iterate_in_threadpool

from hearth import render
from hearth.commands import OFF_WORDS, ON_WORDS, WEB_COMMANDS
from hearth.datastar import patch_elements, patch_signals, read_signals

WEB_DIR = Path(__file__).parent / "web"

# How many images the composer will hold at once. The vision tower re-encodes
# every one of them on every turn, so this is a comfort limit, not a cliff.
MAX_ATTACHMENTS = 8

# Repainting the assistant bubble on every token would mean re-rendering the
# whole Markdown document per token and shipping it down the wire. Coalescing
# to ~15 frames a second looks identical and costs a fraction of that.
FLUSH_INTERVAL = 0.06


@dataclass
class Backend:
    """What the UI routes borrow from `server.create_app`."""

    cfg: Any
    store: Any
    manager: Any
    websearch: Any
    resolve: Callable[[str], Any]
    prepare_turn: Callable[..., Any]
    materialize_images: Callable[[list[str]], list[str]]


class UploadedFile(BaseModel):
    """One entry of the array Datastar's `data-bind` builds from a file input."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    mime: str = ""
    contents: str = ""


class Signals(BaseModel):
    """The page's signals, as they arrive in a request body.

    Only the fields a route reads are declared; the rest of the bag is ignored.
    """

    model_config = ConfigDict(extra="ignore")

    thread: str = ""
    draft: str = ""
    search: str = ""
    imgmode: bool = False
    think: bool = False
    web: bool = False
    strength: float = Field(default=0.6, ge=0.0, le=1.0)
    atts: list[str] = Field(default_factory=list)
    files: list[UploadedFile] = Field(default_factory=list)


def _stream(frames: Iterator[str] | list[str]) -> StreamingResponse:
    return StreamingResponse(
        iterate_in_threadpool(iter(frames)), media_type="text/event-stream"
    )


def register(app: FastAPI, backend: Backend) -> None:
    cfg, store, manager = backend.cfg, backend.store, backend.manager
    websearch = backend.websearch

    def _known(names: list[str]) -> list[str]:
        """Keep only attachment names the image directory actually holds.

        `atts` is a signal, so it is whatever the browser last said it was.
        The JSON API deliberately accepts filesystem paths - the CLI attaches
        images by path - but nothing arriving from a page should be able to
        name a file outside the image directory, so this drops the directory
        part and requires the result to already exist.
        """
        out: list[str] = []
        for name in names:
            if not isinstance(name, str):
                continue
            leaf = Path(name).name
            if leaf and leaf not in out and (cfg.image_dir / leaf).exists():
                out.append(leaf)
        return out[:MAX_ATTACHMENTS]

    def _threads_html(active: str, query: str) -> str:
        return render.thread_list(store.list_threads(), active, query)

    def _message(thread_id: str, message_id: str | None) -> Any:
        for m in reversed(store.get_messages(thread_id)):
            if m.id == message_id:
                return m
        return None

    # ---------------- the page itself ----------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "index.html").read_text()

    @app.get("/datastar.js")
    def datastar_runtime() -> FileResponse:
        """Serve the runtime from disk rather than a CDN.

        hearth is meant to keep working on a laptop with the wifi off, and a
        UI that silently stops reacting without a network is not that.
        """
        return FileResponse(WEB_DIR / "datastar.js", media_type="text/javascript")

    # ---------------- sidebar ----------------

    @app.get("/ui/threads", response_class=HTMLResponse)
    def ui_threads(request: Request) -> str:
        sig = read_signals(request.query_params.get("datastar"))
        return _threads_html(str(sig.get("thread") or ""), str(sig.get("search") or ""))

    @app.get("/ui/commands")
    def ui_commands() -> StreamingResponse:
        """The composer's command menu, and the names it matches against.

        Sent once on load. The names go into a signal so the page can do its
        own prefix matching without a round trip per keystroke; the underscore
        keeps that list out of every subsequent request body.
        """
        return _stream([
            patch_signals({"_slash": [c.name for c in WEB_COMMANDS]}),
            patch_elements(render.slash_menu()),
        ])

    @app.get("/ui/status")
    def ui_status() -> StreamingResponse:
        """The connection dot, the memory line, and whether search is on offer.

        Availability travels as signals rather than markup because the Web
        control wraps a checkbox: morphing that element every four seconds
        would fight the box for its own checked state. A signal leaves the
        element alone and only moves what changed.
        """
        enabled = websearch.enabled
        return _stream([
            patch_elements(render.status(manager.status())),
            patch_signals({
                "searchOn": enabled,
                "searchTip": (
                    f"Search the web before answering ({cfg.search.provider})"
                    if enabled else websearch.unavailable_reason
                ),
            }),
        ])

    @app.get("/ui/threads/{ref}")
    def ui_open_thread(ref: str, request: Request) -> StreamingResponse:
        sig = read_signals(request.query_params.get("datastar"))
        thread = backend.resolve(ref)
        return _stream([
            patch_signals({"thread": thread.id}),
            patch_elements(render.message_list(store.get_messages(thread.id))),
            patch_elements(_threads_html(thread.id, str(sig.get("search") or ""))),
        ])

    @app.delete("/ui/threads/{ref}")
    def ui_delete_thread(ref: str, request: Request) -> StreamingResponse:
        sig = read_signals(request.query_params.get("datastar"))
        thread = backend.resolve(ref)
        store.delete_thread(thread.id)
        active = str(sig.get("thread") or "")
        frames = []
        if active == thread.id:
            active = ""
            frames.append(patch_signals({"thread": ""}))
            frames.append(patch_elements(render.message_list([])))
        frames.append(patch_elements(_threads_html(active, str(sig.get("search") or ""))))
        return _stream(frames)

    @app.post("/ui/new")
    def ui_new(sig: Signals) -> StreamingResponse:
        """Clear the pane. No thread is created until something is sent."""
        return _stream([
            patch_signals({"thread": "", "draft": "", "atts": []}),
            patch_elements(render.message_list([])),
            patch_elements(render.attachments([])),
            patch_elements(_threads_html("", sig.search)),
        ])

    # ---------------- composer ----------------

    @app.post("/ui/attachments")
    def ui_attachments(sig: Signals) -> StreamingResponse:
        """Reconcile the composer's queue with what the page says it should be.

        One route serves three gestures, because all three are the same thing:
        picking files appends to `atts`, removing a thumbnail filters `atts`,
        and "use as base" assigns `atts` outright. The browser edits the
        signal and asks for the matching markup back.

        New files arrive base64-encoded in `files` and are stored here, so
        `atts` only ever holds names the server already has - which is what
        makes re-using a generated image free.
        """
        queued = _known(sig.atts)
        if sig.files:
            uris = [
                f"data:{f.mime or 'image/png'};base64,{f.contents}"
                for f in sig.files
                if f.contents
            ]
            room = MAX_ATTACHMENTS - len(queued)
            queued += backend.materialize_images(uris[:room])
        return _stream([
            patch_signals({"atts": queued, "files": []}),
            patch_elements(render.attachments(queued)),
        ])

    # ---------------- generation ----------------

    @app.post("/ui/cancel")
    def ui_cancel() -> StreamingResponse:
        manager.cancel_current()
        return _stream([patch_elements(render.gen_notice("cancelling..."))])

    def _turn_frames(thread: Any, turn: Any, search: str) -> Iterator[str]:
        """Translate one turn's events into patches for the page.

        The bubble is appended once as a set of empty, named slots
        (`render.PENDING`) and then filled in place: notice, reasoning,
        progress bar and body each get patched on their own, so a token only
        repaints the paragraph it landed in. When the turn ends the whole
        placeholder is replaced by the stored message, which is the same
        markup a page reload would produce - so there is never a "live" and a
        "saved" rendering to keep in step.

        A turn that searched is the exception. Retrieval stores a `tool`
        message of its own, in between the question and the answer, and the
        placeholder was appended before that message existed. Rather than
        guess where it belongs, such a turn ends by re-rendering the whole
        transcript - which is the same markup a reload gives, for the same
        reason.
        """
        yield patch_elements(render.message(turn.user), selector="#mlist", mode="append")
        yield patch_signals({"thread": thread.id, "draft": "", "atts": []})
        yield patch_elements(render.attachments([]))
        yield patch_elements(render.PENDING, selector="#mlist", mode="append")

        content = ""
        thinking = ""
        sources: list[dict[str, Any]] = []
        searched = False
        noticed = True
        last_flush = 0.0
        finished = False

        def flush() -> Iterator[str]:
            if thinking:
                yield patch_elements(render.gen_think(thinking))
            if content:
                yield patch_elements(render.gen_body(content))

        for event in turn.events():
            kind = event.get("type")

            if kind == "search":
                searched = True
                phase = event.get("phase")
                if phase == "querying":
                    yield patch_elements(
                        render.gen_notice(f"searching the web for \u201c{event.get('query')}\u201d...")
                    )
                elif phase == "fetching":
                    count = len(event.get("urls") or [])
                    yield patch_elements(render.gen_notice(f"reading {count} page(s)..."))
                elif phase == "ready":
                    # The strip goes in as soon as the sources are known, so it
                    # is on screen while the answer citing it is still being
                    # written. A later round appends to the same list.
                    sources.extend(event.get("sources") or [])
                    if sources:
                        yield patch_elements(render.gen_sources(sources))
                    yield patch_elements(render.gen_notice("thinking..."))
                elif phase == "error":
                    yield patch_elements(
                        render.gen_notice(f"search failed: {event.get('error')}")
                    )
            elif kind == "queued":
                yield patch_elements(render.gen_notice("waiting for the model..."))
            elif kind == "status":
                yield patch_elements(render.gen_notice(f"{event.get('text')}..."))
            elif kind == "progress":
                step, total = event.get("step", 0), event.get("total", 1)
                yield patch_elements(render.gen_notice(f"generating - step {step}/{total}"))
                yield patch_elements(render.gen_progress(step, total))
            elif kind == "token":
                if noticed:
                    noticed = False
                    yield patch_elements(render.gen_notice(""))
                if event.get("channel") == "thinking":
                    thinking += event.get("text", "")
                else:
                    content += event.get("text", "")
                now = time.monotonic()
                if now - last_flush >= FLUSH_INTERVAL:
                    last_flush = now
                    yield from flush()
            elif kind == "cancelled":
                yield patch_elements(render.gen_notice("cancelled"))
            elif kind == "error":
                yield from flush()
                yield patch_elements(render.gen_error(str(event.get("error", "failed"))))
            elif kind == "done":
                message = _message(thread.id, event.get("message_id"))
                if message is None:
                    continue
                finished = True
                if searched:
                    yield patch_elements(
                        render.message_list(store.get_messages(thread.id))
                    )
                else:
                    yield patch_elements(
                        render.message(message), selector="#gen", mode="replace"
                    )

        if not finished:
            # No terminal event ever arrived. Keep whatever was streamed rather
            # than leaving the bubble stuck on its "..." placeholder.
            yield from flush()
            yield patch_elements(render.gen_notice(""))

        yield patch_elements(_threads_html(thread.id, search))

    def _mode_command(text: str, sig: Signals) -> Response | None:
        """Handle `/think` and `/web on|off`, which set a mode rather than say
        something.

        These live here rather than in `prepare_turn` because they are the web
        UI's own state - the same two signals the checkboxes drive. The REPL
        keeps its own copy for the same reason: there is nothing to send.

        A bare verb toggles, matching the REPL. `/web <query>` is not a mode
        and falls through to the model.
        """
        verb, _, rest = text.partition(" ")
        verb, word = verb.lower(), rest.strip().lower()
        if verb not in ("/think", "/web") or (word and word not in ON_WORDS + OFF_WORDS):
            return None

        signal = "think" if verb == "/think" else "web"
        current = sig.think if verb == "/think" else sig.web
        value = not current if not word else word in ON_WORDS

        # Turning search on when the server has no provider would flip a
        # control the page keeps hidden, which is worse than saying so.
        if signal == "web" and value and not websearch.enabled:
            return _stream([
                patch_signals({"draft": ""}),
                _error_bubble(websearch.unavailable_reason),
            ])
        return _stream([patch_signals({signal: value, "draft": ""})])

    def _error_bubble(message: str) -> str:
        return patch_elements(
            f'<div class="msg assistant"><div class="body">'
            f'<div class="err">{render.escape(message)}</div></div></div>',
            selector="#mlist",
            mode="append",
        )

    @app.post("/ui/send")
    def ui_send(sig: Signals) -> Response:
        text = sig.draft.strip()
        atts = _known(sig.atts)
        if not text and not atts:
            return Response(status_code=204)

        mode = _mode_command(text, sig)
        if mode is not None:
            return mode

        thread = backend.resolve(sig.thread) if sig.thread else None
        if thread is None:
            thread = store.create_thread()

        # Image mode is expressed as the same `/image` and `/edit` verbs the
        # CLI types, so both frontends go down one code path. With something
        # attached, the verb becomes an edit of it.
        if sig.imgmode:
            editing = bool(atts)
            content = ("/edit " if editing else "/image ") + text
            images = atts if editing else []
            strength = sig.strength if editing else None
            thinking = None
            web_search = None
        else:
            content, images, strength = text, atts, None
            thinking = sig.think
            # None leaves the decision to the configured policy; True forces a
            # lookup. The box is never a way to *suppress* an automatic one.
            web_search = True if sig.web else None

        try:
            turn = backend.prepare_turn(
                thread, content, images,
                thinking=thinking,
                search=web_search,
                image_strength=strength,
            )
        except HTTPException as exc:
            # A rejected turn is a message to the user, not a broken page.
            return _stream([_error_bubble(str(exc.detail))])

        return _stream(_turn_frames(thread, turn, sig.search))
