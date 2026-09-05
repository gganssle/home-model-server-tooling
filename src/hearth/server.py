"""FastAPI server: owns the models and the conversation store.

Both frontends (CLI and web) are thin clients over this. That split is the
point: an SSH session gets instant startup because the 38GB model is already
resident in this process, and a thread started in the browser can be picked up
from the terminal.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import queue
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import iterate_in_threadpool

from hearth import config as config_mod
from hearth.engine import ModelManager
from hearth.search import TOOL_SCHEMA, Outcome, WebSearch
from hearth.search import budget as search_budget
from hearth.store import Store
from hearth.textutil import ThinkSplitter, ToolCallSplitter, tool_call_query

log = logging.getLogger("hearth.server")

WEB_DIR = Path(__file__).parent / "web"
IMAGE_PREFIX = "/image "

_EXT_BY_MIME = {
    "image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
    "image/webp": "webp", "image/gif": "gif",
}


def _looks_like_image(path: Path) -> bool:
    """Verify a file really is an image before we hand it to a model.

    An attachment arrives as bytes from a browser or a path from a shell; both
    can be something else entirely, and the failure deep inside the vision
    tower is far less clear than a 400 here.
    """
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


class NewThread(BaseModel):
    title: str = "New conversation"


class RenameThread(BaseModel):
    title: str


class ChatRequest(BaseModel):
    # Strict: an unknown field is a client bug, and silently ignoring it hides
    # the bug until someone notices a flag doing nothing.
    model_config = ConfigDict(extra="forbid")

    content: str
    images: list[str] = Field(default_factory=list, description="Local paths or data: URIs")
    max_tokens: int | None = None
    temperature: float | None = None
    thinking: bool | None = None
    # True forces a web search for this turn, False suppresses one that would
    # otherwise be automatic, None leaves the decision to the configured policy.
    search: bool | None = None


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    thread_id: str | None = None
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    # img2img: a filename already in the image directory, a path, or a data URI.
    init_image: str | None = None
    # 0-1. Low stays close to the base image, high barely resembles it.
    image_strength: float | None = Field(default=None, ge=0.0, le=1.0)


class OpenAIMessage(BaseModel):
    role: str
    content: Any


class OpenAIChatRequest(BaseModel):
    model: str | None = None
    messages: list[OpenAIMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None


def create_app(cfg: config_mod.Config | None = None) -> FastAPI:
    cfg = cfg or config_mod.load()
    store = Store(cfg.db_path)
    manager = ModelManager(cfg)
    websearch = WebSearch(cfg.search)

    # Retrieval happens before a job exists, so manager.cancel_current() cannot
    # reach it. Each in-flight turn parks a flag here for /api/cancel to set.
    search_cancels: set[threading.Event] = set()
    search_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()
        store.close()

    app = FastAPI(title="hearth", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.store = store
    app.state.manager = manager
    app.state.websearch = websearch

    # ---------------- meta ----------------

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        st = manager.status()
        st["threads"] = len(store.list_threads(limit=1000))
        st["images"] = config_mod.image_store_stats(cfg)
        st["search"] = {
            "enabled": websearch.enabled,
            "provider": cfg.search.provider if websearch.enabled else None,
            "autonomous": cfg.search.autonomous,
            "reason": None if websearch.enabled else websearch.unavailable_reason,
        }
        st["version"] = "0.1.0"
        return st

    @app.post("/api/cancel")
    def cancel() -> dict[str, Any]:
        """Stop whatever this server is doing for someone right now.

        That is two different things: a running generation, which the manager
        owns, and a retrieval, which happens in the request path before any job
        exists. Cancelling only the first leaves the user watching a fetch they
        already asked to stop.
        """
        with search_lock:
            pending = list(search_cancels)
        for flag in pending:
            flag.set()
        stopped_job = manager.cancel_current()
        return {"cancelled": stopped_job or bool(pending), "searches": len(pending)}

    @app.post("/api/models/{which}/preload")
    def preload(which: str) -> dict[str, Any]:
        if which not in ("text", "image"):
            raise HTTPException(400, "which must be 'text' or 'image'")
        manager.preload(which)
        return {"loaded": which}

    @app.post("/api/models/unload")
    def unload(which: str = "all") -> dict[str, Any]:
        return {"unloaded": manager.unload(which)}

    # ---------------- threads ----------------

    @app.get("/api/threads")
    def list_threads() -> dict[str, Any]:
        return {"threads": [t.to_dict() for t in store.list_threads()]}

    @app.post("/api/threads")
    def create_thread(body: NewThread) -> dict[str, Any]:
        return store.create_thread(body.title).to_dict()

    def _resolve(ref: str):
        thread = store.resolve_thread(ref)
        if thread is None:
            raise HTTPException(404, f"no thread matching {ref!r}")
        return thread

    @app.get("/api/threads/{ref}")
    def get_thread(ref: str) -> dict[str, Any]:
        thread = _resolve(ref)
        return {
            "thread": thread.to_dict(),
            "messages": [m.to_dict() for m in store.get_messages(thread.id)],
        }

    @app.patch("/api/threads/{ref}")
    def rename_thread(ref: str, body: RenameThread) -> dict[str, Any]:
        thread = _resolve(ref)
        store.rename_thread(thread.id, body.title)
        return store.get_thread(thread.id).to_dict()

    @app.delete("/api/threads/{ref}")
    def delete_thread(ref: str) -> dict[str, Any]:
        thread = _resolve(ref)
        store.delete_thread(thread.id)
        return {"deleted": thread.id}

    @app.get("/api/search")
    def search(q: str) -> dict[str, Any]:
        return {"results": store.search(q)}

    # ---------------- chat ----------------

    def _history_for_model(thread_id: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Build the message list sent to the model, plus the images it refers to.

        Attachments on a user turn are carried forward as `{"type": "image"}`
        markers in that turn's content, which is how mlx-vlm knows which turn
        each image belongs to. The returned paths must stay in the same order
        as those markers.

        Only the most recent few images are carried: the vision tower re-encodes
        every one on every turn. Older attachments degrade to a text note so the
        model still knows they existed.

        Images the model *generated* are never fed back - they are described in
        text instead, since asking about them is rare and re-encoding them every
        turn is not.
        """
        msgs = store.get_messages(thread_id, limit=cfg.text.max_history_messages)

        # Walk backwards to decide which attachments still fit the budget.
        image_budget = max(0, cfg.text.max_history_images)
        keep: set[str] = set()
        for m in reversed(msgs):
            if m.role != "user":
                continue
            for name in reversed((m.meta or {}).get("images", []) or []):
                if len(keep) >= image_budget:
                    break
                keep.add(name)

        # The same walk, for retrieved pages. One search is bigger than the
        # rest of a thread put together, so only the newest few keep their full
        # text; everything older collapses to the one-line source list that is
        # already stored as the message's content.
        doc_budget = max(0, cfg.search.max_history_documents)
        verbatim: set[str] = set()
        for m in reversed(msgs):
            if m.role != "tool" or not (m.meta or {}).get("search"):
                continue
            if len(verbatim) >= doc_budget:
                break
            verbatim.add(m.id)

        out: list[dict[str, Any]] = []
        paths: list[str] = []
        preamble = config_mod.system_prompt(cfg)
        if preamble:
            out.append({"role": "system", "content": preamble})

        for m in msgs:
            if m.role == "tool":
                record = (m.meta or {}).get("search") or {}
                block = ""
                if m.id in verbatim:
                    block = search_budget.pack(
                        record.get("documents") or [],
                        cfg.search.max_context_chars,
                        record.get("query", ""),
                    )
                # Retrieved text is projected onto the user role rather than
                # sent as role="tool". A bare tool message with no matching
                # tool_call ahead of it is not a shape every chat template
                # handles, and when a template mishandles it the result is a
                # quietly malformed prompt rather than an error. The store keeps
                # the honest role - only what the model sees is flattened.
                out.append({"role": "user", "content": block or m.content})
                continue

            content: Any = m.content
            if m.image and m.role == "assistant":
                prompt = (m.meta or {}).get("prompt", "")
                content = content or f"[generated an image: {prompt}]"

            attached = (m.meta or {}).get("images", []) or [] if m.role == "user" else []
            carried = [n for n in attached if n in keep and (cfg.image_dir / n).exists()]
            dropped = len(attached) - len(carried)

            if carried:
                parts: list[dict[str, Any]] = [{"type": "image"} for _ in carried]
                text = content or ""
                if dropped:
                    text = f"[{dropped} earlier image(s) omitted] {text}".strip()
                parts.append({"type": "text", "text": text})
                out.append({"role": m.role, "content": parts})
                paths.extend(str(cfg.image_dir / n) for n in carried)
                continue

            if dropped:
                content = f"[attached {dropped} image(s), no longer in context] {content or ''}".strip()
            if not content:
                continue
            out.append({"role": m.role, "content": content})

        return out, paths

    def _materialize_images(images: list[str]) -> list[str]:
        """Take attachments into the image directory and return their filenames.

        Accepts a data: URI, a filename already in the image directory, or a
        path on the server's filesystem. Everything is copied in rather than
        referenced in place, so an attachment still renders after the original
        is moved or deleted, and so it can be served over HTTP.
        """
        names: list[str] = []
        cfg.image_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(images):
            stamp = f"{int(time.time() * 1000)}_{index}"
            if item.startswith("data:"):
                header, _, payload = item.partition(",")
                ext = _EXT_BY_MIME.get(header.split(";")[0].removeprefix("data:"), "png")
                dest = cfg.image_dir / f"att_{stamp}.{ext}"
                try:
                    dest.write_bytes(base64.b64decode(payload, validate=True))
                except (binascii.Error, ValueError) as exc:
                    raise HTTPException(400, f"malformed data URI: {exc}") from exc
            else:
                # A bare filename may already be one of ours; prefer that over
                # touching the wider filesystem.
                existing = cfg.image_dir / Path(item).name
                if "/" not in item and existing.exists():
                    names.append(existing.name)
                    continue
                src = Path(item).expanduser()
                if not src.is_file():
                    raise HTTPException(400, f"image not found: {item}")
                suffix = src.suffix.lower().lstrip(".") or "png"
                dest = cfg.image_dir / f"att_{stamp}.{suffix}"
                shutil.copyfile(src, dest)

            if not _looks_like_image(dest):
                dest.unlink(missing_ok=True)
                raise HTTPException(400, f"not a readable image: {item}")
            names.append(dest.name)
        return names

    def _resolve_image_ref(ref: str) -> str:
        """Turn an image reference into an absolute path under the image dir."""
        name = _materialize_images([ref])[0]
        return str(cfg.image_dir / name)

    def _openai_messages(
        messages: list["OpenAIMessage"],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Convert OpenAI-style messages, keeping any images they carry.

        A content array may hold image_url parts; those become `{"type":
        "image"}` markers on the turn they arrived with, which is how mlx-vlm
        attributes each image to the right message.
        """
        out: list[dict[str, Any]] = []
        paths: list[str] = []
        for m in messages:
            if isinstance(m.content, str):
                out.append({"role": m.role, "content": m.content})
                continue
            if not isinstance(m.content, list):
                out.append({"role": m.role, "content": str(m.content)})
                continue

            parts: list[dict[str, Any]] = []
            texts: list[str] = []
            for item in m.content:
                if not isinstance(item, dict):
                    texts.append(str(item))
                    continue
                kind = item.get("type")
                if kind in ("text", "input_text"):
                    texts.append(item.get("text") or item.get("content") or "")
                elif kind in ("image_url", "input_image", "image"):
                    url = item.get("image_url") or item.get("image") or item.get("url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if not url:
                        continue
                    name = _materialize_images([url])[0]
                    paths.append(str(cfg.image_dir / name))
                    parts.append({"type": "image"})
            parts.append({"type": "text", "text": " ".join(t for t in texts if t).strip()})
            out.append({"role": m.role, "content": parts if len(parts) > 1 else parts[0]["text"]})
        return out, paths

    def _latest_image(thread_id: str) -> str | None:
        """Newest image in a thread, generated or attached."""
        for m in reversed(store.get_messages(thread_id)):
            if m.image and (cfg.image_dir / m.image).exists():
                return m.image
            for name in reversed((m.meta or {}).get("images", []) or []):
                if (cfg.image_dir / name).exists():
                    return name
        return None

    def _run_search(
        thread_id: str, query: str, cancel: threading.Event, reason: str | None = None
    ):
        """Retrieve, streaming progress, and record the result on the thread.

        This runs on the threadpool thread that is producing the response body,
        so the network wait is off the event loop and - the part that matters -
        off the manager's worker thread, which must only ever be doing GPU work.

        The retrieval itself goes on a side thread so its phase events can be
        streamed as they happen rather than arriving in a lump at the end;
        exactly the shape Job already uses for the engines. Returns the Outcome
        via `yield from`.
        """
        events: queue.Queue = queue.Queue()
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["outcome"] = websearch.run(
                    query, emit=events.put, cancel=cancel, reason=reason
                )
            except Exception as exc:
                log.exception("web search crashed")
                box["crashed"] = True
                box["outcome"] = Outcome(query=query, error=f"{type(exc).__name__}: {exc}")
            finally:
                events.put(None)

        worker = threading.Thread(target=work, name="hearth-search", daemon=True)
        worker.start()
        while True:
            event = events.get()
            if event is None:
                break
            yield sse(event)
        worker.join()

        outcome: Outcome = box["outcome"]
        if box.get("crashed"):
            yield sse({"type": "search", "phase": "error", "error": outcome.error})

        # Stored even when it failed or was cancelled: the transcript should say
        # that a lookup was attempted, and the compact line is what says it.
        store.add_message(
            thread_id, "tool", outcome.compact(), meta={"search": outcome.to_meta()}
        )
        return outcome

    def _autotitle(thread_id: str, first_message: str) -> None:
        thread = store.get_thread(thread_id)
        if thread and thread.title in ("New conversation", "", None):
            title = " ".join(first_message.split())[:60]
            store.rename_thread(thread_id, title or "New conversation")

    def _image_stream(req: ImageRequest, thread_id: str | None) -> Iterator[str]:
        """Shared by POST /api/images and the `/image ...` chat shortcut."""
        init_image = _resolve_image_ref(req.init_image) if req.init_image else None
        job = manager.submit_image(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance=req.guidance,
            seed=req.seed,
            init_image=init_image,
            image_strength=req.image_strength,
        )
        for event in job.events():
            if event.get("type") == "done":
                meta = dict(event.get("meta") or {})
                meta["prompt"] = req.prompt
                if thread_id:
                    msg = store.add_message(
                        thread_id, "assistant", "", image=event["image"], meta=meta
                    )
                    event["message_id"] = msg.id
                event["meta"] = meta
                event["url"] = f"/api/images/{event['image']}"
            yield sse(event)

    @app.post("/api/threads/{ref}/messages")
    async def post_message(ref: str, body: ChatRequest, request: Request):
        thread = _resolve(ref)
        text = body.content.strip()
        if not text and not body.images:
            raise HTTPException(400, "empty message")

        # A leading `/image` or `/edit` turns the single chat box into an image
        # request. Doing this server-side means both frontends behave the same.
        verb, _, rest = text.partition(" ")
        verb = verb.lower()
        if verb in ("/image", "/edit"):
            prompt = rest.strip()
            if not prompt:
                raise HTTPException(400, f"usage: {verb} <prompt>")

            init_image = None
            if verb == "/edit":
                # Edit whatever was just attached, else the newest image in the
                # thread - which is what "make the sky stormier" ought to mean.
                init_image = (
                    _materialize_images(body.images)[0] if body.images
                    else _latest_image(thread.id)
                )
                if init_image is None:
                    raise HTTPException(
                        400, "/edit needs an image: attach one, or generate one first"
                    )

            store.add_message(thread.id, "user", text)
            _autotitle(thread.id, prompt)
            return StreamingResponse(
                iterate_in_threadpool(_image_stream(
                    ImageRequest(prompt=prompt, init_image=init_image), thread.id
                )),
                media_type="text/event-stream",
            )

        # `/web <query>` searches before answering. Routed on the leading verb
        # like /image and /edit, so both frontends behave identically without
        # either of them implementing it.
        forced_query: str | None = None
        if verb == "/web":
            forced_query = rest.strip()
            if not forced_query:
                raise HTTPException(400, "usage: /web <query>")

        attachments = _materialize_images(body.images)
        store.add_message(
            thread.id, "user", text,
            meta={"images": attachments} if attachments else None,
        )
        _autotitle(thread.id, forced_query or text or "image")

        # Three ways a turn can end up searching, in descending order of how
        # much the user meant it.
        if body.search is False:
            want_search, search_why = False, "suppressed for this message"
        elif forced_query is not None or body.search:
            want_search, search_why = True, "requested"
        else:
            want_search, search_why = websearch.wants_search(text)

        # Without an explicit query, the message itself is the query. No model
        # call rewrites it - that would cost a whole extra generation - so a
        # turn that leans on earlier context ("what about the second one?")
        # searches badly. `/web` exists for exactly that case.
        search_text = forced_query or text

        offer_tools = (
            websearch.enabled
            and cfg.search.autonomous == "tool"
            and body.search is not False
        )

        def gen() -> Iterator[str]:
            cancel = threading.Event()
            with search_lock:
                search_cancels.add(cancel)

            saved = False
            answer: list[str] = []
            reasoning: list[str] = []
            sources: list[dict[str, Any]] = []

            def persist(meta: dict[str, Any], cancelled: bool) -> Iterator[str]:
                """Write the assistant turn exactly once, however the stream ended.

                Partial output from a cancelled or failed generation is still
                worth keeping - it is what the user watched appear.
                """
                nonlocal saved
                if saved:
                    return
                saved = True
                meta = dict(meta)
                if reasoning:
                    meta["thinking"] = "".join(reasoning)
                if sources:
                    meta["sources"] = sources
                if cancelled:
                    meta["cancelled"] = True
                content = "".join(answer).strip()
                msg = store.add_message(thread.id, "assistant", content, meta=meta)
                yield sse({
                    "type": "done",
                    "message_id": msg.id,
                    "content": content,
                    "meta": meta,
                    "cancelled": cancelled,
                })

            def emit_token(channel: str, piece: str) -> str:
                (reasoning if channel == "thinking" else answer).append(piece)
                return sse({"type": "token", "channel": channel, "text": piece})

            try:
                # ---- retrieval first, before any GPU work is queued ----
                if want_search and not websearch.enabled:
                    yield sse({"type": "search", "phase": "error",
                               "error": websearch.unavailable_reason})
                elif want_search:
                    log.info("searching (%s): %s", search_why, search_text)
                    outcome = yield from _run_search(
                        thread.id, search_text, cancel, search_why
                    )
                    sources.extend(outcome.sources())
                    if outcome.cancelled or cancel.is_set():
                        yield sse({"type": "cancelled"})
                        yield from persist({}, True)
                        return

                rounds = 0
                while True:
                    # History is rebuilt each round, after storing, so this
                    # turn's attachments and any retrieval just performed are
                    # both included and the image markers line up with paths.
                    history, image_paths = _history_for_model(thread.id)
                    job = manager.submit_text(
                        messages=history,
                        images=image_paths,
                        max_tokens=body.max_tokens,
                        temperature=body.temperature,
                        thinking=body.thinking,
                        tools=[TOOL_SCHEMA] if offer_tools else None,
                    )

                    think = ThinkSplitter()
                    # Reasoning is separated first; only the content channel is
                    # scanned for tool calls, so a <tool_call> the model muses
                    # about inside <think> is not mistaken for a real one.
                    calls = ToolCallSplitter(enabled=offer_tools)
                    terminal: dict[str, Any] = {}

                    for event in job.events():
                        etype = event.get("type")
                        if etype == "start":
                            think = ThinkSplitter(
                                start_in_think=event.get("thinking_open", False)
                            )
                        elif etype == "token":
                            for channel, piece in think.feed(event["text"]):
                                if channel == "thinking":
                                    yield emit_token("thinking", piece)
                                else:
                                    for visible in calls.feed(piece):
                                        yield emit_token("content", visible)
                        elif etype in ("done", "cancelled", "error"):
                            terminal = event
                            if etype != "done":
                                yield sse(event)
                        else:
                            yield sse(event)

                    for channel, piece in think.finish():
                        if channel == "thinking":
                            yield emit_token("thinking", piece)
                        else:
                            for visible in calls.feed(piece):
                                yield emit_token("content", visible)
                    for visible in calls.finish():
                        yield emit_token("content", visible)

                    cancelled = bool(terminal.get("cancelled")) or \
                        terminal.get("type") == "cancelled" or cancel.is_set()
                    meta = dict(terminal.get("meta") or {})
                    if terminal.get("type") == "error":
                        meta["error"] = terminal.get("error")

                    query = next(
                        (q for call in calls.calls if (q := tool_call_query(call))), None
                    )
                    keep_going = (
                        query
                        and offer_tools
                        and not cancelled
                        and terminal.get("type") != "error"
                        and rounds < max(0, cfg.search.max_rounds)
                    )
                    if keep_going:
                        # The model's preamble for this round has already been
                        # streamed but is not written back into history: the
                        # next round sees the question and the sources, which is
                        # the context that actually helps it answer.
                        rounds += 1
                        outcome = yield from _run_search(
                            thread.id, query, cancel, "the model asked"
                        )
                        sources.extend(outcome.sources())
                        if not (outcome.cancelled or cancel.is_set()):
                            continue
                        cancelled = True

                    yield from persist(meta, cancelled)
                    return
            finally:
                with search_lock:
                    search_cancels.discard(cancel)

        return StreamingResponse(
            iterate_in_threadpool(gen()), media_type="text/event-stream"
        )

    # ---------------- images ----------------

    @app.post("/api/images")
    async def generate_image(body: ImageRequest):
        thread_id = None
        if body.thread_id:
            thread_id = _resolve(body.thread_id).id
            verb = "/edit" if body.init_image else "/image"
            store.add_message(thread_id, "user", f"{verb} {body.prompt}")
            _autotitle(thread_id, body.prompt)
        return StreamingResponse(
            iterate_in_threadpool(_image_stream(body, thread_id)),
            media_type="text/event-stream",
        )

    @app.get("/api/images/{filename}")
    def get_image(filename: str):
        # Resolve and confine to the image directory: never serve outside it.
        path = (cfg.image_dir / filename).resolve()
        if not str(path).startswith(str(cfg.image_dir.resolve())) or not path.exists():
            raise HTTPException(404, "no such image")
        return FileResponse(path, media_type="image/png")

    # ---------------- OpenAI-compatible ----------------

    @app.get("/v1/models")
    def openai_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": cfg.text.repo, "object": "model", "owned_by": "hearth"},
                {"id": cfg.image.repo, "object": "model", "owned_by": "hearth"},
            ],
        }

    @app.post("/v1/chat/completions")
    async def openai_chat(body: OpenAIChatRequest):
        """Enough of the OpenAI shape to point other local tools at this box."""
        messages, oai_images = _openai_messages(body.messages)
        job = manager.submit_text(
            messages=messages, images=oai_images,
            max_tokens=body.max_tokens, temperature=body.temperature,
        )
        created = int(time.time())
        cid = f"chatcmpl-{created}"

        if not body.stream:
            splitter = ThinkSplitter()
            meta: dict[str, Any] = {}
            for event in job.events():
                if event.get("type") == "start":
                    splitter = ThinkSplitter(start_in_think=event.get("thinking_open", False))
                elif event.get("type") == "token":
                    splitter.feed(event["text"])
                elif event.get("type") == "done":
                    splitter.finish()
                    meta = event.get("meta") or {}
                elif event.get("type") == "error":
                    raise HTTPException(500, event["error"])
            return {
                "id": cid, "object": "chat.completion", "created": created,
                "model": cfg.text.repo,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": splitter.content_text.strip()},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": meta.get("prompt_tokens", 0),
                    "completion_tokens": meta.get("generation_tokens", 0),
                    "total_tokens": meta.get("prompt_tokens", 0) + meta.get("generation_tokens", 0),
                },
            }

        def gen() -> Iterator[str]:
            splitter = ThinkSplitter()
            for event in job.events():
                if event.get("type") == "start":
                    splitter = ThinkSplitter(start_in_think=event.get("thinking_open", False))
                elif event.get("type") == "token":
                    for channel, piece in splitter.feed(event["text"]):
                        if channel != "content":
                            continue
                        yield sse({
                            "id": cid, "object": "chat.completion.chunk", "created": created,
                            "model": cfg.text.repo,
                            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                        })
                elif event.get("type") == "done":
                    for channel, piece in splitter.finish():
                        if channel == "content":
                            yield sse({
                                "id": cid, "object": "chat.completion.chunk", "created": created,
                                "model": cfg.text.repo,
                                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                            })
                    yield sse({
                        "id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": cfg.text.repo,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    })
                    yield "data: [DONE]\n\n"

        return StreamingResponse(iterate_in_threadpool(gen()), media_type="text/event-stream")

    # ---------------- web UI ----------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "index.html").read_text()

    return app


def serve(cfg: config_mod.Config | None = None) -> None:
    import uvicorn

    cfg = cfg or config_mod.load()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port, log_level="info")
