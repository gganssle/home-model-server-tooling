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
import shutil
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
from hearth.store import Store
from hearth.textutil import ThinkSplitter

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

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()
        store.close()

    app = FastAPI(title="hearth", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.store = store
    app.state.manager = manager

    # ---------------- meta ----------------

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        st = manager.status()
        st["threads"] = len(store.list_threads(limit=1000))
        st["images"] = config_mod.image_store_stats(cfg)
        st["version"] = "0.1.0"
        return st

    @app.post("/api/cancel")
    def cancel() -> dict[str, Any]:
        return {"cancelled": manager.cancel_current()}

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
        budget = max(0, cfg.text.max_history_images)
        keep: set[str] = set()
        for m in reversed(msgs):
            if m.role != "user":
                continue
            for name in reversed((m.meta or {}).get("images", []) or []):
                if len(keep) >= budget:
                    break
                keep.add(name)

        out: list[dict[str, Any]] = []
        paths: list[str] = []
        if cfg.text.system_prompt:
            out.append({"role": "system", "content": cfg.text.system_prompt})

        for m in msgs:
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

        attachments = _materialize_images(body.images)
        store.add_message(
            thread.id, "user", text,
            meta={"images": attachments} if attachments else None,
        )
        _autotitle(thread.id, text or "image")

        # History is rebuilt after storing, so this turn's attachments are
        # included and the marker order matches the paths exactly.
        history, image_paths = _history_for_model(thread.id)
        job = manager.submit_text(
            messages=history,
            images=image_paths,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            thinking=body.thinking,
        )

        def gen() -> Iterator[str]:
            splitter = ThinkSplitter()
            saved = False

            def persist(meta: dict[str, Any], cancelled: bool) -> Iterator[str]:
                """Write the assistant turn exactly once, however the stream ended.

                Partial output from a cancelled or failed generation is still
                worth keeping - it is what the user watched appear.
                """
                nonlocal saved
                if saved:
                    return
                saved = True
                for channel, piece in splitter.finish():
                    yield sse({"type": "token", "channel": channel, "text": piece})
                meta = dict(meta)
                if splitter.thinking_text:
                    meta["thinking"] = splitter.thinking_text
                if cancelled:
                    meta["cancelled"] = True
                content = splitter.content_text.strip()
                msg = store.add_message(thread.id, "assistant", content, meta=meta)
                yield sse({
                    "type": "done",
                    "message_id": msg.id,
                    "content": content,
                    "meta": meta,
                    "cancelled": cancelled,
                })

            for event in job.events():
                etype = event.get("type")
                if etype == "start":
                    # Internal: tells us whether the prompt pre-opened <think>.
                    splitter = ThinkSplitter(start_in_think=event.get("thinking_open", False))
                elif etype == "token":
                    for channel, piece in splitter.feed(event["text"]):
                        yield sse({"type": "token", "channel": channel, "text": piece})
                elif etype == "done":
                    yield from persist(event.get("meta") or {}, event.get("cancelled", False))
                elif etype == "cancelled":
                    yield sse(event)
                    yield from persist({}, True)
                elif etype == "error":
                    yield sse(event)
                    yield from persist({"error": event.get("error")}, False)
                else:
                    yield sse(event)

            # A stream that ended without any terminal event still gets saved.
            yield from persist({}, False)

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
