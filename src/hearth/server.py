"""FastAPI server: owns the models and the conversation store.

Both frontends (CLI and web) are thin clients over this. That split is the
point: an SSH session gets instant startup because the 38GB model is already
resident in this process, and a thread started in the browser can be picked up
from the terminal.
"""
from __future__ import annotations

import base64
import json
import logging
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

    def _history_for_model(thread_id: str) -> list[dict[str, Any]]:
        """Build the message list sent to the model.

        Image messages become a text placeholder: re-feeding generated images
        back through the vision encoder every turn would be slow and is almost
        never what the user means.
        """
        msgs = store.get_messages(thread_id, limit=cfg.text.max_history_messages)
        out: list[dict[str, Any]] = []
        if cfg.text.system_prompt:
            out.append({"role": "system", "content": cfg.text.system_prompt})
        for m in msgs:
            content = m.content
            if m.image and m.role == "assistant":
                prompt = (m.meta or {}).get("prompt", "")
                content = content or f"[generated an image: {prompt}]"
            if not content:
                continue
            out.append({"role": m.role, "content": content})
        return out

    def _materialize_images(images: list[str]) -> list[str]:
        """Accept local paths or data: URIs; return paths mlx-vlm can open."""
        paths = []
        for item in images:
            if item.startswith("data:"):
                header, _, payload = item.partition(",")
                ext = "png" if "png" in header else "jpg"
                dest = cfg.image_dir / f"upload_{int(time.time()*1000)}_{len(paths)}.{ext}"
                dest.write_bytes(base64.b64decode(payload))
                paths.append(str(dest))
            else:
                p = Path(item).expanduser()
                if not p.exists():
                    raise HTTPException(400, f"image not found: {item}")
                paths.append(str(p))
        return paths

    def _autotitle(thread_id: str, first_message: str) -> None:
        thread = store.get_thread(thread_id)
        if thread and thread.title in ("New conversation", "", None):
            title = " ".join(first_message.split())[:60]
            store.rename_thread(thread_id, title or "New conversation")

    def _image_stream(req: ImageRequest, thread_id: str | None) -> Iterator[str]:
        """Shared by POST /api/images and the `/image ...` chat shortcut."""
        job = manager.submit_image(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance=req.guidance,
            seed=req.seed,
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

        # A leading `/image` turns the single chat box into an image request.
        # Doing this server-side means the CLI and the web UI behave the same.
        if text.lower().startswith(IMAGE_PREFIX.strip()) and (
            text.lower().startswith(IMAGE_PREFIX) or text.strip().lower() == "/image"
        ):
            prompt = text[len(IMAGE_PREFIX):].strip() if len(text) > len(IMAGE_PREFIX) else ""
            if not prompt:
                raise HTTPException(400, "usage: /image <prompt>")
            store.add_message(thread.id, "user", text)
            _autotitle(thread.id, prompt)
            return StreamingResponse(
                iterate_in_threadpool(_image_stream(ImageRequest(prompt=prompt), thread.id)),
                media_type="text/event-stream",
            )

        image_paths = _materialize_images(body.images)
        stored_image = Path(image_paths[0]).name if image_paths else None
        store.add_message(thread.id, "user", text, image=stored_image)
        _autotitle(thread.id, text)

        history = _history_for_model(thread.id)
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
            store.add_message(thread_id, "user", f"/image {body.prompt}")
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
        messages = [
            {"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)}
            for m in body.messages
        ]
        job = manager.submit_text(
            messages=messages, max_tokens=body.max_tokens, temperature=body.temperature
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
