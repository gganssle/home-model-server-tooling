"""Owns the models and serializes all GPU work.

Two things matter here:

1. MLX generation is not safe to run concurrently, and two big models racing
   for unified memory is how you wedge the machine. So every generation runs
   on one worker thread, fed by a FIFO queue. Requests queue rather than
   collide.
2. The text model (~38GB at 8-bit) and the image model (~24GB at 4-bit) both
   fit in 96GB together, but not on smaller boxes. `memory.exclusive` evicts
   one before loading the other, and an idle reaper gives memory back when
   nothing is using it.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Iterator

from hearth.config import Config
from hearth.engine.image import ImageEngine
from hearth.engine.text import TextEngine

log = logging.getLogger("hearth.engine")

_SENTINEL = object()


class Job:
    """A queued unit of GPU work whose events the caller consumes as a stream."""

    def __init__(
        self,
        kind: str,
        make_stream: Callable[
            [threading.Event, Callable[[dict[str, Any]], None]], Iterator[dict[str, Any]]
        ],
    ):
        self.kind = kind
        self.make_stream = make_stream
        self.cancel_event = threading.Event()
        self.out: queue.Queue = queue.Queue()
        self.started = threading.Event()
        self.finished = threading.Event()
        self.queued_at = time.time()

    def cancel(self) -> None:
        self.cancel_event.set()

    def emit(self, event: dict[str, Any]) -> None:
        """Push an event from inside a running generation.

        The image engine uses this for denoise progress: it cannot yield those
        events, because it has to stay on the thread that loaded the model.
        """
        self.out.put(event)

    def events(self, queue_notice: bool = True) -> Iterator[dict[str, Any]]:
        """Consume this job's events until completion.

        Emits a `queued` event if the worker is busy, so a client waiting behind
        another request sees why nothing is happening yet.
        """
        if queue_notice and not self.started.is_set():
            yield {"type": "queued"}
        while True:
            item = self.out.get()
            if item is _SENTINEL:
                return
            yield item


class ModelManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.text = TextEngine(cfg.text)
        self.image = ImageEngine(cfg.image, cfg.image_dir)
        self._jobs: queue.Queue[Job] = queue.Queue()
        self._current: Job | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._worker = threading.Thread(target=self._run, name="hearth-worker", daemon=True)
        self._worker.start()
        self._reaper = threading.Thread(target=self._reap, name="hearth-reaper", daemon=True)
        self._reaper.start()

    # ---------------- public API ----------------

    def submit_text(self, **kwargs: Any) -> Job:
        def make(cancel: threading.Event, emit):
            if self.cfg.memory.exclusive and self.image.loaded:
                log.info("exclusive mode: evicting image model to make room for text")
                self.image.unload()
            return self.text.stream(cancel=cancel, **kwargs)

        return self._submit(Job("text", make))

    def submit_image(self, **kwargs: Any) -> Job:
        def make(cancel: threading.Event, emit):
            if self.cfg.memory.exclusive and self.text.loaded:
                log.info("exclusive mode: evicting text model to make room for image")
                self.text.unload()
            return self.image.stream(cancel=cancel, emit=emit, **kwargs)

        return self._submit(Job("image", make))

    def preload(self, which: str) -> None:
        """Warm a model without generating anything."""
        job = Job(f"preload:{which}", lambda cancel, emit: self._preload_stream(which))
        self._submit(job)
        for _ in job.events(queue_notice=False):
            pass

    def _preload_stream(self, which: str) -> Iterator[dict[str, Any]]:
        yield {"type": "status", "text": f"loading {which} model"}
        engine = self.text if which == "text" else self.image
        engine.load()
        yield {"type": "done", "text": f"{which} model loaded"}

    def unload(self, which: str = "all") -> list[str]:
        freed = []
        if which in ("all", "text") and self.text.loaded:
            self.text.unload()
            freed.append("text")
        if which in ("all", "image") and self.image.loaded:
            self.image.unload()
            freed.append("image")
        return freed

    def cancel_current(self) -> bool:
        with self._lock:
            if self._current is not None:
                self._current.cancel()
                return True
        return False

    def status(self) -> dict[str, Any]:
        import mlx.core as mx

        with self._lock:
            current = self._current.kind if self._current else None
        return {
            "text": {
                "repo": self.cfg.text.repo,
                "loaded": self.text.loaded,
                "idle_s": round(time.time() - self.text.last_used, 1) if self.text.loaded else None,
            },
            "image": {
                "repo": self.cfg.image.repo,
                "loaded": self.image.loaded,
                "idle_s": round(time.time() - self.image.last_used, 1) if self.image.loaded else None,
            },
            "busy_with": current,
            "queue_depth": self._jobs.qsize(),
            "memory": {
                "active_gb": round(mx.get_active_memory() / 1e9, 2),
                "cache_gb": round(mx.get_cache_memory() / 1e9, 2),
                "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            },
            "settings": {
                "exclusive": self.cfg.memory.exclusive,
                "idle_evict_seconds": self.cfg.memory.idle_evict_seconds,
            },
        }

    def shutdown(self) -> None:
        self._stop.set()

    # ---------------- internals ----------------

    def _submit(self, job: Job) -> Job:
        self._jobs.put(job)
        return job

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._lock:
                self._current = job
            job.started.set()
            try:
                for event in job.make_stream(job.cancel_event, job.emit):
                    job.out.put(event)
            except Exception as exc:
                log.exception("job %s failed", job.kind)
                job.out.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            finally:
                job.out.put(_SENTINEL)
                job.finished.set()
                with self._lock:
                    self._current = None

    def _reap(self) -> None:
        """Evict models that have gone unused, returning unified memory to the OS."""
        while not self._stop.is_set():
            self._stop.wait(30)
            ttl = self.cfg.memory.idle_evict_seconds
            if ttl <= 0:
                continue
            with self._lock:
                busy = self._current is not None
            if busy or self._jobs.qsize():
                continue
            now = time.time()
            for engine in (self.text, self.image):
                if engine.loaded and now - engine.last_used > ttl:
                    log.info("evicting idle %s model after %.0fs", engine.name, now - engine.last_used)
                    engine.unload()
