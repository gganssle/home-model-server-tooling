"""Image generation backed by mflux's MLX-native Qwen-Image implementation."""
from __future__ import annotations

import random
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from hearth.config import ImageModelConfig


class _StepReporter:
    """mflux in-loop callback that reports denoising progress.

    Also the cancellation hook: raising inside the loop is how mflux expects a
    generation to be interrupted.
    """

    def __init__(self, total: int, emit: Callable[[dict[str, Any]], None],
                 cancel: threading.Event | None):
        self.total = total
        self.emit = emit
        self.cancel = cancel

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        if self.cancel is not None and self.cancel.is_set():
            from mflux.utils.exceptions import StopImageGenerationException

            raise StopImageGenerationException("cancelled by user")
        # `t` is the zero-based index of the step just completed.
        self.emit({"type": "progress", "step": min(int(t) + 1, self.total), "total": self.total})


def _image_size(path: str) -> tuple[int, int]:
    """Pixel dimensions of a base image, rounded to the multiple of 16 that
    mflux's latent grid requires."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    return max(16, (w // 16) * 16), max(16, (h // 16) * 16)


class ImageEngine:
    name = "image"

    def __init__(self, cfg: ImageModelConfig, image_dir: Path):
        self.cfg = cfg
        self.image_dir = image_dir
        self.model = None
        self.last_used = 0.0
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def repo(self) -> str:
        return self.cfg.repo

    def load(self) -> None:
        with self._load_lock:
            if self.model is not None:
                return
            from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

            # The mlx-community repos are already quantized, so quantize=None
            # leaves the on-disk precision alone.
            self.model = QwenImage(model_path=self.cfg.repo, quantize=None)
            self.last_used = time.time()

    def unload(self) -> None:
        import mlx.core as mx

        with self._load_lock:
            self.model = None
            mx.clear_cache()

    def stream(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        init_image: str | None = None,
        image_strength: float | None = None,
        cancel: threading.Event | None = None,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Generate one image, reporting progress as it denoises.

        Progress goes out through `emit` rather than being yielded, because
        generation has to run on this very thread: MLX streams are per-thread,
        and mflux captures the loading thread's stream, so running the denoise
        loop anywhere else fails with "There is no Stream(cpu, N)". Yielding
        progress would require a second thread, which is exactly what breaks.
        """
        from mflux.utils.exceptions import StopImageGenerationException

        yield {"type": "status", "text": "loading image model"}
        self.load()
        self.last_used = time.time()

        steps = steps or self.cfg.steps
        guidance = guidance if guidance is not None else self.cfg.guidance
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        if init_image is not None:
            # Working from a base image: match its shape unless told otherwise,
            # so the result is a variation rather than a reframing.
            if width is None or height is None:
                src_w, src_h = _image_size(init_image)
                width = width or src_w
                height = height or src_h
            if image_strength is None:
                image_strength = self.cfg.image_strength
        width = width or self.cfg.width
        height = height or self.cfg.height

        sink = emit if emit is not None else (lambda _event: None)
        reporter = _StepReporter(steps, sink, cancel)
        self.model.callbacks.register(reporter)

        started = time.time()
        try:
            image = self.model.generate_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance=guidance,
                num_inference_steps=steps,
                image_path=init_image,
                image_strength=image_strength,
            )
        except StopImageGenerationException:
            yield {"type": "cancelled"}
            return
        except Exception as exc:  # surfaced to the client as an error event
            yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
            return
        finally:
            # The registry lives on the model, which outlives this call, so a
            # failed generation must not leave its reporter behind.
            try:
                self.model.callbacks.in_loop.remove(reporter)
            except (AttributeError, ValueError):
                pass

        filename = f"img_{int(time.time() * 1000)}_{seed}.png"
        path = self.image_dir / filename
        self.image_dir.mkdir(parents=True, exist_ok=True)
        # overwrite=True: without it mflux resolves a *different* path when the
        # file exists, and the name we record would not match the file on disk.
        image.save(path=str(path), overwrite=True)
        self.last_used = time.time()

        yield {
            "type": "done",
            "image": filename,
            "meta": {
                "model": self.cfg.repo,
                "seed": seed,
                "steps": steps,
                "width": width,
                "height": height,
                "guidance": guidance,
                "elapsed_s": round(time.time() - started, 2),
                **({"image_strength": image_strength, "from_image": Path(init_image).name}
                   if init_image else {}),
            },
        }
