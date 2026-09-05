"""Text generation backed by mlx-vlm.

The configured Qwen model is a vision-language model, so mlx-vlm (not mlx-lm)
is the right loader; it also means the same engine handles image attachments
on user turns.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Iterator

from hearth.config import TextModelConfig


class TextEngine:
    name = "text"

    def __init__(self, cfg: TextModelConfig):
        self.cfg = cfg
        self.model = None
        self.processor = None
        self.config = None
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
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            model, processor = load(self.cfg.repo)
            self.model = model
            self.processor = processor
            try:
                self.config = load_config(self.cfg.repo)
            except Exception:
                self.config = getattr(model, "config", None)
            self.last_used = time.time()

    def unload(self) -> None:
        import mlx.core as mx

        with self._load_lock:
            self.model = None
            self.processor = None
            self.config = None
            mx.clear_cache()

    def _build_prompt(
        self,
        messages: list[dict[str, Any]],
        images: list[str],
        thinking: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Render chat messages with the model's own chat template.

        `enable_thinking` has to go to the template as well as the generator:
        the template decides whether the turn pre-opens a <think> block, and
        mlx-vlm defaults it to False when it is not passed.

        `tools` is forwarded the same way. mlx-vlm passes unrecognised kwargs
        straight through to the tokenizer's own template, so the model renders
        its native tool syntax rather than us hand-writing a schema into the
        system prompt and hoping it matches.
        """
        from mlx_vlm import apply_chat_template

        extra: dict[str, Any] = {"tools": tools} if tools else {}
        return apply_chat_template(
            self.processor,
            self.config,
            messages,
            num_images=len(images),
            add_generation_prompt=True,
            enable_thinking=thinking,
            **extra,
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        images: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        thinking: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield {"type": "token"|"done"} events for one assistant turn."""
        from mlx_vlm import stream_generate

        self.load()
        self.last_used = time.time()
        images = images or []

        want_thinking = thinking if thinking is not None else self.cfg.enable_thinking
        prompt = self._build_prompt(messages, images, want_thinking, tools)

        # With thinking on, the template leaves the prompt ending in "<think>",
        # so generation begins *inside* the reasoning block. With it off the
        # template emits "<think></think>", which does not match this test.
        preopened = prompt.rstrip().endswith("<think>")
        yield {"type": "start", "thinking_open": preopened}

        started = time.time()
        text_parts: list[str] = []
        prompt_tokens = 0
        gen_tokens = 0

        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "top_p": top_p if top_p is not None else self.cfg.top_p,
            "enable_thinking": want_thinking,
        }
        if want_thinking:
            kwargs["thinking_budget"] = self.cfg.thinking_budget

        for result in stream_generate(
            self.model,
            self.processor,
            prompt,
            image=images or None,
            **kwargs,
        ):
            if cancel is not None and cancel.is_set():
                yield {"type": "cancelled"}
                break
            chunk = getattr(result, "text", "") or ""
            if chunk:
                text_parts.append(chunk)
                yield {"type": "token", "text": chunk}
            prompt_tokens = getattr(result, "prompt_tokens", prompt_tokens) or prompt_tokens
            gen_tokens = getattr(result, "generation_tokens", gen_tokens) or gen_tokens
        else:
            elapsed = time.time() - started
            yield {
                "type": "done",
                "text": "".join(text_parts),
                "meta": {
                    "model": self.cfg.repo,
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": gen_tokens,
                    "elapsed_s": round(elapsed, 2),
                    "tokens_per_second": round(gen_tokens / elapsed, 1) if elapsed > 0 else 0.0,
                },
            }
            self.last_used = time.time()
            return

        # Reached only when the loop broke on cancellation.
        elapsed = time.time() - started
        yield {
            "type": "done",
            "text": "".join(text_parts),
            "cancelled": True,
            "meta": {
                "model": self.cfg.repo,
                "prompt_tokens": prompt_tokens,
                "generation_tokens": gen_tokens,
                "elapsed_s": round(elapsed, 2),
                "tokens_per_second": round(gen_tokens / elapsed, 1) if elapsed > 0 else 0.0,
            },
        }
        self.last_used = time.time()
