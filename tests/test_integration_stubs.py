"""Stub replacements for the two model engines, shared by the test scripts.

These mirror the real engines' signatures and event contracts closely enough
that a mismatch shows up here rather than only under 60GB of weights.
"""
from __future__ import annotations

import time
from pathlib import Path


def _make_png(width: int = 8, height: int = 8, colour=(200, 90, 40)) -> bytes:
    """A real, checksum-valid PNG.

    Generated rather than hard-coded: a hand-written blob is easy to get subtly
    wrong, and the server verifies attachments with PIL before using them.
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="PNG")
    return buf.getvalue()


SAMPLE_PNG = _make_png()


def _describe(messages):
    """Summarise the message list the way the real prompt builder sees it.

    Returns (text of the last user turn, number of image markers across all
    turns) so tests can assert that vision plumbing reached the engine.
    """
    last_text = ""
    markers = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image":
                    markers += 1
                elif part.get("type") == "text" and m.get("role") == "user":
                    last_text = part.get("text", "")
        elif m.get("role") == "user":
            last_text = content or ""
    return last_text, markers


def fake_text_stream(self, messages, images=None, max_tokens=None, temperature=None,
                     top_p=None, thinking=None, cancel=None):
    """Echo the last user turn, wrapped in a think block when reasoning is on.

    A prompt containing SLOW emits tokens slowly so cancellation can be tested.
    """
    self.model = object()  # pretend we loaded
    images = images or []
    last, markers = _describe(messages)
    yield {"type": "start", "thinking_open": False}

    if "SLOW" in last:
        for i in range(200):
            if cancel is not None and cancel.is_set():
                yield {"type": "cancelled"}
                return
            yield {"type": "token", "text": f"{i} "}
            time.sleep(0.02)
        yield {"type": "done", "text": "", "meta": {"model": "stub"}}
        return

    if thinking:
        # Deliberately split the tags across chunks: the splitter must cope.
        for piece in ["<th", "ink>let me consider", " that</think>"]:
            yield {"type": "token", "text": piece}

    # Reporting the counts lets a test prove images actually arrived, and that
    # the markers in the prompt line up with the paths handed to the engine.
    if images:
        for piece in [f"saw {len(images)} image(s), {markers} marker(s): ", last]:
            yield {"type": "token", "text": piece}
        reply = f"saw {len(images)} image(s), {markers} marker(s): {last}"
    else:
        for piece in ["you said: ", last]:
            yield {"type": "token", "text": piece}
        reply = f"you said: {last}"

    yield {
        "type": "done",
        "text": reply,
        "meta": {"model": "stub", "prompt_tokens": 10, "generation_tokens": 4,
                 "elapsed_s": 0.1, "tokens_per_second": 40.0,
                 "images": len(images), "markers": markers},
    }


def fake_image_stream(self, prompt, negative_prompt=None, width=None, height=None,
                      steps=None, guidance=None, seed=None, init_image=None,
                      image_strength=None, cancel=None, emit=None):
    self.model = object()
    total = steps or 4
    yield {"type": "status", "text": "loading image model"}
    # Progress goes through `emit`, matching the real engine.
    for i in range(1, total + 1):
        if cancel is not None and cancel.is_set():
            yield {"type": "cancelled"}
            return
        if emit is not None:
            emit({"type": "progress", "step": i, "total": total})
        else:
            yield {"type": "progress", "step": i, "total": total}

    if init_image is not None and not Path(init_image).is_file():
        yield {"type": "error", "error": f"base image missing: {init_image}"}
        return

    filename = "stub.png"
    self.image_dir.mkdir(parents=True, exist_ok=True)
    # A real PNG, so the file genuinely round-trips through the API.
    (self.image_dir / filename).write_bytes(SAMPLE_PNG)
    meta = {"model": "stub", "seed": seed or 7, "steps": total,
            "width": width or 8, "height": height or 8, "guidance": 4.0,
            "elapsed_s": 0.2}
    if init_image is not None:
        meta["from_image"] = Path(init_image).name
        meta["image_strength"] = image_strength
    yield {"type": "done", "image": filename, "meta": meta}

