"""Stub replacements for the two model engines, shared by the test scripts."""
from __future__ import annotations

import time


def fake_text_stream(self, messages, images=None, max_tokens=None, temperature=None,
                     top_p=None, thinking=None, cancel=None):
    """Echo the last user turn, wrapped in a think block when reasoning is on.

    A prompt containing SLOW emits tokens slowly so cancellation can be tested.
    """
    self.model = object()  # pretend we loaded
    last = messages[-1]["content"] if messages else ""
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
    for piece in ["you said: ", last]:
        yield {"type": "token", "text": piece}
    yield {
        "type": "done",
        "text": f"you said: {last}",
        "meta": {"model": "stub", "prompt_tokens": 10, "generation_tokens": 4,
                 "elapsed_s": 0.1, "tokens_per_second": 40.0},
    }


def fake_image_stream(self, prompt, negative_prompt=None, width=None, height=None,
                      steps=None, guidance=None, seed=None, cancel=None, emit=None):
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
    filename = "stub.png"
    self.image_dir.mkdir(parents=True, exist_ok=True)
    # A real 1x1 PNG, so the file genuinely round-trips through the API.
    (self.image_dir / filename).write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
        "00000049454e44ae426082"
    ))
    yield {"type": "done", "image": filename,
           "meta": {"model": "stub", "seed": seed or 7, "steps": total,
                    "width": width or 8, "height": height or 8, "guidance": 4.0,
                    "elapsed_s": 0.2}}
