"""Incremental parsing of Qwen's <think> reasoning blocks.

The model emits reasoning inline. Splitting it server-side means the CLI and
the web UI agree on what is reasoning and what is the answer, without either
of them reimplementing the parse.
"""
from __future__ import annotations

OPEN = "<think>"
CLOSE = "</think>"


class ThinkSplitter:
    """Feed raw token text in, get (channel, text) pieces out.

    Holds back any trailing text that could be the start of a tag, so a tag
    split across two tokens is never emitted as visible content.
    """

    def __init__(self, start_in_think: bool = False) -> None:
        """`start_in_think` handles templates that pre-open the block.

        Qwen3.6's chat template ends the prompt with a bare `<think>`, so the
        model's very first token is already reasoning and the only tag we ever
        see is the closing one.
        """
        self._buf = ""
        self._in_think = start_in_think
        self.thinking: list[str] = []
        self.content: list[str] = []

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        self._buf += chunk
        out: list[tuple[str, str]] = []
        while True:
            tag = CLOSE if self._in_think else OPEN
            idx = self._buf.find(tag)
            if idx == -1:
                break
            before = self._buf[:idx]
            if before:
                out.append(self._emit(before))
            self._buf = self._buf[idx + len(tag):]
            self._in_think = not self._in_think

        # Keep back anything that might be a partial tag at the boundary.
        tag = CLOSE if self._in_think else OPEN
        hold = _partial_suffix_len(self._buf, tag)
        flushable = self._buf[: len(self._buf) - hold]
        self._buf = self._buf[len(self._buf) - hold:]
        if flushable:
            out.append(self._emit(flushable))
        return [piece for piece in out if piece[1]]

    def finish(self) -> list[tuple[str, str]]:
        out = []
        if self._buf:
            out.append(self._emit(self._buf))
            self._buf = ""
        return [piece for piece in out if piece[1]]

    def _emit(self, text: str) -> tuple[str, str]:
        if self._in_think:
            self.thinking.append(text)
            return ("thinking", text)
        self.content.append(text)
        return ("content", text)

    @property
    def content_text(self) -> str:
        return "".join(self.content)

    @property
    def thinking_text(self) -> str:
        return "".join(self.thinking)


def _partial_suffix_len(buf: str, tag: str) -> int:
    """Length of the longest suffix of `buf` that is a proper prefix of `tag`."""
    max_len = min(len(buf), len(tag) - 1)
    for n in range(max_len, 0, -1):
        if buf.endswith(tag[:n]):
            return n
    return 0
