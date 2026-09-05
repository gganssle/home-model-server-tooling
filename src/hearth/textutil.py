"""Incremental parsing of the tagged blocks Qwen emits inline.

The model interleaves reasoning and tool calls with its actual answer.
Splitting them server-side means the CLI and the web UI agree on what is what,
without either of them reimplementing the parse.

Both splitters here share one rule: never emit text that might turn out to be
the first half of a tag. A tag split across two tokens - which happens
constantly - must not flash up as visible content and then have to be taken
back.
"""
from __future__ import annotations

import json

OPEN = "<think>"
CLOSE = "</think>"

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


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


class ToolCallSplitter:
    """Pull `<tool_call>{...}</tool_call>` blocks out of a token stream.

    Sits downstream of `ThinkSplitter`: reasoning is separated first, and only
    the content channel is scanned for calls. Visible text comes back out; the
    calls accumulate on `calls` for the caller to act on once the turn ends.
    """

    def __init__(self, enabled: bool = True) -> None:
        """`enabled=False` makes this a pass-through.

        If no tool was offered, a model that emits <tool_call> anyway is just
        writing text, and swallowing it would leave the user staring at a reply
        with a hole in it. Only strip what we asked for.
        """
        self.enabled = enabled
        self._buf = ""
        self._in_call = False
        self._current = ""
        self.calls: list[dict] = []
        self.visible: list[str] = []

    def feed(self, chunk: str) -> list[str]:
        if not self.enabled:
            self.visible.append(chunk)
            return [chunk] if chunk else []
        self._buf += chunk
        out: list[str] = []
        while True:
            tag = TOOL_CLOSE if self._in_call else TOOL_OPEN
            idx = self._buf.find(tag)
            if idx == -1:
                break
            before = self._buf[:idx]
            if self._in_call:
                self._current += before
                self._finish_call()
            elif before:
                out.append(before)
                self.visible.append(before)
            self._buf = self._buf[idx + len(tag):]
            self._in_call = not self._in_call

        tag = TOOL_CLOSE if self._in_call else TOOL_OPEN
        hold = _partial_suffix_len(self._buf, tag)
        flushable = self._buf[: len(self._buf) - hold]
        self._buf = self._buf[len(self._buf) - hold:]
        if flushable:
            if self._in_call:
                self._current += flushable
            else:
                out.append(flushable)
                self.visible.append(flushable)
        return [piece for piece in out if piece]

    def finish(self) -> list[str]:
        if not self.enabled:
            return []
        out: list[str] = []
        if self._buf:
            if self._in_call:
                self._current += self._buf
            else:
                out.append(self._buf)
                self.visible.append(self._buf)
            self._buf = ""
        if self._in_call:
            # An unterminated call means the model ran out of tokens mid-JSON.
            # Salvage it if it parses; otherwise drop it rather than showing
            # the user a half-written function call.
            self._finish_call()
            self._in_call = False
        return [piece for piece in out if piece]

    def _finish_call(self) -> None:
        raw, self._current = self._current.strip(), ""
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except ValueError:
            return
        if isinstance(payload, dict):
            self.calls.append(payload)

    @property
    def visible_text(self) -> str:
        return "".join(self.visible)


def tool_call_query(call: dict) -> str | None:
    """Pull the query out of a web_search call, whatever shape it arrived in.

    Models are inconsistent here: arguments come as a nested object on some
    turns and as a JSON string on others, and the name is sometimes nested
    under "function".
    """
    name = call.get("name")
    args = call.get("arguments")
    if isinstance(call.get("function"), dict):
        inner = call["function"]
        name = name or inner.get("name")
        args = args if args is not None else inner.get("arguments")
    if name != "web_search":
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return args.strip() or None
    if isinstance(args, dict):
        query = args.get("query") or args.get("q")
        return str(query).strip() if query else None
    return None
