"""The Datastar wire protocol, written out rather than pulled in.

Datastar is a hypermedia library: the browser holds almost no application
logic, and the server drives the page by sending it two kinds of patch over
an ordinary SSE stream. That protocol is small enough to write down in full,
which is the reason it is here instead of behind a dependency - the frames
below are the whole contract between `web/index.html` and `webui.py`.

    event: datastar-patch-elements
    data: mode inner
    data: selector #mlist
    data: elements <div class="msg">hello</div>

    event: datastar-patch-signals
    data: signals {"busy": false}

Elements are matched by `id` and *morphed* by default, so patching an element
keeps the DOM node - and with it focus, scroll position and any listeners -
instead of replacing it. That is what lets a token stream repaint a message
bubble sixty times a second without the page flickering.
"""
from __future__ import annotations

import json
from typing import Any

# Every mode the 1.0 client accepts. `outer` is the default and needs no
# selector: the incoming element's own id says where it goes.
MODES = ("remove", "outer", "inner", "replace", "prepend", "append", "before", "after")


def _data_lines(key: str, value: str) -> list[str]:
    """Emit one `data:` line per line of `value`, keyed by `key`.

    The client splits each data line at its *first* space and joins the pieces
    back together with newlines, so a key with an empty value still has to
    carry its trailing space - `data: elements` would be parsed as the key
    `element` with the value `s`. Hence the explicit space in the f-string.
    """
    return [f"data: {key} {line}" for line in value.split("\n")]


def patch_elements(
    html: str,
    *,
    selector: str | None = None,
    mode: str = "outer",
    use_view_transition: bool = False,
) -> str:
    """A `datastar-patch-elements` frame."""
    if mode not in MODES:
        raise ValueError(f"unknown patch mode {mode!r}")
    lines = ["event: datastar-patch-elements"]
    if selector:
        lines.append(f"data: selector {selector}")
    if mode != "outer":
        lines.append(f"data: mode {mode}")
    if use_view_transition:
        lines.append("data: useViewTransition true")
    if mode != "remove":
        lines += _data_lines("elements", html)
    return "\n".join(lines) + "\n\n"


def remove_elements(selector: str) -> str:
    """Delete whatever matches `selector`."""
    return patch_elements("", selector=selector, mode="remove")


def patch_signals(signals: dict[str, Any], *, only_if_missing: bool = False) -> str:
    """A `datastar-patch-signals` frame.

    The payload is merged into the client's signal store, so this only needs to
    carry what actually changed. A `None` value deletes the signal.
    """
    lines = ["event: datastar-patch-signals"]
    if only_if_missing:
        lines.append("data: onlyIfMissing true")
    lines += _data_lines("signals", json.dumps(signals, separators=(",", ":")))
    return "\n".join(lines) + "\n\n"


def read_signals(raw: str | None) -> dict[str, Any]:
    """Decode the signal bag a GET request carries in its `datastar` query param.

    Non-GET requests put the same JSON in the body, where FastAPI can bind it
    to a model directly; only the query-string form needs unpacking by hand.
    """
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
