"""The slash commands, in one place so the two frontends cannot disagree.

The REPL implements all of these in its own loop. The web UI implements a
subset - the ones marked `web` below - because the rest either need a terminal
(`/quit`), or duplicate something the page already has a control for
(`/threads` is the sidebar, `/attach` is the paperclip).

Three of them are not really frontend features at all: `/image`, `/edit` and
`/web <query>` are routed by `server.py` on the leading verb, so they work from
any client that can send a message, including `curl`. The `web` flag is about
which ones the browser *offers*, not which ones it is able to send.
"""
from __future__ import annotations

from typing import NamedTuple


class Slash(NamedTuple):
    name: str
    args: str          # how the arguments are shown in help, e.g. "<prompt>"
    help: str
    aliases: tuple[str, ...] = ()
    choices: tuple[str, ...] = ()   # fixed words, completed after the name
    # Offered by the web UI's composer menu as well as the REPL.
    web: bool = False

    @property
    def usage(self) -> str:
        return f"{self.name} {self.args}".strip()


SLASH_COMMANDS = [
    Slash("/new", "[title]", "start a new conversation"),
    Slash("/threads", "", "list conversations"),
    Slash("/switch", "<id|prefix|last>", "jump to another conversation", choices=("last",)),
    Slash("/image", "<prompt>", "generate an image in this conversation", web=True),
    Slash("/edit", "<prompt>", "redraw the newest image in this conversation", web=True),
    Slash("/attach", "<path>", "queue an image for the model to look at"),
    Slash("/detach", "", "drop queued attachments"),
    Slash("/think", "on|off", "toggle reasoning mode", choices=("on", "off"), web=True),
    Slash("/web", "on|off|<query>", "search the web now, or for every message",
          choices=("on", "off"), web=True),
    Slash("/retry", "", "re-run your last message"),
    Slash("/title", "<text>", "rename this conversation"),
    Slash("/show", "", "reprint the conversation so far"),
    Slash("/status", "", "what is loaded, memory use"),
    Slash("/unload", "[all|text|image]", "free memory", choices=("all", "text", "image")),
    Slash("/help", "", "this list"),
    Slash("/quit", "", "exit  (/exit, /q and Ctrl-D also work)", aliases=("/exit", "/q")),
]

WEB_COMMANDS = [c for c in SLASH_COMMANDS if c.web]

# The two that set a mode rather than saying something. Both frontends treat a
# bare verb as a toggle and an explicit word as an assignment.
ON_WORDS = ("on", "true", "1")
OFF_WORDS = ("off", "false", "0")
