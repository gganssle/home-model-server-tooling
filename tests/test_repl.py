"""Drive the interactive REPL through a pty, the way SSH would.

prompt_toolkit needs a real terminal, so this allocates one rather than
piping, which is also the closest thing to how the command is actually used.
"""
from __future__ import annotations

import fcntl
import os
import pty
import re
import struct
import termios
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_web import FixtureWeb  # noqa: E402
from helpers import free_port  # noqa: E402

WEB = FixtureWeb()

ROOT = Path(__file__).resolve().parents[1]
HEARTH = ROOT / ".venv" / "bin" / "hearth"
PYTHON = ROOT / ".venv" / "bin" / "python"
PORT = str(free_port())

TMP = Path(tempfile.mkdtemp(prefix="hearth-repl-test-"))
ENV = {
    **os.environ,
    "HEARTH_DATA_DIR": str(TMP / "data"),
    "HEARTH_CONFIG_DIR": str(TMP / "config"),
    "HEARTH_PORT": PORT,
    "HEARTH_HOST": "127.0.0.1",
    "PYTHONPATH": str(ROOT / "tests"),
    "HEARTH_SEARCH": "1",
    "HEARTH_SEARCH_PROVIDER": "searxng",
    "HEARTH_SEARXNG_URL": WEB.base,
    "HEARTH_SEARCH_ALLOW_PRIVATE": "1",
    "TERM": "dumb",
    "COLUMNS": "100",
    "LINES": "40",
}

PASSED, FAILED = [], []
ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f"  {detail}" if not condition else ""))


class Repl:
    """A `hearth chat` process attached to a pseudo-terminal."""

    def __init__(self, term: str | None = None):
        self.master, slave = pty.openpty()
        # Give the pty a real size, or prompt_toolkit has no room to draw the
        # completion menu.
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 100, 0, 0))
        self.proc = subprocess.Popen(
            [str(HEARTH), "chat"],
            stdin=slave, stdout=slave, stderr=slave,
            env={**ENV, "TERM": term} if term else ENV, close_fds=True,
        )
        os.close(slave)
        self.buf = ""

    def read_until(self, needle: str, timeout: float = 30.0) -> bool:
        """Accumulate output until `needle` shows up (or we run out of time)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            import select

            r, _, _ = select.select([self.master], [], [], 0.2)
            if r:
                try:
                    chunk = os.read(self.master, 8192).decode("utf-8", "replace")
                except OSError:
                    break
                self.buf += ANSI.sub("", chunk)
                if needle in self.buf:
                    return True
            elif self.proc.poll() is not None:
                break
        return needle in self.buf

    def send(self, line: str) -> None:
        os.write(self.master, (line + "\n").encode())

    def type(self, text: str) -> None:
        """Type without pressing return, so the line stays open."""
        os.write(self.master, text.encode())

    def close(self) -> None:
        try:
            self.send("/quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        try:
            os.close(self.master)
        except OSError:
            pass


def main() -> int:
    WEB.__enter__()
    WEB.set_results([
        {"url": WEB.url("/article"), "title": "MLX notes", "content": "release notes"},
    ])
    server = subprocess.Popen(
        [str(PYTHON), str(ROOT / "tests" / "stub_server.py")],
        env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    repl = None
    try:
        for _ in range(100):
            probe = subprocess.run([str(HEARTH), "status"], env=ENV,
                                   capture_output=True, text=True, timeout=20)
            if probe.returncode == 0:
                break
            if server.poll() is not None:
                print("server died:", server.stderr.read()[:2000])
                return 1
            time.sleep(0.2)

        print("\nstartup")
        repl = Repl()
        check("banner shows a new conversation", repl.read_until("new conversation"))
        check("the banner names the search provider",
              repl.read_until("web search via searxng"), repr(repl.buf[:400]))
        check("prompt appears", repl.read_until("you"))

        print("\nchat turn")
        repl.send("hello there")
        check("answer streamed back", repl.read_until("you said: hello there"),
              repr(repl.buf[-300:]))
        check("stats line shown", repl.read_until("tok/s"), repr(repl.buf[-200:]))

        print("\nslash commands")
        repl.buf = ""
        repl.send("/help")
        check("/help lists commands", repl.read_until("/switch"), repr(repl.buf[-300:]))
        check("/help lists /attach", repl.read_until("/attach"), repr(repl.buf[-400:]))
        check("/help lists /edit", repl.read_until("/edit"), repr(repl.buf[-400:]))
        check("/help keeps the argument hints", repl.read_until("/new [title]"),
              repr(repl.buf[-400:]))

        repl.buf = ""
        repl.send("/think on")
        check("/think toggles on", repl.read_until("reasoning mode on"), repr(repl.buf[-200:]))

        repl.buf = ""
        repl.send("what is 2+2")
        check("reasoning is labelled", repl.read_until("thinking"), repr(repl.buf[-400:]))
        check("answer still arrives", repl.read_until("you said: what is 2+2"),
              repr(repl.buf[-300:]))

        repl.buf = ""
        repl.send("/title Arithmetic")
        check("/title renames", repl.read_until("renamed"), repr(repl.buf[-200:]))

        repl.buf = ""
        repl.send("/threads")
        check("/threads lists the thread", repl.read_until("Arithmetic"), repr(repl.buf[-300:]))

        print("\nattachments")
        from test_integration_stubs import SAMPLE_PNG
        pic = TMP / "photo.png"
        pic.parent.mkdir(parents=True, exist_ok=True)
        pic.write_bytes(SAMPLE_PNG)

        repl.buf = ""
        repl.send(f"/attach {pic}")
        check("/attach queues the image", repl.read_until("attached photo.png"),
              repr(repl.buf[-300:]))
        repl.buf = ""
        repl.send("/attach")
        check("/attach with no argument lists the queue",
              repl.read_until("queued for the next message"), repr(repl.buf[-300:]))
        repl.buf = ""
        repl.send("what is this?")
        check("the queued image goes with the next message",
              repl.read_until("saw 1 image(s)"), repr(repl.buf[-400:]))
        repl.buf = ""
        repl.send("and now?")
        check("the queue is emptied after sending",
              repl.read_until("saw 1 image(s)"), repr(repl.buf[-400:]))

        repl.buf = ""
        repl.send(f"/attach {pic}")
        repl.read_until("attached")
        repl.buf = ""
        repl.send("/detach")
        check("/detach clears the queue", repl.read_until("cleared 1 attachment"),
              repr(repl.buf[-200:]))

        repl.buf = ""
        repl.send(f"/attach {TMP}/missing.png")
        check("a bad path is reported", repl.read_until("no such file"), repr(repl.buf[-200:]))

        repl.buf = ""
        repl.send("/image a red barn")
        check("/image reports progress", repl.read_until("step"), repr(repl.buf[-300:]))
        check("/image reports the saved file", repl.read_until("image:"), repr(repl.buf[-300:]))

        print("\nediting an image")
        repl.buf = ""
        repl.send("/edit make it stormier")
        check("/edit reports progress", repl.read_until("step"), repr(repl.buf[-300:]))
        check("/edit names the base image", repl.read_until("from "), repr(repl.buf[-300:]))

        repl.buf = ""
        repl.send("/edit")
        check("bare /edit explains itself", repl.read_until("usage: /edit"),
              repr(repl.buf[-200:]))

        repl.buf = ""
        repl.send("/new")
        check("/new starts a conversation", repl.read_until("new conversation"),
              repr(repl.buf[-200:]))

        repl.buf = ""
        repl.send("/switch last")
        check("/switch moves threads", repl.read_until("switched to"), repr(repl.buf[-200:]))

        print("\nweb search")
        repl.buf = ""
        repl.send("/web")
        check("bare /web reports the current state",
              repl.read_until("web search:"), repr(repl.buf[-300:]))
        check("bare /web explains both forms",
              repl.read_until("/web <query>"), repr(repl.buf[-300:]))

        repl.buf = ""
        repl.send("/web what changed in mlx")
        check("/web narrates the search",
              repl.read_until("searching the web"), repr(repl.buf[-400:]))
        check("/web lists the sources", repl.read_until("MLX release notes"),
              repr(repl.buf[-600:]))
        check("/web answers from the sources",
              repl.read_until("source block(s)"), repr(repl.buf[-600:]))

        repl.buf = ""
        repl.send("/web on")
        check("/web on is acknowledged",
              repl.read_until("on for every message"), repr(repl.buf[-200:]))
        repl.buf = ""
        repl.send("anything at all")
        check("with /web on an ordinary message searches",
              repl.read_until("searching the web"), repr(repl.buf[-400:]))

        repl.buf = ""
        repl.send("/web off")
        check("/web off is acknowledged", repl.read_until("web search off"),
              repr(repl.buf[-200:]))
        repl.buf = ""
        repl.send("and now something else")
        repl.read_until("tok/s")
        # The reply still mentions a source block: the previous search's
        # documents are carried forward by max_history_documents. What must not
        # happen is a *new* lookup.
        check("with /web off an ordinary message does not search",
              "searching the web" not in repl.buf, repr(repl.buf[-400:]))

        repl.buf = ""
        repl.send("/status")
        check("/status reports memory", repl.read_until("memory"), repr(repl.buf[-300:]))

        repl.buf = ""
        repl.send("/nonsense")
        check("unknown command is reported", repl.read_until("unknown command"),
              repr(repl.buf[-200:]))

        print("\ncommand menu")
        # The menu is drawn by prompt_toolkit, which needs a terminal it can
        # address - the rest of these tests run under TERM=dumb.
        menu = Repl(term="xterm-256color")
        try:
            menu.read_until("new conversation")
            menu.buf = ""
            menu.type("/")
            check("typing / lists the commands", menu.read_until("/threads", timeout=10),
                  repr(menu.buf[-400:]))
            check("the menu describes each command",
                  menu.read_until("list conversations", timeout=10), repr(menu.buf[-400:]))
            menu.buf = ""
            menu.type("th")
            check("the list narrows as you type",
                  menu.read_until("toggle reasoning mode", timeout=10), repr(menu.buf[-400:]))
            menu.buf = ""
            menu.type("ink ")
            check("a command's argument words are offered too",
                  menu.read_until("off", timeout=10), repr(menu.buf[-400:]))
        finally:
            menu.type("\x03")   # abandon the half-typed line
            menu.close()

        print("\nexit")
        repl.buf = ""
        repl.send("/quit")
        check("quits cleanly", repl.read_until("bye", timeout=15), repr(repl.buf[-200:]))
        repl.proc.wait(timeout=10)
        check("exit status is zero", repl.proc.returncode == 0, str(repl.proc.returncode))
        repl = None

    finally:
        WEB.__exit__(None, None, None)
        if repl is not None:
            repl.close()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        shutil.rmtree(TMP, ignore_errors=True)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failures:", FAILED)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
