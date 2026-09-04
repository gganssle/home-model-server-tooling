"""Drive the interactive REPL through a pty, the way SSH would.

prompt_toolkit needs a real terminal, so this allocates one rather than
piping, which is also the closest thing to how the command is actually used.
"""
from __future__ import annotations

import os
import pty
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import free_port  # noqa: E402

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

    def __init__(self):
        self.master, slave = pty.openpty()
        self.proc = subprocess.Popen(
            [str(HEARTH), "chat"],
            stdin=slave, stdout=slave, stderr=slave,
            env=ENV, close_fds=True,
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

        repl.buf = ""
        repl.send("/image a red barn")
        check("/image reports progress", repl.read_until("step"), repr(repl.buf[-300:]))
        check("/image reports the saved file", repl.read_until("image:"), repr(repl.buf[-300:]))

        repl.buf = ""
        repl.send("/new")
        check("/new starts a conversation", repl.read_until("new conversation"),
              repr(repl.buf[-200:]))

        repl.buf = ""
        repl.send("/switch last")
        check("/switch moves threads", repl.read_until("switched to"), repr(repl.buf[-200:]))

        repl.buf = ""
        repl.send("/status")
        check("/status reports memory", repl.read_until("memory"), repr(repl.buf[-300:]))

        repl.buf = ""
        repl.send("/nonsense")
        check("unknown command is reported", repl.read_until("unknown command"),
              repr(repl.buf[-200:]))

        print("\nexit")
        repl.buf = ""
        repl.send("/quit")
        check("quits cleanly", repl.read_until("bye", timeout=15), repr(repl.buf[-200:]))
        repl.proc.wait(timeout=10)
        check("exit status is zero", repl.proc.returncode == 0, str(repl.proc.returncode))
        repl = None

    finally:
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
