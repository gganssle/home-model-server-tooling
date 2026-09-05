"""Exercise the CLI against a real (stubbed-model) server over HTTP.

This is the SSH path: a client process that never imports mlx, talking to a
daemon that holds the models.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_web import FixtureWeb  # noqa: E402
from helpers import free_port  # noqa: E402

# A fake web in this process; the stub server reaches it over loopback.
WEB = FixtureWeb()

ROOT = Path(__file__).resolve().parents[1]
HEARTH = ROOT / ".venv" / "bin" / "hearth"
PYTHON = ROOT / ".venv" / "bin" / "python"
PORT = str(free_port())

TMP = Path(tempfile.mkdtemp(prefix="hearth-cli-test-"))
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
    "NO_COLOR": "1",
}

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f"  {detail}" if not condition else ""))


# Generous: each call is a cold Python start, and the box may be busy.
def run(*args, stdin=None, timeout=240):
    # Always pass an explicit `input`: with input=None the child inherits our
    # stdin, and if that is an open pipe nobody closes, a command that reads
    # stdin blocks forever.
    return subprocess.run(
        [str(HEARTH), *args], env=ENV, capture_output=True, text=True,
        input=stdin if stdin is not None else "", timeout=timeout,
    )


def main() -> int:
    if not HEARTH.exists():
        print(f"missing {HEARTH}; run: uv pip install -e .")
        return 1

    print("\nserver not running")
    r = run("status")
    check("clear error when server is down",
          r.returncode != 0 and "cannot reach" in (r.stderr + r.stdout).lower(),
          repr((r.stderr + r.stdout)[:160]))

    print("\nstarting stub server")
    WEB.__enter__()
    WEB.set_results([
        {"url": WEB.url("/article"), "title": "MLX notes", "content": "release notes"},
    ])
    server = subprocess.Popen(
        [str(PYTHON), str(ROOT / "tests" / "stub_server.py")],
        env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    try:
        ready = False
        for _ in range(100):
            if run("status", timeout=60).returncode == 0:
                ready = True
                break
            if server.poll() is not None:
                print("server died:", server.stderr.read()[:2000])
                return 1
            time.sleep(0.2)
        check("server reachable", ready)
        if not ready:
            return 1

        print("\nstatus")
        r = run("status")
        check("status emits json when piped", '"memory"' in r.stdout, r.stdout[:160])
        check("status reports the search provider", '"searxng"' in r.stdout, r.stdout[:400])

        print("\nweb search")
        r = run("ask", "--web", "what changed in mlx")
        check("--web puts the answer on stdout",
              "source block(s)" in r.stdout, repr(r.stdout[:200]))
        check("--web keeps the sources off stdout",
              "http" not in r.stdout, repr(r.stdout[:200]))
        check("--web narrates on stderr",
              "searching the web" in r.stderr, repr(r.stderr[:300]))
        check("--web lists its sources on stderr",
              "/article" in r.stderr, repr(r.stderr[:400]))
        check("the page's own title wins over the provider's",
              "MLX release notes" in r.stderr, repr(r.stderr[:400]))
        r = run("ask", "--web", "--quiet", "what changed in mlx")
        check("--quiet silences the source list", "searching the web" not in r.stderr,
              repr(r.stderr[:200]))
        check("--quiet still answers", "source block(s)" in r.stdout, repr(r.stdout[:200]))
        r = run("ask", "--no-web", "what is the latest mlx")
        check("--no-web answers from memory",
              "you said:" in r.stdout, repr(r.stdout[:200]))
        check("--no-web does not search", "searching the web" not in r.stderr)
        r = run("ask", "--help")
        check("ask documents --web", "--web" in r.stdout, r.stdout[:600])

        print("\nask")
        r = run("ask", "hello there")
        check("ask returns the answer on stdout",
              r.stdout.strip() == "you said: hello there", repr(r.stdout[:160]))
        check("ask keeps stdout clean for pipes",
              r.stdout.count("\n") == 1, repr(r.stdout))

        print("\nask via stdin")
        r = run("ask", "summarize", stdin="some piped input")
        check("stdin is appended to the prompt",
              "some piped input" in r.stdout, repr(r.stdout[:200]))

        print("\nstdin that never closes")
        # An inherited pipe with no writer is what launchd/cron/CI hand you. A
        # blocking read here used to hang the command forever.
        rfd, wfd = os.pipe()
        try:
            t0 = time.time()
            done = subprocess.run(
                [str(HEARTH), "ask", "hello there"], env=ENV, stdin=rfd,
                capture_output=True, text=True, timeout=90,
            )
            check("does not hang on an idle stdin", True)
            check("still answers", "you said: hello there" in done.stdout,
                  repr(done.stdout[:160]))
            check("returns promptly", time.time() - t0 < 60, f"{time.time()-t0:.0f}s")
        except subprocess.TimeoutExpired:
            check("does not hang on an idle stdin", False, "timed out")
        finally:
            os.close(rfd)
            os.close(wfd)

        print("\nthreads")
        r = run("threads")
        lines = [ln for ln in r.stdout.strip().splitlines() if ln]
        check("threads are listed tab-separated", len(lines) >= 2 and "\t" in lines[0],
              repr(r.stdout[:200]))
        first_id = lines[0].split("\t")[0]
        check("thread ids look right", first_id.startswith("t_"), first_id)

        print("\nthread continuation")
        r = run("ask", "--thread", first_id, "again")
        check("continuing a thread works", "again" in r.stdout, repr(r.stdout[:160]))
        r = run("show", first_id)
        check("show prints the transcript",
              "[user]" in r.stdout and "[assistant]" in r.stdout, repr(r.stdout[:200]))

        print("\nprefix + last refs")
        r = run("show", first_id[:6])
        check("id prefix resolves", r.returncode == 0 and "[user]" in r.stdout)
        r = run("show", "last")
        check("'last' resolves", r.returncode == 0)

        print("\nimage")
        out = TMP / "out.png"
        r = run("image", "a red barn", "-o", str(out), "--steps", "3")
        check("image command succeeds", r.returncode == 0, r.stderr[:300])
        check("png written to disk", out.exists() and out.read_bytes()[:4] == b"\x89PNG")
        check("path printed to stdout", str(out) in r.stdout, repr(r.stdout[:200]))

        print("\nimage input")
        from test_integration_stubs import SAMPLE_PNG
        pic = TMP / "photo.png"
        pic.write_bytes(SAMPLE_PNG)
        r = run("ask", "-i", str(pic), "what is this?")
        check("ask -i sends the image", "saw 1 image(s)" in r.stdout, repr(r.stdout[:200]))
        r = run("ask", "-i", str(pic), "-i", str(pic), "compare these")
        check("ask -i is repeatable", "saw 2 image(s)" in r.stdout, repr(r.stdout[:200]))
        r = run("ask", "-i", str(TMP / "nope.png"), "hello")
        check("a missing attachment is reported", r.returncode != 0
              and "no such image" in (r.stderr + r.stdout).lower(),
              repr((r.stderr + r.stdout)[:200]))
        r = run("ask", "-i", str(pic))
        check("an image with no prompt is allowed", r.returncode == 0, r.stderr[:200])

        print("\nimg2img")
        base = TMP / "base.png"
        r = run("image", "a barn", "-o", str(base), "--steps", "2")
        check("base image generated", base.exists(), r.stderr[:200])
        out2 = TMP / "variation.png"
        r = run("image", "the same barn in winter", "--from", str(base),
                "--strength", "0.4", "-o", str(out2), "--steps", "2")
        check("--from succeeds", r.returncode == 0, r.stderr[:300])
        check("variation written", out2.exists() and out2.read_bytes()[:4] == b"\x89PNG")
        check("base image reported", "from " in r.stderr, repr(r.stderr[:200]))

        print("\nsearch")
        r = run("search", "hello")
        check("search finds the message", "hello" in r.stdout, repr(r.stdout[:200]))

        print("\nimage folder is surfaced")
        r = run("status")
        payload = json.loads(r.stdout)
        check("status reports the image folder", "images" in payload, r.stdout[:200])
        check("it names the server's directory",
              payload["images"]["dir"] == str(Path(ENV["HEARTH_DATA_DIR"]) / "images"),
              str(payload.get("images")))

        print("\nunload")
        r = run("unload", "all")
        check("unload runs", r.returncode == 0, r.stderr[:200])

        print("\nconfig")
        r = run("config", "--path")
        check("config path printed", "config.toml" in r.stdout, r.stdout[:200])
        check("config file created", Path(r.stdout.strip()).exists())

        print("\ndelete")
        r = run("rm", first_id, "-y")
        check("thread deleted", r.returncode == 0, r.stderr[:200])
        r = run("show", first_id)
        check("deleted thread is gone", r.returncode != 0)

        print("\nhelp surfaces every command")
        r = run("--help")
        r2 = run("ask", "--help")
        check("ask documents --image", "--image" in r2.stdout, r2.stdout[:400])
        r2 = run("image", "--help")
        check("image documents --from", "--from" in r2.stdout, r2.stdout[:400])
        check("image documents --strength", "--strength" in r2.stdout, r2.stdout[:400])
        for cmd in ("chat", "ask", "image", "threads", "serve", "pull", "status"):
            check(f"help lists {cmd}", cmd in r.stdout, r.stdout[:400])

        r2 = run("chat", "--help")
        for slash in ("/new", "/switch", "/attach", "/edit", "/think", "/unload", "/quit"):
            check(f"chat --help documents {slash}", slash in r2.stdout, r2.stdout[:600])
        check("chat --help keeps the argument hints",
              "[title]" in r2.stdout, r2.stdout[:600])

        print("\nserve announces where images are kept")
        banner_env = {**ENV, "HEARTH_PORT": str(free_port())}
        proc = subprocess.Popen(
            [str(HEARTH), "serve"], env=banner_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            # The folder line is followed by two explanatory lines, so keep
            # reading a little past the match rather than stopping on it.
            banner = ""
            deadline = time.time() + 90
            trailing = 0
            while time.time() < deadline and trailing < 3:
                line = proc.stdout.readline()
                if not line:
                    break
                banner += line
                if "images:" in banner:
                    trailing += 1
            check("startup names the image folder", "images:" in banner, repr(banner[:400]))
            check("startup says they are never deleted",
                  "kept indefinitely" in banner, repr(banner[:400]))
            check("startup shows the count", "files," in banner or "empty" in banner,
                  repr(banner[:400]))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    finally:
        WEB.__exit__(None, None, None)
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
