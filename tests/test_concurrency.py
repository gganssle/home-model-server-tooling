"""Cancellation and queueing against a live (stubbed-model) server.

These need a real server process: the interesting behaviour is what happens
when a second request arrives while the single GPU worker is busy.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_web import FixtureWeb  # noqa: E402
from helpers import free_port  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
PORT = free_port()
BASE = f"http://127.0.0.1:{PORT}"

# The fake web lives in this process; the server under test is a subprocess and
# reaches it over loopback, which is why the private-address guard has to be
# waived for it.
WEB = FixtureWeb()

TMP = Path(tempfile.mkdtemp(prefix="hearth-conc-test-"))
ENV = {
    **os.environ,
    "HEARTH_DATA_DIR": str(TMP / "data"),
    "HEARTH_CONFIG_DIR": str(TMP / "config"),
    "HEARTH_PORT": str(PORT),
    "HEARTH_HOST": "127.0.0.1",
    "PYTHONPATH": str(ROOT / "tests"),
    "HEARTH_SEARCH": "1",
    "HEARTH_SEARCH_PROVIDER": "searxng",
    "HEARTH_SEARXNG_URL": WEB.base,
    "HEARTH_SEARCH_ALLOW_PRIVATE": "1",
    "HEARTH_SEARCH_TIMEOUT": "30",
}

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f"  {detail}" if not condition else ""))


def sse(resp):
    for line in resp.iter_lines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload != "[DONE]":
                yield json.loads(payload)


def main() -> int:
    WEB.__enter__()
    server = subprocess.Popen(
        [str(PYTHON), str(ROOT / "tests" / "stub_server.py")],
        env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    client = httpx.Client(base_url=BASE, timeout=60.0)
    try:
        for _ in range(100):
            try:
                if client.get("/api/status").status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            if server.poll() is not None:
                print("server died:", server.stderr.read()[:2000])
                return 1
            time.sleep(0.2)

        tid = client.post("/api/threads", json={"title": "t"}).json()["id"]

        # ---------------- cancellation ----------------
        print("\ncancellation")
        received = []

        def consume():
            with client.stream("POST", f"/api/threads/{tid}/messages",
                               json={"content": "SLOW please"}) as resp:
                for ev in sse(resp):
                    received.append(ev)

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()

        # Wait until tokens are actually flowing, then pull the plug.
        deadline = time.time() + 15
        while time.time() < deadline and len([e for e in received if e["type"] == "token"]) < 3:
            time.sleep(0.05)
        tokens_before = len([e for e in received if e["type"] == "token"])
        check("generation started", tokens_before >= 3, str(tokens_before))

        r = client.post("/api/cancel").json()
        check("cancel acknowledged", r["cancelled"] is True, str(r))

        worker.join(timeout=20)
        check("stream ended promptly after cancel", not worker.is_alive())
        tokens_after = len([e for e in received if e["type"] == "token"])
        check("generation stopped early", tokens_after < 200, str(tokens_after))
        check("cancelled turn still persisted",
              len(client.get(f"/api/threads/{tid}").json()["messages"]) == 2)

        # The server must be healthy for the next request after a cancel.
        print("\nserver recovers after cancel")
        with client.stream("POST", f"/api/threads/{tid}/messages",
                           json={"content": "hi"}) as resp:
            evs = list(sse(resp))
        text = "".join(e["text"] for e in evs if e["type"] == "token")
        check("next request works", text == "you said: hi", repr(text))

        # ---------------- queueing ----------------
        print("\nqueueing")
        results: dict[str, list] = {"a": [], "b": []}

        def run_one(key, content):
            with client.stream("POST", f"/api/threads/{tid}/messages",
                               json={"content": content}) as resp:
                for ev in sse(resp):
                    results[key].append(ev)

        ta = threading.Thread(target=run_one, args=("a", "SLOW one"), daemon=True)
        ta.start()
        time.sleep(0.6)  # let A take the worker
        tb = threading.Thread(target=run_one, args=("b", "second"), daemon=True)
        tb.start()
        time.sleep(0.6)
        check("second request is told it is queued",
              any(e["type"] == "queued" for e in results["b"]),
              str(results["b"][:2]))
        check("second request has not started generating",
              not any(e["type"] == "token" for e in results["b"]))

        client.post("/api/cancel")  # release the worker
        ta.join(timeout=20)
        tb.join(timeout=30)
        text_b = "".join(e["text"] for e in results["b"] if e["type"] == "token")
        check("queued request runs once the worker frees up",
              text_b == "you said: second", repr(text_b))

        # ---------------- cancelling a retrieval ----------------
        # Retrieval happens before a job exists, so manager.cancel_current()
        # cannot see it. This is the path that needs its own flag.
        print("\nsearch cancellation")
        WEB.set_results([{"url": WEB.url("/slow"), "title": "Slow page", "content": "x"}])
        sid = client.post("/api/threads", json={"title": "s"}).json()["id"]
        search_events: list = []

        def consume_search():
            with client.stream("POST", f"/api/threads/{sid}/messages",
                               json={"content": "/web something slow"}) as resp:
                for ev in sse(resp):
                    search_events.append(ev)

        searcher = threading.Thread(target=consume_search, daemon=True)
        searcher.start()

        deadline = time.time() + 15
        while time.time() < deadline and not any(
            e.get("phase") == "fetching" for e in search_events
        ):
            time.sleep(0.05)
        check("the fetch actually started",
              any(e.get("phase") == "fetching" for e in search_events), str(search_events))
        check("no answer has been generated yet",
              not any(e["type"] == "token" for e in search_events), str(search_events))

        started = time.time()
        r = client.post("/api/cancel").json()
        check("cancel sees the in-flight search", r["searches"] >= 1, str(r))

        searcher.join(timeout=20)
        elapsed = time.time() - started
        check("the stream ends promptly", not searcher.is_alive())
        check("it did not wait out the whole fetch", elapsed < 8, f"{elapsed:.1f}s")
        check("the turn reports itself cancelled",
              any(e["type"] == "done" and e.get("cancelled") for e in search_events),
              str([e["type"] for e in search_events]))
        check("the attempted search is still in the transcript",
              any(m["role"] == "tool"
                  for m in client.get(f"/api/threads/{sid}").json()["messages"]))

        # And the server is still healthy afterwards.
        with client.stream("POST", f"/api/threads/{sid}/messages",
                           json={"content": "hi", "search": False}) as resp:
            evs = list(sse(resp))
        check("the server recovers after a cancelled search",
              "".join(e["text"] for e in evs if e["type"] == "token") == "you said: hi")

        # ---------------- unload ----------------
        print("\nmemory controls")
        st = client.get("/api/status").json()
        check("status shows text model as loaded", st["text"]["loaded"] is True, str(st["text"]))
        freed = client.post("/api/models/unload", params={"which": "all"}).json()
        check("unload reports what it freed", "text" in freed["unloaded"], str(freed))
        st = client.get("/api/status").json()
        check("status reflects the unload", st["text"]["loaded"] is False)

    finally:
        client.close()
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
