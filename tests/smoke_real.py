"""Smoke test against the actual models. Requires the weights and a GPU.

Not part of run_tests.sh - that suite stubs the engines so it stays fast and
portable. Run this once after `hearth pull` to confirm the real thing works:

    ./.venv/bin/python tests/smoke_real.py
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

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import free_port  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HEARTH = ROOT / ".venv" / "bin" / "hearth"
PORT = free_port()
BASE = f"http://127.0.0.1:{PORT}"

TMP = Path(tempfile.mkdtemp(prefix="hearth-smoke-"))
ENV = {
    **os.environ,
    "HEARTH_DATA_DIR": str(TMP / "data"),
    "HEARTH_CONFIG_DIR": str(TMP / "config"),
    "HEARTH_PORT": str(PORT),
    "HEARTH_HOST": "127.0.0.1",
}

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f"  {detail}" if not condition else ""),
          flush=True)


def sse(resp):
    for line in resp.iter_lines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload != "[DONE]":
                yield json.loads(payload)


def main() -> int:
    print(f"starting server on {BASE}", flush=True)
    server = subprocess.Popen(
        [str(HEARTH), "serve"], env=ENV,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Model loads are slow; the first request has to be allowed to take minutes.
    client = httpx.Client(base_url=BASE, timeout=1200.0)
    try:
        for _ in range(200):
            try:
                if client.get("/api/status").status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            if server.poll() is not None:
                print("server exited early")
                return 1
            time.sleep(0.5)

        cfg = client.get("/api/status").json()
        print(f"  text : {cfg['text']['repo']}")
        print(f"  image: {cfg['image']['repo']}\n", flush=True)

        tid = client.post("/api/threads", json={"title": "smoke"}).json()["id"]

        # ---------------- text ----------------
        print("text generation (first call loads ~38GB, be patient)", flush=True)
        t0 = time.time()
        with client.stream("POST", f"/api/threads/{tid}/messages", json={
            "content": "Reply with exactly the word: pineapple", "max_tokens": 32,
        }) as resp:
            events = list(sse(resp))
        load_and_gen = time.time() - t0
        done = [e for e in events if e["type"] == "done"]
        text = done[0]["content"] if done else ""
        print(f"  -> {text!r}  ({load_and_gen:.0f}s incl. load)", flush=True)
        check("text generation produced output", bool(text.strip()), repr(text))
        check("model followed the instruction", "pineapple" in text.lower(), repr(text))
        check("stats reported", done and done[0]["meta"].get("tokens_per_second", 0) > 0,
              str(done[0]["meta"] if done else None))

        st = client.get("/api/status").json()
        print(f"  memory after text load: {st['memory']['active_gb']} GB", flush=True)
        check("text model reports loaded", st["text"]["loaded"] is True)

        # ---------------- multi-turn ----------------
        print("\nconversation memory", flush=True)
        client.post(f"/api/threads/{tid}/messages",
                    json={"content": "My favourite number is 41.", "max_tokens": 64}).read()
        with client.stream("POST", f"/api/threads/{tid}/messages", json={
            "content": "What number did I just say? Answer with digits only.",
            "max_tokens": 32,
        }) as resp:
            events = list(sse(resp))
        answer = [e for e in events if e["type"] == "done"][0]["content"]
        print(f"  -> {answer!r}", flush=True)
        check("earlier turns are in context", "41" in answer, repr(answer))

        # ---------------- reasoning ----------------
        print("\nreasoning mode", flush=True)
        with client.stream("POST", f"/api/threads/{tid}/messages", json={
            "content": "A bat and ball cost 1.10 total. The bat costs 1.00 more "
                       "than the ball. How much is the ball?",
            "thinking": True, "max_tokens": 1200,
        }) as resp:
            events = list(sse(resp))
        thinking = "".join(e["text"] for e in events if e.get("channel") == "thinking")
        content = "".join(e["text"] for e in events if e.get("channel") == "content")
        print(f"  thinking: {len(thinking)} chars, answer: {len(content)} chars", flush=True)
        check("reasoning was produced", len(thinking) > 0, repr(thinking[:120]))
        check("answer is separate from reasoning", len(content) > 0, repr(content[:120]))
        check("reasoning is not leaking into the answer",
              "<think>" not in content and "</think>" not in content, repr(content[:200]))

        # ---------------- images ----------------
        print("\nimage generation (first call loads ~24GB)", flush=True)
        t0 = time.time()
        with client.stream("POST", "/api/images", json={
            "prompt": "a red barn in a snowy field, golden hour",
            "thread_id": tid, "steps": 8, "width": 512, "height": 512,
        }) as resp:
            events = list(sse(resp))
        img_secs = time.time() - t0
        progress = [e for e in events if e["type"] == "progress"]
        img_done = [e for e in events if e["type"] == "done"]
        errors = [e for e in events if e["type"] == "error"]
        if errors:
            print("  error:", errors[0]["error"][:400], flush=True)
        check("image progress streamed", len(progress) >= 4, str(len(progress)))
        check("image produced", bool(img_done), str(errors[:1]))
        if img_done:
            fn = img_done[0]["image"]
            print(f"  -> {fn}  ({img_secs:.0f}s incl. load)", flush=True)
            r = client.get(f"/api/images/{fn}")
            check("image served over http", r.status_code == 200 and r.content[:4] == b"\x89PNG")
            check("image is a plausible size", len(r.content) > 10_000, f"{len(r.content)} bytes")
            out = Path.cwd() / "smoke-image.png"
            out.write_bytes(r.content)
            print(f"  saved to {out}", flush=True)

        # ---------------- vision ----------------
        # Draw something unambiguous with the image model, then ask the text
        # model what it is. That exercises both models against each other and
        # proves the attachment actually reached the vision tower.
        print("\nvision: describing a generated image", flush=True)
        with client.stream("POST", "/api/images", json={
            "prompt": "a single ripe banana on a plain white background, centered, "
                      "product photo",
            "steps": 12, "width": 512, "height": 512,
        }) as resp:
            events = list(sse(resp))
        subject = [e for e in events if e["type"] == "done"]
        check("test subject generated", bool(subject), str(events[-1:]))
        if subject:
            name = subject[0]["image"]
            vt = client.post("/api/threads", json={"title": "vision"}).json()["id"]
            t0 = time.time()
            with client.stream("POST", f"/api/threads/{vt}/messages", json={
                "content": "What fruit is in this image? Reply with one word.",
                "images": [name], "max_tokens": 32,
            }) as resp:
                events = list(sse(resp))
            answer = [e for e in events if e["type"] == "done"][0]["content"]
            print(f"  -> {answer!r}  ({time.time() - t0:.0f}s)", flush=True)
            check("the model actually saw the image", "banana" in answer.lower(),
                  repr(answer))

            # A follow-up with no new attachment must still have the image in
            # context, which is the multi-turn marker placement working.
            with client.stream("POST", f"/api/threads/{vt}/messages", json={
                "content": "What colour is it? One word.", "max_tokens": 32,
            }) as resp:
                events = list(sse(resp))
            colour = [e for e in events if e["type"] == "done"][0]["content"]
            print(f"  -> {colour!r}", flush=True)
            check("the image is still in context on the next turn",
                  "yellow" in colour.lower(), repr(colour))

            stored = client.get(f"/api/threads/{vt}").json()["messages"][0]
            check("attachment recorded on the message",
                  (stored.get("meta") or {}).get("images") == [name], str(stored)[:200])

            # ---------------- img2img ----------------
            print("\nimg2img: varying an existing image", flush=True)
            t0 = time.time()
            with client.stream("POST", "/api/images", json={
                "prompt": "the same banana, but painted in thick oil paint impasto",
                "init_image": name, "image_strength": 0.55, "steps": 12,
            }) as resp:
                events = list(sse(resp))
            varied = [e for e in events if e["type"] == "done"]
            errs = [e for e in events if e["type"] == "error"]
            if errs:
                print("  error:", errs[0]["error"][:400], flush=True)
            check("variation produced", bool(varied), str(errs[:1]))
            if varied:
                meta = varied[0]["meta"]
                print(f"  -> {varied[0]['image']}  ({time.time() - t0:.0f}s)", flush=True)
                check("variation records its base", meta.get("from_image") == name,
                      str(meta))
                check("variation records its strength", meta.get("image_strength") == 0.55,
                      str(meta))
                check("variation matched the base dimensions",
                      meta.get("width") == 512 and meta.get("height") == 512, str(meta))
                r = client.get(f"/api/images/{varied[0]['image']}")
                check("variation is a real image",
                      r.status_code == 200 and len(r.content) > 10_000,
                      f"{len(r.content)} bytes")
                base_bytes = client.get(f"/api/images/{name}").content
                check("variation differs from its base", r.content != base_bytes)
                out = Path.cwd() / "smoke-img2img.png"
                out.write_bytes(r.content)
                Path(Path.cwd() / "smoke-base.png").write_bytes(base_bytes)
                print(f"  saved {out} and smoke-base.png", flush=True)

        st = client.get("/api/status").json()
        print(f"\n  memory with both models: {st['memory']['active_gb']} GB", flush=True)
        check("both models resident", st["text"]["loaded"] and st["image"]["loaded"],
              str({k: st[k]["loaded"] for k in ("text", "image")}))

        # ---------------- eviction ----------------
        print("\nunload", flush=True)
        client.post("/api/models/unload", params={"which": "all"})
        st = client.get("/api/status").json()
        check("memory released", not st["text"]["loaded"] and not st["image"]["loaded"])
        print(f"  memory after unload: {st['memory']['active_gb']} GB", flush=True)

    finally:
        client.close()
        server.terminate()
        try:
            server.wait(timeout=30)
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
