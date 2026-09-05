"""End-to-end tests of the HTTP surface with the models stubbed out.

The real models are ~60GB and take a minute to load, so everything except the
two `*Engine.stream` methods is exercised for real here: routing, SSE framing,
the think-block split, thread persistence, and the OpenAI shim.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="hearth-test-"))
os.environ["HEARTH_DATA_DIR"] = str(TMP / "data")
os.environ["HEARTH_CONFIG_DIR"] = str(TMP / "config")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from hearth import config as config_mod  # noqa: E402
from hearth.engine.image import ImageEngine  # noqa: E402
from hearth.engine.text import TextEngine  # noqa: E402


from test_integration_stubs import fake_image_stream, fake_text_stream  # noqa: E402

TextEngine.stream = fake_text_stream
ImageEngine.stream = fake_image_stream

from hearth.server import create_app  # noqa: E402

cfg = config_mod.load()
client = TestClient(create_app(cfg))

PASSED = []
FAILED = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def sse_events(resp) -> list[dict]:
    out = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload != "[DONE]":
                out.append(json.loads(payload))
    return out


# --------------------------------------------------------------------------
print("\nstatus + threads")
st = client.get("/api/status").json()
check("status reports both models", "text" in st and "image" in st)
check("status reports memory", "memory" in st and "active_gb" in st["memory"])

t = client.post("/api/threads", json={"title": "New conversation"}).json()
tid = t["id"]
check("thread created", tid.startswith("t_"))
check("thread listed", any(x["id"] == tid for x in client.get("/api/threads").json()["threads"]))

print("\nchat streaming")
r = client.post(f"/api/threads/{tid}/messages", json={"content": "hello there"})
events = sse_events(r)
tokens = "".join(e["text"] for e in events if e["type"] == "token")
done = [e for e in events if e["type"] == "done"]
check("tokens streamed", tokens == "you said: hello there", repr(tokens))
check("done event carries content", done and done[0]["content"] == "you said: hello there")
check("done event carries stats", done and done[0]["meta"]["tokens_per_second"] == 40.0)

data = client.get(f"/api/threads/{tid}").json()
check("both turns persisted", len(data["messages"]) == 2, str(len(data["messages"])))
check("user turn stored", data["messages"][0]["content"] == "hello there")
check("assistant turn stored", data["messages"][1]["content"] == "you said: hello there")
check("thread auto-titled", data["thread"]["title"] == "hello there", data["thread"]["title"])

print("\nreasoning split")
r = client.post(f"/api/threads/{tid}/messages", json={"content": "think please", "thinking": True})
events = sse_events(r)
think_text = "".join(e["text"] for e in events if e.get("channel") == "thinking")
content_text = "".join(e["text"] for e in events if e.get("channel") == "content")
check("think block split out", think_text == "let me consider that", repr(think_text))
check("content excludes think block", content_text == "you said: think please", repr(content_text))
final = [e for e in events if e["type"] == "done"][0]
check("thinking saved to meta", final["meta"].get("thinking") == "let me consider that")

print("\nhistory")
msgs = client.get(f"/api/threads/{tid}").json()["messages"]
check("history accumulates", len(msgs) == 4, str(len(msgs)))

print("\nimage generation")
r = client.post("/api/images", json={"prompt": "a red barn", "thread_id": tid, "steps": 3})
events = sse_events(r)
progress = [e for e in events if e["type"] == "progress"]
img_done = [e for e in events if e["type"] == "done"]
check("progress events emitted", len(progress) == 3, str(len(progress)))
check("image done event", img_done and img_done[0]["image"] == "stub.png")
check("image url provided", img_done and img_done[0]["url"] == "/api/images/stub.png")
check("image meta has seed", img_done and "seed" in img_done[0]["meta"])
img = client.get("/api/images/stub.png")
check("image served", img.status_code == 200 and img.content[:4] == b"\x89PNG")
check("image attached to thread",
      any(m.get("image") == "stub.png" for m in client.get(f"/api/threads/{tid}").json()["messages"]))

print("\n/image chat shortcut")
r = client.post(f"/api/threads/{tid}/messages", json={"content": "/image a blue heron"})
events = sse_events(r)
check("chat shortcut routes to image", any(e["type"] == "progress" for e in events))
check("chat shortcut returns image",
      any(e["type"] == "done" and e.get("image") for e in events))

print("\npath traversal is refused")
bad = client.get("/api/images/..%2f..%2f..%2fetc%2fpasswd")
check("traversal blocked", bad.status_code == 404, str(bad.status_code))

print("\nthread refs")
short = client.get(f"/api/threads/{tid[:6]}")
check("prefix lookup works", short.status_code == 200 and short.json()["thread"]["id"] == tid)
last = client.get("/api/threads/last")
check("'last' resolves", last.status_code == 200)
check("unknown ref 404s", client.get("/api/threads/nope_nope").status_code == 404)

print("\nrename + search")
client.patch(f"/api/threads/{tid}", json={"title": "Barn talk"})
check("rename applied", client.get(f"/api/threads/{tid}").json()["thread"]["title"] == "Barn talk")
results = client.get("/api/search", params={"q": "heron"}).json()["results"]
check("search finds message", len(results) >= 1)

print("\nOpenAI-compatible endpoint")
oai = client.post("/v1/chat/completions", json={
    "model": "whatever", "messages": [{"role": "user", "content": "ping"}]}).json()
check("openai non-stream shape",
      oai["choices"][0]["message"]["content"] == "you said: ping",
      json.dumps(oai)[:200])
check("openai usage populated", oai["usage"]["total_tokens"] == 14)
r = client.post("/v1/chat/completions", json={
    "model": "x", "messages": [{"role": "user", "content": "ping"}], "stream": True})
chunks = sse_events(r)
streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
check("openai stream shape", streamed == "you said: ping", repr(streamed))
check("openai stream terminates", "[DONE]" in r.text)
check("models listed", len(client.get("/v1/models").json()["data"]) == 2)

print("\nimage input (vision)")
import base64  # noqa: E402
from test_integration_stubs import SAMPLE_PNG  # noqa: E402

DATA_URI = "data:image/png;base64," + base64.b64encode(SAMPLE_PNG).decode()

vt = client.post("/api/threads", json={"title": "New conversation"}).json()["id"]
r = client.post(f"/api/threads/{vt}/messages",
                json={"content": "what is this?", "images": [DATA_URI]})
events = sse_events(r)
final = [e for e in events if e["type"] == "done"][0]
check("engine received the image", final["meta"].get("images") == 1, str(final["meta"]))
check("image marker landed on the user turn", final["meta"].get("markers") == 1,
      str(final["meta"]))

msgs = client.get(f"/api/threads/{vt}").json()["messages"]
stored = (msgs[0].get("meta") or {}).get("images") or []
check("attachment recorded on the message", len(stored) == 1, str(msgs[0]))
check("attachment is served back", client.get(f"/api/images/{stored[0]}").status_code == 200)
check("attachment survives a reload",
      (client.get(f"/api/threads/{vt}").json()["messages"][0]["meta"]["images"] == stored))

print("\nimages persist across turns")
r = client.post(f"/api/threads/{vt}/messages", json={"content": "and its colour?"})
final = [e for e in sse_events(r) if e["type"] == "done"][0]
check("earlier image still reaches the model", final["meta"].get("images") == 1,
      str(final["meta"]))
check("marker stayed on the original turn", final["meta"].get("markers") == 1,
      str(final["meta"]))

print("\nmultiple images in one turn")
r = client.post(f"/api/threads/{vt}/messages",
                json={"content": "compare these", "images": [DATA_URI, DATA_URI]})
final = [e for e in sse_events(r) if e["type"] == "done"][0]
# 1 carried from the first turn + 2 new, and markers must match paths exactly.
check("all images reach the model", final["meta"].get("images") == 3, str(final["meta"]))
check("markers match image count",
      final["meta"].get("markers") == final["meta"].get("images"), str(final["meta"]))

print("\nhistory image budget")
# max_history_images defaults to 4; a fifth attachment pushes the oldest out.
for i in range(3):
    client.post(f"/api/threads/{vt}/messages",
                json={"content": f"another {i}", "images": [DATA_URI]}).read()
r = client.post(f"/api/threads/{vt}/messages", json={"content": "and now?"})
final = [e for e in sse_events(r) if e["type"] == "done"][0]
check("image count is capped", final["meta"].get("images") == 4, str(final["meta"]))
check("markers still match after trimming",
      final["meta"].get("markers") == final["meta"].get("images"), str(final["meta"]))

print("\nattachment validation")
check("an image-only message is allowed",
      client.post(f"/api/threads/{vt}/messages",
                  json={"content": "", "images": [DATA_URI]}).status_code == 200)
check("non-image bytes are refused",
      client.post(f"/api/threads/{vt}/messages", json={
          "content": "x",
          "images": ["data:image/png;base64," + base64.b64encode(b"not a png").decode()],
      }).status_code == 400)
check("missing file is refused",
      client.post(f"/api/threads/{vt}/messages",
                  json={"content": "x", "images": ["/nope/missing.png"]}).status_code == 400)

print("\nimg2img")
et = client.post("/api/threads", json={"title": "New conversation"}).json()["id"]
r = client.post("/api/images", json={"prompt": "a barn", "thread_id": et, "steps": 2})
base_name = [e for e in sse_events(r) if e["type"] == "done"][0]["image"]
r = client.post("/api/images", json={
    "prompt": "the same barn in winter", "thread_id": et, "steps": 2,
    "init_image": base_name, "image_strength": 0.4,
})
final = [e for e in sse_events(r) if e["type"] == "done"][0]
check("base image passed through", final["meta"].get("from_image") == base_name,
      str(final["meta"]))
check("strength passed through", final["meta"].get("image_strength") == 0.4,
      str(final["meta"]))
check("strength is range-checked",
      client.post("/api/images", json={"prompt": "x", "image_strength": 3.0}).status_code == 422)

print("\n/edit chat shortcut")
r = client.post(f"/api/threads/{et}/messages", json={"content": "/edit make it stormier"})
final = [e for e in sse_events(r) if e["type"] == "done"][0]
check("/edit uses the newest image in the thread",
      final["meta"].get("from_image") is not None, str(final["meta"]))
empty = client.post("/api/threads", json={"title": "New conversation"}).json()["id"]
check("/edit with nothing to edit is refused",
      client.post(f"/api/threads/{empty}/messages",
                  json={"content": "/edit something"}).status_code == 400)
check("bare /edit is refused",
      client.post(f"/api/threads/{et}/messages", json={"content": "/edit"}).status_code == 400)

print("\nOpenAI multimodal content")
oai = client.post("/v1/chat/completions", json={"model": "x", "messages": [
    {"role": "user", "content": [
        {"type": "text", "text": "describe it"},
        {"type": "image_url", "image_url": {"url": DATA_URI}},
    ]}]}).json()
check("openai image_url reaches the model",
      "1 image(s)" in oai["choices"][0]["message"]["content"],
      json.dumps(oai)[:250])
check("openai text-only still works",
      "you said: hi" in client.post("/v1/chat/completions", json={
          "model": "x", "messages": [{"role": "user", "content": "hi"}]},
      ).json()["choices"][0]["message"]["content"])

print("\nstrict request fields")
check("unknown chat field is rejected",
      client.post(f"/api/threads/{tid}/messages",
                  json={"content": "x", "think": True}).status_code == 422)
check("known chat field is accepted",
      client.post(f"/api/threads/{tid}/messages",
                  json={"content": "x", "thinking": True}).status_code == 200)
check("unknown image field is rejected",
      client.post("/api/images", json={"prompt": "x", "step": 3}).status_code == 422)
check("openai shim stays permissive",
      client.post("/v1/chat/completions",
                  json={"model": "x", "messages": [{"role": "user", "content": "p"}],
                        "top_p": 0.9, "user": "someone"}).status_code == 200)

print("\nvalidation + cleanup")
check("empty message rejected",
      client.post(f"/api/threads/{tid}/messages", json={"content": "  "}).status_code == 400)
check("bare /image rejected",
      client.post(f"/api/threads/{tid}/messages", json={"content": "/image"}).status_code == 400)
check("web UI served", client.get("/").status_code == 200 and "hearth" in client.get("/").text)
check("thread deleted", client.delete(f"/api/threads/{tid}").status_code == 200)
check("deleted thread gone", client.get(f"/api/threads/{tid}").status_code == 404)

print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("failures:", FAILED)
    sys.exit(1)
