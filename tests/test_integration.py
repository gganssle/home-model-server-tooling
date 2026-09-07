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

print("\nbase image sizing")
from PIL import Image  # noqa: E402

from hearth.engine.image import _source_size  # noqa: E402

big = TMP / "big.png"
Image.new("RGB", (3030, 2670), (10, 20, 30)).save(big)
bw, bh = _source_size(str(big), 1024 * 1024)
# A phone photo at full resolution asks mflux for a latent grid so large that
# attention over it exceeds the Metal buffer limit; it has to be scaled down.
check("an oversized base image is brought under the pixel budget",
      bw * bh <= 1024 * 1024, f"{bw}x{bh}")
check("its aspect ratio survives",
      abs((bw / bh) - (3030 / 2670)) < 0.02, f"{bw}x{bh}")
check("the scaled size sits on the multiple-of-16 grid",
      bw % 16 == 0 and bh % 16 == 0, f"{bw}x{bh}")

small = TMP / "small.png"
Image.new("RGB", (600, 400), (10, 20, 30)).save(small)
check("a small base image keeps its own size", _source_size(str(small), 1024 * 1024) == (592, 400),
      str(_source_size(str(small), 1024 * 1024)))

tiny = TMP / "tiny.png"
Image.new("RGB", (8, 8), (10, 20, 30)).save(tiny)
check("a tiny base image still yields a legal grid", _source_size(str(tiny), 1024 * 1024) == (16, 16),
      str(_source_size(str(tiny), 1024 * 1024)))

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

print("\nimage store is reported")
st = client.get("/api/status").json()
check("status reports the image folder", "images" in st and "dir" in st["images"],
      str(st.get("images")))
check("the reported folder is the one images are served from",
      st["images"]["dir"] == str(cfg.image_dir), str(st.get("images")))
check("file count is reported", isinstance(st["images"]["files"], int),
      str(st.get("images")))
check("size is reported", isinstance(st["images"]["bytes"], int), str(st.get("images")))
check("the count reflects images actually written", st["images"]["files"] > 0,
      str(st.get("images")))

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

# --------------------------------------------------------------------------
# Web search. A second app, because search config is read when the app is
# built. The provider is faked and every URL points at the fixture server, so
# none of this touches the real internet.
print("\nweb search")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_web import FakeProvider, FixtureWeb  # noqa: E402

web = FixtureWeb()
web.__enter__()
try:
    scfg = config_mod.load()
    scfg.search.enabled = True
    scfg.search.provider = "searxng"
    scfg.search.searxng_url = web.base
    scfg.search.allow_private_hosts = True
    scfg.search.max_results = 3
    scfg.search.max_fetch = 2
    scfg.search.max_context_chars = 4000
    scfg.search.max_history_documents = 1
    scfg.text.knowledge_cutoff = "mid 2024"

    sapp = create_app(scfg)
    sclient = TestClient(sapp)
    provider = FakeProvider([
        {"title": "MLX notes", "url": web.url("/article"), "snippet": "notes"},
        {"title": "Plain", "url": web.url("/plain"), "snippet": "plain"},
    ])
    sapp.state.websearch._provider = provider

    st = sclient.get("/api/status").json()
    check("status reports search enabled", st["search"]["enabled"] is True, str(st["search"]))
    check("status names the provider", st["search"]["provider"] == "searxng")

    # ---- the /web verb
    wt = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    r = sclient.post(f"/api/threads/{wt}/messages", json={"content": "/web mlx 0.32 release"})
    events = sse_events(r)
    phases = [e["phase"] for e in events if e["type"] == "search"]
    check("/web streams search phases",
          phases == ["querying", "results", "fetching", "ready"], str(phases))
    check("/web uses the given query", provider.queries[-1] == "mlx 0.32 release",
          str(provider.queries))
    ready = next(e for e in events if e.get("phase") == "ready")
    check("sources are reported to the client", len(ready["sources"]) == 2)
    check("sources carry a url", ready["sources"][0]["url"].endswith("/article"))

    answer = "".join(e["text"] for e in events
                     if e["type"] == "token" and e.get("channel") != "thinking")
    check("the model saw exactly one source block", "from 1 source block(s)" in answer, answer)
    done = [e for e in events if e["type"] == "done"][0]
    check("the assistant turn records its sources", len(done["meta"]["sources"]) == 2)

    msgs = sclient.get(f"/api/threads/{wt}").json()["messages"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    check("the search is stored as a tool message", len(tool_msgs) == 1)
    check("the stored content is the compact line",
          tool_msgs[0]["content"].startswith("[searched the web for"), tool_msgs[0]["content"])
    check("the compact line stays short", len(tool_msgs[0]["content"]) < 400)
    check("the full page text is on meta, not content",
          len(tool_msgs[0]["meta"]["search"]["documents"][0]["text"]) > 100)
    check("the page text is NOT in content",
          "unified memory" not in tool_msgs[0]["content"])
    check("bare /web is rejected",
          sclient.post(f"/api/threads/{wt}/messages",
                       json={"content": "/web"}).status_code == 400)

    # ---- a follow-up reuses the stored documents rather than fetching again
    before = len(provider.queries)
    r = sclient.post(f"/api/threads/{wt}/messages",
                     json={"content": "and what about performance?"})
    events = sse_events(r)
    answer = "".join(e["text"] for e in events
                     if e["type"] == "token" and e.get("channel") != "thinking")
    check("a follow-up triggers no new search", len(provider.queries) == before)

    # `/web off` in the REPL is sticky for the session, so the next explicit
    # `/web <query>` arrives with search=False. The verb has to win, or the
    # lookup the user just typed is silently dropped.
    before = len(provider.queries)
    stid = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    forced = sse_events(sclient.post(
        f"/api/threads/{stid}/messages",
        json={"content": "/web mlx release notes", "search": False},
    ))
    check("/web outranks a standing search=False",
          [e["phase"] for e in forced if e["type"] == "search"][:1] == ["querying"],
          str([e for e in forced if e["type"] == "search"][:2]))
    check("and it really did query the provider", len(provider.queries) > before)

    # The plain suppression it is meant to leave alone.
    before = len(provider.queries)
    sse_events(sclient.post(
        f"/api/threads/{stid}/messages",
        json={"content": "tell me about mlx", "search": False},
    ))
    check("search=False still suppresses an ordinary message",
          len(provider.queries) == before)
    check("the follow-up still sees the stored sources",
          "from 1 source block(s)" in answer, answer)

    # ---- only the newest search keeps its full text
    sclient.post(f"/api/threads/{wt}/messages", json={"content": "/web something else"})
    r = sclient.post(f"/api/threads/{wt}/messages", json={"content": "so what now?"})
    answer = "".join(e["text"] for e in sse_events(r)
                     if e["type"] == "token" and e.get("channel") != "thinking")
    check("older searches degrade to their compact line",
          "from 1 source block(s)" in answer, answer)

    # ---- the explicit per-message flag
    ft = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    before = len(provider.queries)
    r = sclient.post(f"/api/threads/{ft}/messages",
                     json={"content": "tell me about mlx", "search": True})
    check("search:true searches", len(provider.queries) == before + 1)
    check("search:true derives the query from the message",
          provider.queries[-1] == "tell me about mlx", str(provider.queries[-1]))
    answer = "".join(e["text"] for e in sse_events(r) if e["type"] == "token")
    check("search:true reaches the model", "source block(s)" in answer)

    before = len(provider.queries)
    sclient.post(f"/api/threads/{ft}/messages",
                 json={"content": "just answer from memory", "search": False})
    check("search:false does not search", len(provider.queries) == before)

    # ---- the heuristic tier
    scfg.search.autonomous = "heuristic"
    ht = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    before = len(provider.queries)
    sclient.post(f"/api/threads/{ht}/messages",
                 json={"content": "what is the latest version of mlx?"})
    check("the heuristic fires on time-sensitive wording",
          len(provider.queries) == before + 1)
    r = sclient.post(f"/api/threads/{ht}/messages",
                     json={"content": "what is the current price of an M3 Ultra?"})
    querying = next(e for e in sse_events(r) if e.get("phase") == "querying")
    check("the client is told why it searched",
          "changes over time" in (querying.get("reason") or ""), str(querying))

    before = len(provider.queries)
    sclient.post(f"/api/threads/{ht}/messages", json={"content": "write me a haiku"})
    check("the heuristic stays quiet otherwise", len(provider.queries) == before)
    before = len(provider.queries)
    sclient.post(f"/api/threads/{ht}/messages",
                 json={"content": "what is the latest mlx?", "search": False})
    check("search:false overrides the heuristic", len(provider.queries) == before)
    scfg.search.autonomous = "off"

    # ---- the model-initiated tier
    scfg.search.autonomous = "tool"
    scfg.search.max_rounds = 2
    tt = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    before = len(provider.queries)
    r = sclient.post(f"/api/threads/{tt}/messages", json={"content": "LOOKUP mlx please"})
    events = sse_events(r)
    answer = "".join(e["text"] for e in events
                     if e["type"] == "token" and e.get("channel") != "thinking")
    check("a tool call triggers a search", len(provider.queries) == before + 1)
    check("the model's own query is used", provider.queries[-1] == "mlx release",
          str(provider.queries[-1]))
    check("the tool call never reaches the client", "tool_call" not in answer, answer)
    check("the preamble before the call is still shown", "let me check." in answer, answer)
    check("the second round answers from the sources",
          "from 1 source block(s)" in answer, answer)
    tool_msgs = [m for m in sclient.get(f"/api/threads/{tt}").json()["messages"]
                 if m["role"] == "tool"]
    check("the model-initiated search is recorded too", len(tool_msgs) == 1)

    scfg.search.max_rounds = 0
    zt = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    before = len(provider.queries)
    r = sclient.post(f"/api/threads/{zt}/messages", json={"content": "LOOKUP mlx please"})
    check("max_rounds=0 refuses to act on the call", len(provider.queries) == before)
    check("the turn still completes",
          any(e["type"] == "done" for e in sse_events(r)))
    scfg.search.max_rounds = 2
    scfg.search.autonomous = "off"

    # A model that emits tool syntax when no tool was offered is just writing
    # text; stripping it would leave a hole in the reply.
    nt = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    r = sclient.post(f"/api/threads/{nt}/messages",
                     json={"content": '<tool_call>{"name": "web_search"}</tool_call>'})
    answer = "".join(e["text"] for e in sse_events(r) if e["type"] == "token")
    check("with no tool offered, tool syntax survives as text",
          "tool_call" in answer, repr(answer))

    # ---- a broken provider costs a note, not the turn
    from hearth.search.providers import SearchError  # noqa: E402
    sapp.state.websearch._provider = FakeProvider(error=SearchError("instance down"))
    et = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    events = sse_events(sclient.post(f"/api/threads/{et}/messages",
                                     json={"content": "/web anything"}))
    errors = [e for e in events if e["type"] == "search" and e.get("phase") == "error"]
    check("a provider failure is reported to the client", len(errors) == 1, str(events))
    check("the model still answers", any(e["type"] == "done" for e in events))
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    check("the answer is generated without sources", "you said:" in answer, answer)
    sapp.state.websearch._provider = provider

    # ---- cancel reports what it stopped
    body = sclient.post("/api/cancel").json()
    check("cancel counts in-flight searches", "searches" in body, str(body))

    # ---- strict fields
    check("an unknown search-ish field is still rejected",
          sclient.post(f"/api/threads/{wt}/messages",
                       json={"content": "x", "web": True}).status_code == 422)
    check("the search field is accepted",
          sclient.post(f"/api/threads/{wt}/messages",
                       json={"content": "x", "search": False}).status_code == 200)

    # ---- with search off, none of this happens
    off = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    events = sse_events(client.post(f"/api/threads/{off}/messages",
                                    json={"content": "/web anything", "search": True}))
    errs = [e for e in events if e["type"] == "search" and e.get("phase") == "error"]
    check("a server with search off says so", len(errs) == 1, str(events))
    check("and answers anyway", any(e["type"] == "done" for e in events))
finally:
    web.__exit__(None, None, None)


print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("failures:", FAILED)
    sys.exit(1)
