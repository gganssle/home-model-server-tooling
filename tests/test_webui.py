"""End-to-end tests of the Datastar frontend's routes.

The browser never sees JSON, so what is checked here is the wire format: which
`datastar-patch-*` frames a gesture produces, what markup they carry, and which
signals move. Models are stubbed exactly as in test_integration.py.
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="hearth-webui-test-"))
os.environ["HEARTH_DATA_DIR"] = str(TMP / "data")
os.environ["HEARTH_CONFIG_DIR"] = str(TMP / "config")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from hearth.engine.image import ImageEngine  # noqa: E402
from hearth.engine.text import TextEngine  # noqa: E402
from test_integration_stubs import (  # noqa: E402
    SAMPLE_PNG,
    fake_image_stream,
    fake_text_stream,
)

TextEngine.stream = fake_text_stream
ImageEngine.stream = fake_image_stream

from hearth import config as config_mod  # noqa: E402
from hearth.server import create_app  # noqa: E402

cfg = config_mod.load()
client = TestClient(create_app(cfg))

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def frames(resp) -> list[dict]:
    """Parse a Datastar SSE response into {event, selector, mode, elements...}.

    Deliberately a separate implementation from `hearth/datastar.py`: a test
    that reuses the encoder cannot catch the encoder being wrong.
    """
    out = []
    for block in resp.text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        parsed: dict[str, str] = {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                parsed["event"] = line[7:]
                continue
            assert line.startswith("data: "), line
            key, _, value = line[6:].partition(" ")
            parsed[key] = parsed[key] + "\n" + value if key in parsed else value
        out.append(parsed)
    return out


def elements(fs: list[dict]) -> str:
    return "".join(f.get("elements", "") for f in fs)


def signals(fs: list[dict]) -> str:
    return "".join(f.get("signals", "") for f in fs)


def send(**sig) -> list[dict]:
    body = {"thread": "", "draft": "", "search": "", "imgmode": False,
            "think": False, "strength": 0.6, "atts": []}
    body.update(sig)
    return frames(client.post("/ui/send", json=body))


# --------------------------------------------------------------------------
print("\nthe page and its runtime")
page = client.get("/")
check("index served", page.status_code == 200)
check("index loads the runtime from this server", '/datastar.js"' in page.text)
check("index carries no markdown renderer", "function md(" not in page.text)
runtime = client.get("/datastar.js")
check("runtime served locally", runtime.status_code == 200)
check("runtime is the vendored bundle", runtime.text.startswith("// Datastar v1."))
check("runtime is javascript",
      "javascript" in runtime.headers["content-type"], runtime.headers["content-type"])

# --------------------------------------------------------------------------
print("\nsidebar")
threads = client.get("/ui/threads")
check("thread list is html", threads.headers["content-type"].startswith("text/html"))
check("thread list is addressable by id", 'id="threads"' in threads.text)
check("empty list says so", "no conversations" in threads.text)

first = client.post("/api/threads", json={"title": "kitchen remodel"}).json()["id"]
client.post("/api/threads", json={"title": "engine notes"})
listed = client.get("/ui/threads").text
check("threads are listed", "kitchen remodel" in listed and "engine notes" in listed)

filtered = client.get("/ui/threads", params={"datastar": '{"search": "kitchen"}'}).text
check("search filters server-side", "kitchen remodel" in filtered)
check("search excludes non-matches", "engine notes" not in filtered)

active = client.get("/ui/threads", params={"datastar": f'{{"thread": "{first}"}}'}).text
check("the open thread is marked active", 'class="thread active"' in active)

status = frames(client.get("/ui/status"))
check("status patches the dot and the line",
      'id="dot"' in elements(status) and 'id="sysinfo"' in elements(status))
check("status reports search availability as a signal", '"searchOn"' in signals(status))
check("with no provider, search is off", '"searchOn":false' in signals(status))
check("and it says why", "no search provider" in signals(status)
      or "web search is off" in signals(status), signals(status))

# --------------------------------------------------------------------------
print("\nopening, clearing and deleting a thread")
opened = frames(client.get(f"/ui/threads/{first}"))
check("opening sets the thread signal", f'"thread":"{first}"' in signals(opened))
check("opening replaces the message list", 'id="mlist"' in elements(opened))
check("an empty thread shows the welcome panel", 'class="empty"' in elements(opened))

cleared = frames(client.post("/ui/new", json={"search": ""}))
check("new conversation clears the thread signal", '"thread":""' in signals(cleared))
check("new conversation empties the composer", '"atts":[]' in signals(cleared))

gone = frames(client.request("DELETE", f"/ui/threads/{first}",
                             params={"datastar": f'{{"thread": "{first}"}}'}))
check("deleting the open thread unsets it", '"thread":""' in signals(gone))
check("deleting redraws the sidebar", 'id="threads"' in elements(gone))
check("deleted thread is gone", "kitchen remodel" not in elements(gone))

# --------------------------------------------------------------------------
print("\nattachments")
data_uri = "data:image/png;base64," + base64.b64encode(SAMPLE_PNG).decode()
up = frames(client.post("/ui/attachments", json={
    "atts": [], "files": [{"name": "a.png", "mime": "image/png",
                           "contents": base64.b64encode(SAMPLE_PNG).decode()}]}))
check("upload clears the files signal", '"files":[]' in signals(up))
check("upload renders a thumbnail", 'class="thumb"' in elements(up))
stored = [line for line in signals(up).split('"atts":[')[1].split("]")[0].split(",")]
name = stored[0].strip('"')
check("upload stores the image under a name", (cfg.image_dir / name).exists(), name)

again = frames(client.post("/ui/attachments", json={"atts": [name], "files": []}))
check("a known name survives a round trip", f'"{name}"' in signals(again))

dropped = frames(client.post("/ui/attachments", json={"atts": [], "files": []}))
check("removing the last thumbnail empties the row",
      '<div id="attachments"></div>' in elements(dropped))

escaped = frames(client.post("/ui/attachments",
                             json={"atts": ["../../../../etc/passwd"], "files": []}))
check("a path outside the image directory is refused", '"atts":[]' in signals(escaped))
check("an unknown filename is refused",
      '"atts":[]' in signals(frames(client.post(
          "/ui/attachments", json={"atts": ["nope.png"], "files": []}))))

# --------------------------------------------------------------------------
print("\nthe command menu")
menu = frames(client.get("/ui/commands"))
menu_html, menu_sig = elements(menu), signals(menu)
check("the menu is addressable", 'id="slash"' in menu_html)
check("the names go into a signal for matching", '"_slash"' in menu_sig)
check("the menu offers the four browser commands",
      all(c in menu_sig for c in ("/image", "/edit", "/think", "/web")), menu_sig)
check("and nothing the browser cannot do",
      not any(c in menu_sig for c in ("/quit", "/threads", "/attach", "/retry")), menu_sig)
check("each row shows its usage", "/web on|off|&lt;query&gt;" in menu_html, menu_html)
check("each row shows what it does", "search the web now" in menu_html)
check("a row fills the composer when clicked", "$draft = '/image '" in menu_html)
check("rows hide themselves when they stop matching",
      '$_slashMatch.includes(&#39;/edit&#39;)' in menu_html
      or '$_slashMatch.includes(\'/edit\')' in menu_html, menu_html)

print("\nmode commands")
mode = frames(client.post("/ui/send", json={"draft": "/think on", "think": False}))
check("/think on sets the reasoning signal", '"think":true' in signals(mode))
check("/think on clears the composer", '"draft":""' in signals(mode))
check("/think on sends nothing to the model", not elements(mode), elements(mode))
check("/think off unsets it", '"think":false' in signals(
    frames(client.post("/ui/send", json={"draft": "/think off", "think": True}))))
check("a bare /think toggles", '"think":true' in signals(
    frames(client.post("/ui/send", json={"draft": "/think", "think": False}))))
check("a bare /think toggles the other way", '"think":false' in signals(
    frames(client.post("/ui/send", json={"draft": "/think", "think": True}))))
refused = frames(client.post("/ui/send", json={"draft": "/web on", "web": False}))
check("/web on says so when no provider is configured", 'class="err"' in elements(refused))
check("and does not flip a hidden control", '"web":true' not in signals(refused))

# --------------------------------------------------------------------------
print("\nsending a message")
check("an empty composer sends nothing", client.post("/ui/send", json={}).status_code == 204)

chat = send(draft="hello there")
html = elements(chat)
check("the user's turn is appended",
      any(f.get("mode") == "append" and "hello there" in f.get("elements", "") for f in chat))
check("a placeholder bubble is appended", 'id="gen"' in html)
check("the composer is cleared", '"draft":""' in signals(chat))
check("a thread is created on the fly", '"thread":"t_' in signals(chat))
check("the reply is streamed", "you said: hello there" in html)
check("the placeholder is replaced by the stored message",
      any(f.get("selector") == "#gen" and f.get("mode") == "replace" for f in chat))
check("the final bubble carries the stored id",
      'class="msg assistant" id="m_' in html)
check("the sidebar is refreshed at the end", 'id="threads"' in html)

thread_id = signals(chat).split('"thread":"')[1].split('"')[0]
reopened = elements(frames(client.get(f"/ui/threads/{thread_id}")))
check("the reply survives a reload", "you said: hello there" in reopened)
check("reload markup matches the stream", 'class="msg assistant" id="m_' in reopened)

reasoned = elements(send(thread=thread_id, draft="think about it", think=True))
check("reasoning is rendered in its own block", 'class="think"' in reasoned)
check("reasoning text is shown", "let me consider that" in reasoned)

# --------------------------------------------------------------------------
print("\nescaping on the way out")
nasty = elements(send(thread=thread_id, draft="<script>alert(1)</script>"))
check("a script tag never reaches the page", "<script>alert(1)" not in nasty)
check("it is shown escaped instead", "&lt;script&gt;alert(1)" in nasty)

# --------------------------------------------------------------------------
print("\ndrawing")
drawn = send(thread=thread_id, draft="a red barn", imgmode=True)
html = elements(drawn)
check("progress is reported", "<progress" in html)
check("the image is shown", 'class="gen"' in html)
check("the image can seed the next one", "Use as base" in html)
check("generation stats are shown", "seed" in html and "steps" in html)

edited = send(thread=thread_id, draft="make it stormy", imgmode=True,
              atts=[name], strength=0.35)
check("an edit reports what it started from", "from " in elements(edited))
check("an edit passes the strength through", "@ 0.35" in elements(edited))

# --------------------------------------------------------------------------
print("\nwhen a turn is refused")
refused = elements(send(thread=thread_id, draft="", imgmode=True, atts=[name]))
check("the refusal is shown in the conversation", 'class="err"' in refused)
check("the refusal explains itself", "usage: /edit" in refused)

# --------------------------------------------------------------------------
# Web search needs its own app: the search config is read when the app is
# built. The provider is faked and every URL points at a local fixture server,
# so none of this touches the real internet.
print("\nweb search")

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

    sapp = create_app(scfg)
    sclient = TestClient(sapp)
    sapp.state.websearch._provider = FakeProvider([
        {"title": "MLX notes", "url": web.url("/article"), "snippet": "notes"},
        {"title": "Plain", "url": web.url("/plain"), "snippet": "plain"},
    ])

    sstatus = signals(frames(sclient.get("/ui/status")))
    check("a configured provider turns the Web control on", '"searchOn":true' in sstatus)
    check("the control explains what it will do", "searxng" in sstatus, sstatus)

    st = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    searched = frames(sclient.post("/ui/send", json={
        "thread": st, "draft": "/web mlx release notes", "search": ""}))
    html, sig = elements(searched), signals(searched)

    check("the query is named while it runs", "searching the web for" in html)
    check("fetching is reported", "page(s)" in html)
    # Specifically the *live* strip: the final reconcile would show sources
    # from stored meta either way, so assert on the slot only streaming fills.
    check("the sources strip lands before the answer is written",
          any('id="gen-sources"' in f.get("elements", "") and "sources</summary>" in
              f.get("elements", "") for f in searched))
    check("sources are linked", 'rel="noopener noreferrer"' in html)
    check("a source title is shown", "MLX" in html)
    check("the retrieval is shown as its own dim line", 'class="msg tool"' in html)
    check("the transcript is reconciled once a search is involved",
          any(f.get("elements", "").startswith('<div id="mlist">') for f in searched))

    reloaded = elements(frames(sclient.get(f"/ui/threads/{st}")))
    check("the tool line survives a reload", 'class="msg tool"' in reloaded)
    check("the sources survive a reload", 'class="sources"' in reloaded)

    # The Web box forces a lookup on a message that would not have triggered one.
    st2 = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    forced = elements(frames(sclient.post("/ui/send", json={
        "thread": st2, "draft": "hello", "web": True, "search": ""})))
    check("the Web box forces a search", "searching the web for" in forced)

    on = frames(sclient.post("/ui/send", json={"draft": "/web on", "web": False}))
    check("/web on turns the mode on when a provider exists", '"web":true' in signals(on))
    check("/web on sends nothing to the model", not elements(on), elements(on))
    check("/web off turns it back off", '"web":false' in signals(
        frames(sclient.post("/ui/send", json={"draft": "/web off", "web": True}))))

    # The one that must not be mistaken for a mode word.
    stq = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    query = elements(frames(sclient.post("/ui/send", json={
        "thread": stq, "draft": "/web what changed in mlx", "search": ""})))
    check("/web <query> is still a search, not a mode",
          "searching the web for" in query, query[:400])
    check("and the query reaches the provider intact",
          "what changed in mlx" in query, query[:400])

    st3 = sclient.post("/api/threads", json={"title": "New conversation"}).json()["id"]
    unforced = elements(frames(sclient.post("/ui/send", json={
        "thread": st3, "draft": "hello", "web": False, "search": ""})))
    check("leaving it unticked does not force one", "searching the web for" not in unforced)
finally:
    web.__exit__(None, None, None)

print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
sys.exit(1 if FAILED else 0)
