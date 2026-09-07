# hearth

A small, self-hosted chat app for your own models. Chat, image generation,
image understanding, conversation threads, a web UI, and a CLI that works well
over SSH.

Built for Apple Silicon: text runs on [mlx-vlm](https://github.com/Blaizzy/mlx-vlm),
images on [mflux](https://github.com/filipstrand/mflux), both MLX-native.

| | |
|---|---|
| Text model | [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) via `mlx-community/Qwen3.6-35B-A3B-8bit` |
| Image model | [`Qwen/Qwen-Image-2512`](https://huggingface.co/Qwen/Qwen-Image-2512) via `mlx-community/Qwen-Image-2512-4bit` |

## How it is put together

One daemon owns the models; everything else is a thin client over HTTP.

```
   SSH session                  browser on the Mac
   $ hearth chat                http://127.0.0.1:8080
        |                              |
        +---------- HTTP / SSE --------+
                       |
                 hearth serve
                       |
        +--------------+--------------+
        |              |              |
   SQLite threads   text model    image model
                    (mlx-vlm)     (mflux)
```

That split is the whole point. Your SSH session starts instantly because it
never loads a 38 GB model — the daemon already has it resident. And because
both frontends share one SQLite store, a conversation you start in the browser
can be picked up from the terminal, and the other way round.

All GPU work runs on a single worker thread. Concurrent requests queue instead
of colliding, which is what you want when two large models are competing for
the same unified memory.

## Install

```bash
uv venv --python 3.12
uv pip install -e .
```

Download the weights (~62 GB, once):

```bash
hearth pull            # or: hearth pull text  /  hearth pull image
```

## Run

```bash
hearth serve                 # foreground
hearth serve --preload       # load the text model at startup
```

Then open <http://127.0.0.1:8080>.

To keep it running whenever the Mac is on — which is what makes the SSH case
work — install it as a launch agent:

```bash
hearth install-service
launchctl load -w ~/Library/LaunchAgents/com.hearth.server.plist
```

## Command line

```bash
hearth chat                  # interactive REPL; the one to use over SSH
hearth chat last             # resume the most recent conversation
hearth chat t_9f2            # resume by id (a unique prefix is enough)

hearth ask "explain kubelet in two sentences"
hearth ask --thread last "and how does that differ from kube-proxy?"

hearth ask -i screenshot.png "what is this error telling me?"
hearth ask -i before.png -i after.png "what changed?"

hearth image "a lighthouse in fog, 35mm" -o lighthouse.png --open
hearth image "the same lighthouse at night" --from lighthouse.png --strength 0.5

hearth threads               # list conversations
hearth show last             # print a transcript
hearth search kubelet
hearth rm t_9f2

hearth status                # what is loaded, memory in use
hearth unload all            # hand memory back without stopping the server
```

It behaves properly in a pipeline — when stdout is not a terminal you get
plain text and nothing else, and stdin is folded into the prompt:

```bash
cat crash.log | hearth ask "what is failing here?" > diagnosis.txt
hearth ask -q "one word: is this valid JSON? $(cat x.json)"
git diff | hearth ask "write a commit message"
```

### In the REPL

Type `/` at the prompt and the command list opens under it, filtering as you
keep typing — pick one with the arrow keys or Tab.

```
/new [title]               start a new conversation
/threads                   list conversations
/switch <id|prefix|last>   jump to another conversation
/image <prompt>            generate an image in this conversation
/edit <prompt>             redraw the newest image in this conversation
/attach <path>             queue an image for the model to look at
/detach                    drop queued attachments
/think on|off              toggle reasoning mode
/web on|off|<query>        search the web now, or for every message
/retry                     re-run your last message
/title <text>              rename this conversation
/show                      reprint the conversation so far
/status                    what is loaded, memory use
/unload [all|text|image]   free memory
/help                      this list
/quit                      exit  (/exit, /q and Ctrl-D also work)
```

Ctrl-C stops a running generation without leaving the chat. The partial answer
is kept.

## Images

### Generating

Three ways in, all landing in the same place:

- `hearth image "a red barn at dusk"`
- `/image a red barn at dusk` — inside a chat, in the REPL or the web UI
- The **Image mode** checkbox in the web UI

### Understanding

The text model is a vision model, so it can look at images as well as write
about them:

```bash
hearth ask -i chart.png "what is the trend here?"
hearth ask -i a.jpg -i b.jpg "which of these is sharper?"
```

In the REPL, `/attach path/to/image.png` queues an image for your next message.
In the web UI, drag an image onto the page, paste one from the clipboard, or use
the paperclip.

Attachments stay in context for follow-up questions — ask "and what colour is
it?" without re-attaching. The most recent few are carried forward
(`max_history_images`, default 4); older ones become a note in the transcript,
because the vision tower re-encodes every image on every turn.

### Editing

Give a generation a starting image and it produces a variation rather than
starting from noise:

```bash
hearth image "the same barn in winter" --from barn.png --strength 0.5
```

`--strength` is how far it may move from the original, 0 to 1. Low values stay
close; high values barely resemble it.

The result keeps the base image's shape, but is scaled to fit the pixel budget
of the configured `width` x `height` (1024x1024 by default). A camera-sized
photo redrawn at full resolution asks for more memory than Metal will hand out
in one buffer, and the generation fails before the first step. Pass `--width`
and `--height` to override.

In a conversation, `/edit make the sky stormier` works on the newest image in
that thread — in the REPL and the web UI both. In the web UI, **Use as base**
under any generated image loads it into the composer with a strength slider.

Generated and attached images live in `~/.local/share/hearth/images/` and are
attached to the conversation they belong to. Attachments are copied in rather
than referenced, so a conversation still renders after you move or delete the
original.

## Web search

Off by default. Turn it on and the assistant can read live pages instead of
guessing at anything that happened after its training data ends.

### Pointing it at a provider

**SearXNG** is the default, because you can run it yourself and the query never
leaves your hardware — the same reason the models are local:

```bash
docker run -d -p 8888:8080 -v "$PWD/searxng:/etc/searxng" searxng/searxng
```

Add `json` to the `formats` list in its `settings.yml`, or it will serve HTML
only and hearth will tell you so. Then:

```toml
[search]
enabled = true
provider = "searxng"
searxng_url = "http://127.0.0.1:8888"
```

If your config predates this feature it has no `[search]` section at all — run
`hearth config --sync` first, then edit. `hearth status` shows whether the
server can search, and `hearth chat` says so in its banner and refuses to
pretend otherwise when you type `/web on`.

**Tavily** is the easiest if you would rather not host anything. It is built
for this use rather than being a general web index, and it returns the
extracted page text alongside each hit — so hearth does no fetching at all,
which is both faster and the only thing that works on the large number of
sites that refuse anything not shaped like a browser.

```toml
[search]
enabled = true
provider = "tavily"
tavily_api_key = "tvly-..."
tavily_depth = "basic"     # "advanced" digs harder and costs two credits
```

hearth never asks Tavily for its `answer` field. That has a cloud model write
the reply, which is exactly what this project exists not to do — only the
search results come back, and your local model does the answering.

**Brave** is also supported — `provider = "brave"` and a `brave_api_key` (or
`HEARTH_BRAVE_KEY`).

`hearth status` shows which one is live, or why none is.

Page text is extracted with a small stdlib HTML parser. If `trafilatura` is
installed it is used instead, which is noticeably better at finding the article
in a page full of furniture:

```bash
uv pip install -e '.[search]'
```

### Using it

```bash
hearth ask --web "what changed in the latest mlx release?"
hearth ask --no-web "explain unified memory"      # never search this one
```

In the REPL:

```
/web what changed in mlx 0.32     search for exactly that, then answer
/web on                           search on every message from here
/web off                          stop
/web                              say which of those is in force
```

In the browser, tick **Web** in the composer toolbar. It only appears when the
server actually has a provider.

Sources are listed under the answer, and go to stderr from the CLI so they
survive a pipe — `--quiet` is what silences them.

### Deciding when to search, without being asked

`search.autonomous` picks the policy:

| | |
|---|---|
| `"off"` | search only when explicitly asked. The default. |
| `"heuristic"` | a cheap pre-filter decides before generating: temporal wording, a recent year, a URL, "look it up". Crude, but it costs nothing and it is inspectable. |
| `"tool"` | offer the model a `web_search` tool and let it decide. |

`"tool"` is the one everyone expects and the one to be careful with: a ~30B
local model is materially worse calibrated about *when* to call a tool than a
frontier model is. Searching when you did not need to is not free — it costs
seconds, and a mediocre SEO-spam page in the context will happily displace
knowledge the model already had. Start with `"heuristic"`, which is also the
baseline `"tool"` has to beat before it is worth the extra generation round.

Either way, tell the model when its knowledge ends:

```toml
[models.text]
knowledge_cutoff = "mid 2024"
```

That, plus today's date, goes into the system prompt. It is the cheapest change
in the whole feature and does most of the work: it is what lets the model say
"that is after my time" instead of inventing an answer with the same
confidence it uses for everything else.

### What it will not do

Fetching refuses anything that is not `http`/`https`, and refuses to resolve to
a loopback, private, link-local or reserved address — then pins the connection
to the address it checked, so a second DNS answer cannot redirect it, and
re-checks every redirect hop. This matters more here than in a hosted product:
the daemon may be bound to your LAN, and under `autonomous = "tool"` the URL
originates with the model. Set `allow_private_hosts = true` if you genuinely
want it reading an intranet wiki.

Retrieved pages arrive wrapped in `<source>` tags with a standing instruction
that everything inside them is untrusted data and never an instruction, and a
page cannot close its own wrapper. Treat that as a speed bump rather than a
guarantee — prompt injection through retrieved content is not a solved problem
anywhere.

Only the most recent search keeps its full page text in the prompt
(`max_history_documents`); older ones collapse to a one-line source list. A
single web page is bigger than most whole conversations, so without that a
thread runs out of context in about three turns.

## Reasoning mode

Qwen3.6 is a hybrid reasoning model. Thinking is **off** by default so chat
feels quick. Turn it on per request with `hearth ask --think`, `/think on` in
the REPL, or the **Reasoning** checkbox in the web UI.

The server separates reasoning from the answer before it reaches either
frontend, so the CLI dims it and the web UI puts it in a collapsible block. It
is stored on the message, so you can go back and read it later.

## Configuration

`~/.config/hearth/config.toml`, created on first run. `hearth config --edit`
opens it.

The file is written once and never rewritten, so one created before a feature
existed simply will not mention it — `hearth config` lists anything you are
running on defaults, and `hearth config --sync` writes those settings in
without touching the values you already set:

```bash
hearth config --sync
```

```toml
[server]
host = "127.0.0.1"     # 0.0.0.0 to reach it from elsewhere on your LAN
port = 8080

[models.text]
repo = "mlx-community/Qwen3.6-35B-A3B-8bit"
max_tokens = 4096
temperature = 0.7
max_history_messages = 40
max_history_images = 4     # attached images carried forward between turns
enable_thinking = false
system_prompt = "You are a helpful assistant running locally on the user's own machine."
knowledge_cutoff = ""      # e.g. "mid 2024"; goes in the prompt beside today's date

[models.image]
repo = "mlx-community/Qwen-Image-2512-4bit"
steps = 20
width = 1024
height = 1024
guidance = 4.0
image_strength = 0.6       # default for --from / /edit, 0-1

[search]
enabled = false            # off until you point it at a provider
provider = "searxng"       # searxng | tavily | brave | none
searxng_url = "http://127.0.0.1:8888"
tavily_api_key = ""
tavily_depth = "basic"     # basic | advanced
brave_api_key = ""
max_results = 5            # results asked of the provider
max_fetch = 3              # of those, how many pages are read (Tavily supplies them)
max_context_chars = 6000   # hard ceiling on retrieved text in the prompt
max_history_documents = 1  # how many past searches keep their full page text
autonomous = "off"         # off | heuristic | tool
max_rounds = 2             # cap on model-initiated search rounds per turn
timeout_s = 10.0
allow_private_hosts = false  # true to allow an intranet wiki or a LAN SearXNG

[memory]
idle_evict_seconds = 900   # give memory back after 15 min idle; 0 to never
exclusive = false          # true = only one model resident at a time
```

Any of it can be overridden per-invocation by environment variable, which is
handy over SSH:

```bash
HEARTH_TEXT_MODEL=mlx-community/Qwen3.6-35B-A3B-4bit hearth serve
HEARTH_HOST=192.168.1.50 hearth ask "hello"      # talk to another box
```

### Memory

Measured on an M3 Ultra with 96 GB:

| | |
|---|---|
| Text model resident | 37.7 GB |
| Both models resident | 63.5 GB |
| Text load + first token | ~8 s warm, ~18 s cold |
| Image, 512x512 / 8 steps | ~34 s including the 24 GB load |
| After `hearth unload all` | 0 GB |

Both models fit together on this machine, though loading the second one while
the first is resident does push the system into some swap. If that bothers you,
or you have less memory, set `exclusive = true` so loading one evicts the
other, or drop the text model to `mlx-community/Qwen3.6-35B-A3B-4bit` (~20 GB).

Idle models are evicted after `idle_evict_seconds`, so a machine that sits
unused gives its memory back.

## Using it from other tools

The server also speaks enough of the OpenAI API to point other local tooling at
it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

`/v1/models` and both streaming and non-streaming `/v1/chat/completions` are
implemented, including multimodal `image_url` content parts.

## Reaching it over SSH

The server binds to localhost by default. The safe way in from another machine
is an SSH tunnel rather than exposing the port:

```bash
# on the laptop
ssh -N -L 8080:127.0.0.1:8080 you@studio    # then use the web UI locally too

# or just run the CLI on the box
ssh you@studio -t hearth chat
```

If you would rather bind to the LAN directly, set `host = "0.0.0.0"` — but note
there is no authentication, so only do that on a network you trust.

## Data

```
~/.config/hearth/config.toml       configuration
~/.local/share/hearth/hearth.db    conversations (SQLite)
~/.local/share/hearth/images/      generated images and attachments
~/.local/share/hearth/cli_history  REPL history
```

### Images are kept, not reclaimed

Nothing in hearth ever deletes an image. Deleting a conversation removes its
messages from the database but leaves its images on disk, so the folder only
grows — a 1024x1024 PNG is roughly 1-2 MB, and attaching the same file twice
stores it twice.

This is deliberate: images are cheap to keep and annoying to lose, so tidying
is left to you. Both `hearth serve` and `hearth status` print the folder with a
current file count and size, so you can see when it is worth a look:

```
images: ~/.local/share/hearth/images (4 files, 5.5 MB)
        kept indefinitely - nothing is deleted, even when a conversation is
        removed, so tidy this folder by hand when it gets large
```

`hearth status` reports the *server's* folder, so the figure is right even when
the CLI is pointed at another machine.

## Tests

```bash
./run_tests.sh
```

The suite stubs out the two model engines, so it needs no weights and no GPU.
It also never touches the real internet: search is answered by a fake provider
and a fixture web server on an ephemeral loopback port.

| | |
|---|---|
| `test_search.py` | SSRF guards, HTML extraction, the context budget, tool-call parsing, the search heuristic |
| `test_integration.py` | routing, SSE framing, the reasoning split, thread persistence, path traversal, retrieval end to end, the OpenAI shim |
| `test_cli.py` | every CLI command against a live server, including pipe behaviour |
| `test_concurrency.py` | cancellation mid-generation and mid-fetch, request queueing, unloading |
| `test_repl.py` | the interactive REPL, driven through a real pty |
| `test_webui.py` | the web UI's routes: which Datastar frames each gesture produces, and what markup they carry |
| `test_markdown.py` | the Markdown subset, its HTML escaping, and source-link defusing |
| `web/datastar.test.js` | every `data-*` attribute in the page, checked against the runtime vendored beside it |

Once the weights are downloaded, there is also a smoke test against the real
models — text, multi-turn context, reasoning, and image generation:

```bash
./.venv/bin/python tests/smoke_real.py
```

## Layout

```
src/hearth/
  config.py       TOML + env configuration
  store.py        SQLite threads and messages
  textutil.py     incremental <think> and <tool_call> splitting
  search/
    providers.py  SearXNG and Brave behind one interface
    fetch.py      guarded fetch (SSRF) and HTML text extraction
    budget.py     fitting retrieved pages into the context window
    heuristics.py deciding whether a message needs the web
  engine/
    manager.py    job queue, lazy loading, idle eviction
    text.py       mlx-vlm text generation
    image.py      mflux image generation
  server.py       FastAPI: thread API, SSE, OpenAI shim
  client.py       HTTP client used by the CLI
  cli.py          commands and the interactive REPL
  search/         web search: providers, fetching, budgeting
  datastar.py     the two SSE frame types the browser understands
  mdrender.py     Markdown -> HTML, a small deliberate subset
  render.py       server-rendered HTML fragments for the web UI
  webui.py        the browser's routes: HTML and patches, not JSON
  web/index.html  the page: signals and URLs, no build step
  web/datastar.js the Datastar runtime, vendored (v1.0.3)
```

## The web UI

The browser frontend is [Datastar](https://data-star.dev/). There is no build
step, no framework, and almost no JavaScript: the page declares a handful of
signals and says which URL each gesture asks, and the server answers with the
HTML that should now be on screen.

```html
<input data-bind:search
       data-on:input__debounce.200ms="@get('/ui/threads')">
```

That is the entire search box. Typing updates the `search` signal, the signal
travels with the request, and `/ui/threads` returns a new `<div id="threads">`
that Datastar morphs into place.

Updates arrive as ordinary SSE. Two event types cover everything:

```
event: datastar-patch-elements
data: selector #mlist
data: mode append
data: elements <div class="msg user">...</div>

event: datastar-patch-signals
data: signals {"draft":"","atts":[]}
```

So a generation is one long response. The assistant's bubble is appended once
as a set of named, empty slots, and the reasoning block, sources strip,
progress bar, body and status line are then patched individually as events
arrive — coalesced to about fifteen frames a second, since re-rendering
Markdown per token is wasteful. When the turn ends the placeholder is replaced
by the *stored* message, which is byte-for-byte what a page reload would
render, so there is never a live version and a saved version to keep in step.
A turn that searched re-renders the whole transcript instead, because
retrieval inserts a `tool` message between the question and the answer.

The runtime is served from `/datastar.js` rather than a CDN, so the UI still
works with the network off.

The JSON API under `/api` is untouched by any of this — it is what the CLI, the
REPL and the OpenAI shim speak, and both frontends share one turn pipeline in
`server.py`, so they cannot drift apart.
