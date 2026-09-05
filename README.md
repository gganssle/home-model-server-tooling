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

```
/new [title]        start a new conversation
/threads            list conversations
/switch <ref>       jump to another conversation
/image <prompt>     generate an image in this conversation
/attach <path>      queue an image for the model to look at
/detach             drop queued attachments
/edit <prompt>      redraw the newest image in this conversation
/think on|off       toggle reasoning mode
/retry              re-run your last message
/title <text>       rename this conversation
/show               reprint the conversation
/status             what is loaded, memory use
/unload [what]      free memory
/quit               exit (Ctrl-D works too)
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

In a conversation, `/edit make the sky stormier` works on the newest image in
that thread — in the REPL and the web UI both. In the web UI, **Use as base**
under any generated image loads it into the composer with a strength slider.

Generated and attached images live in `~/.local/share/hearth/images/` and are
attached to the conversation they belong to. Attachments are copied in rather
than referenced, so a conversation still renders after you move or delete the
original.

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

[models.image]
repo = "mlx-community/Qwen-Image-2512-4bit"
steps = 20
width = 1024
height = 1024
guidance = 4.0
image_strength = 0.6       # default for --from / /edit, 0-1

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

The suite stubs out the two model engines, so it needs no weights and no GPU:

| | |
|---|---|
| `test_integration.py` | routing, SSE framing, the reasoning split, thread persistence, path traversal, the OpenAI shim |
| `test_cli.py` | every CLI command against a live server, including pipe behaviour |
| `test_concurrency.py` | cancellation mid-generation, request queueing, unloading |
| `test_repl.py` | the interactive REPL, driven through a real pty |
| `web/md.test.js` | the UI's markdown renderer and its HTML escaping |

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
  textutil.py     incremental <think> block splitting
  engine/
    manager.py    job queue, lazy loading, idle eviction
    text.py       mlx-vlm text generation
    image.py      mflux image generation
  server.py       FastAPI: thread API, SSE, OpenAI shim
  client.py       HTTP client used by the CLI
  cli.py          commands and the interactive REPL
  web/index.html  the web UI (single file, no build step)
```
