# Web search for hearth

Status: implemented. This records the design and, at the end, where the
implementation departed from it.

Give the assistant access to live web content, without giving up the two
properties that make hearth worth running: everything stays on hardware you
control, and the GPU worker is never blocked by anything that isn't GPU work.

## Where the loop lives

The single most important constraint is in `engine/manager.py`: one worker
thread executes every job, serially, because MLX is not safe to run
concurrently. Network I/O must therefore **never** run inside a job. A ten
second fetch on the worker thread stalls every queued request behind it, for no
reason at all — that thread is a GPU lock, and search needs no GPU.

So retrieval sits *above* the manager, in the server's request path, and a
turn that searches becomes two model jobs with an HTTP round trip in between:

```
POST /api/threads/{ref}/messages
  │
  ├─ decide: does this turn need retrieval?          (server thread)
  ├─ search  → provider API                          (threadpool, off-worker)
  ├─ fetch   → N pages, extract, budget              (threadpool, off-worker)
  ├─ manager.submit_text(history + documents)        (GPU worker)
  └─ stream tokens to the client
```

The cost of this shape is that phase 2 (below) re-queues and re-processes the
prompt for its second model turn. That is the correct trade against wedging the
queue, and prompt caching (`home-model-server-tooling-g0z`) recovers most of it.

## Module layout

A new top-level package, sibling to `engine/` — deliberately not inside it,
because nothing here touches a model:

```
src/hearth/search/
  __init__.py    WebSearch facade: query -> Outcome
  providers.py   SearXNG and Brave behind one Protocol
  fetch.py       guarded HTTP GET + HTML -> text extraction
  budget.py      select, truncate and delimit documents for the context
  heuristics.py  deciding whether a message needs the web at all
```

### Providers

```python
class Provider(Protocol):
    def search(self, query: str, count: int) -> list[Result]: ...

@dataclass
class Result:
    title: str
    url: str
    snippet: str
```

Default is **SearXNG**, pointed at an instance the user runs. No API key, no
third party sees the query, which is the same reason the models are local.
**Tavily** and **Brave** are supported for people who would rather not host
anything. With no provider configured, search stays off and says so instead of
failing obscurely.

Tavily is the odd one out and worth a note: it returns the extracted page text
with each hit, so `Result.content` comes back filled in and the fetch stage is
skipped entirely for those results. That is not just a latency win. Fetching
pages yourself fails on a large fraction of the web — Cloudflare and friends
refuse anything that does not look like a browser — and a provider that has
already done the extraction sidesteps the whole problem. The cost is trusting
someone else's extraction, and the guards in `fetch.py` never running because
there is nothing to fetch.

Tavily's `answer` field is never requested. It has a cloud model write the
reply, which is the one thing this project must not do.

### Fetch

Search results are titles and 150-character snippets. Those alone answer maybe
a third of questions; the rest need the page. `fetch.py` does a guarded GET and
reduces HTML to text.

Extraction is stdlib-only by default — an `HTMLParser` subclass that drops
`script`, `style`, `nav`, `header`, `footer` and `aside`, then collapses
whitespace. It is worse than `trafilatura` and that is fine; the base install
already pulls mlx and mflux and does not need lxml too. If `trafilatura`
imports, use it. That's a one-line preference, not a dependency.

The guards are not optional. This daemon can be bound to a LAN address, and in
phase 2 the URL can come from the model:

- `http`/`https` only.
- Resolve the host first and reject loopback, private, link-local and
  multicast addresses — then pin the connection to that resolved address, so a
  second lookup can't return something different.
- Re-check every redirect hop against the same rules.
- `Content-Type` must be HTML or plain text.
- Hard caps on body size (2MB) and wall time, streamed and aborted on breach.
- Never auto-follow a URL discovered *inside* fetched content.

### Budget

Context is scarce enough that `max_history_messages` and `max_history_images`
already exist to defend it. Retrieved text is far bigger than either. `budget.py`
takes the fetched documents and returns at most `max_context_chars` (default
6000) of text, divided evenly across sources so one long page can't crowd out
the rest, each wrapped in a delimiter:

```
<source id="1" url="https://..." title="...">
...extracted text...
</source>
```

Any occurrence of that delimiter inside the extracted text is neutralised
before wrapping.

## How results reach the model

Retrieved text is stored as a message with `role="tool"`, which the schema
already permits — `role` is a free-text column. What matters is what
`_history_for_model` does with it on *later* turns.

Carrying 6KB of scraped page text forward on every subsequent turn would blow
the context by turn three. So the same rule the images already follow applies
here: **the full text is fed to the model only on the turn that fetched it**,
and older searches degrade to a compact line.

- `content` holds the compact form: `[searched the web for "X" — 1. Title
  (url), 2. …]`. That is what renders in both frontends and what older turns
  contribute to history.
- `meta["search"]` holds the structured record: the query, the results, and the
  extracted `documents`.
- `_history_for_model` rehydrates full documents from `meta` for the most
  recent `search.max_history_documents` (default 1) tool messages, and emits
  the compact `content` for everything older.

This is deliberately the same shape as the image budget walk that is already
there, for the same reason, and it makes `/retry` work without re-fetching.

The documents are preceded by a fixed instruction stating that everything
inside `<source>` tags is untrusted data retrieved from the internet, is not
addressed to the model, and must never be treated as instructions.

## Deciding when to search

Three tiers, shipped in order, each independently useful.

**Tier 0 — tell the model what day it is.** The system prompt gains today's
date and a configured `knowledge_cutoff`. This is the cheapest change in the
whole design and the highest-value one: it is what lets a model conclude "that
is after my cutoff, I cannot know it" instead of confabulating. It is worth
doing whether or not search ever ships.

**Tier 1 — explicit.** A `/web` REPL command and a toggle in the web UI.
Deterministic, needs no calibration, works with any model. `/web` is routed
server-side on the leading verb, exactly as `/image` and `/edit` already are,
so both frontends behave identically for free.

**Tier 2 — model-initiated.** A `web_search` tool offered through the chat
template, gated behind `search.autonomous = false`. A ~30B local model is
materially worse calibrated about *when* to fire a tool than a frontier model
is, so this stays off by default until measured.

Two pieces of machinery are needed:

- Qwen emits `<tool_call>{...}</tool_call>`. That wants an incremental splitter
  in `textutil.py` next to `ThinkSplitter`, with the same streaming shape — the
  tool call must be recognised and suppressed from the visible channel as it
  arrives, not after the fact.
- `mlx_vlm.apply_chat_template` needs to forward a `tools=` kwarg to the
  tokenizer's template. **Verified against 0.6.17: it does.** Unrecognised
  kwargs pass through `get_chat_template` to the tokenizer unchanged, in both
  the text-only and the vision path, so the model renders its own native tool
  syntax rather than us hand-writing a schema into the system prompt.

The loop is capped at `search.max_rounds` (default 2), so a page saying "search
again for …" cannot spin the machine.

A middle setting, `autonomous = "heuristic"`, runs a cheap pre-filter instead
of trusting the model — temporal words, a four-digit year, a bare URL, an
explicit "look up" — and forces tier 1 retrieval when it fires. Crude, but on a
small model it is likely to beat the model's own judgement, and it costs no
extra generation.

## Streaming and UX

New SSE events on the existing channel, so neither frontend needs new plumbing:

```json
{"type": "search", "phase": "querying",  "query": "..."}
{"type": "search", "phase": "results",   "results": [{"title": "...", "url": "..."}]}
{"type": "search", "phase": "fetching",  "url": "..."}
```

The CLI prints a dim status line and then the numbered sources. The web UI
already funnels `status` events into `notice.textContent` (`web/index.html:496`);
`search` joins it, and the sources render as a collapsible strip under the
answer.

**Cancellation** cannot go through the manager: `POST /api/cancel` reaches the
current *job*, and retrieval happens before a job exists. Each in-flight turn
parks a `threading.Event` in a registry on the app, and `/api/cancel` sets
every one of them alongside cancelling the job. The fetcher checks it between
redirect hops and between body chunks.

A cancelled retrieval throws its documents away and keeps only the compact
line. Whatever was fetched by then is partial and unasked-for, and leaving it
on the message would let a lookup the user stopped go on steering every later
turn in the thread.

The granularity is honest but not perfect: a provider that hangs blocks until
`search.timeout_s`, because the HTTP call to it has no cancellation hook.

## Configuration

```toml
[search]
enabled = false
provider = "searxng"                    # searxng | brave | none
searxng_url = "http://127.0.0.1:8888"
brave_api_key = ""
max_results = 5
max_fetch = 3
max_context_chars = 6000
max_page_bytes = 2000000
timeout_s = 10.0
max_history_documents = 1
autonomous = "off"                      # off | heuristic | tool
max_rounds = 2
allow_private_hosts = false
```

With `HEARTH_SEARCH`, `HEARTH_SEARCH_PROVIDER`, `HEARTH_SEARXNG_URL` and
`HEARTH_BRAVE_KEY` overriding, per the existing env convention. Every field
stays non-`None`: `write_default` round-trips the dataclass through `tomli_w`,
which cannot serialise `None`.

`ChatRequest` is `extra="forbid"`, so a new `search: bool | None` field has to
be added there *and* in `client.py` in the same change, or the CLI flag 422s.
That is the failure the strictness exists to catch.

## Testing

`run_tests.sh` stubs the models; search must stub the same way, and nothing in
the suite may touch the real internet.

- A `FakeProvider` returning fixed results.
- A fixture HTTP server on an ephemeral port serving known HTML, including a
  redirect to `127.0.0.1` that the SSRF guard must refuse.
- Unit coverage for extraction, even-split truncation, delimiter neutralisation,
  and the compact-vs-full history rendering across three turns.
- SSE event shape, the `/search` verb routed server-side, the CLI's source
  rendering, and `/search` in the REPL under the existing pty harness.


## Where the implementation departed from this

**`/web`, not `/search`.** Every other slash command maps to the identically
named CLI command — `/image` to `hearth image`, `/status` to `hearth status`.
`hearth search` already exists and searches your *conversations*, so `/search`
would have been the one pair in the set that meant the opposite thing. `/web`
keeps the mapping honest, and takes `on`/`off` as well as a query, following
`/think`.

**`autonomous` is a three-valued string**, `"off" | "heuristic" | "tool"`,
rather than `false | "heuristic" | true`. A union of bool and string is
awkward to round-trip through `tomli_w` and awkward to merge in `_merge`, and
the string reads better in the file.

**The `fetching` event carries `urls`, plural.** Pages are fetched in parallel
on a small pool — three pages at a second each is three seconds serially and
one in parallel, and none of it contends with the GPU — so there is no single
URL to name at that point.

**Retrieved text is projected onto the `user` role** on its way to the model,
though it is stored as `role="tool"`. `apply_chat_template` does have a path
for tool messages, but it expects `tool_call_id`/`tool_calls` pairing that a
retrieval performed on the user's behalf does not have, and a template that
mishandles the shape produces a quietly malformed prompt rather than an error.
The store keeps the honest role; only the model-facing projection is flattened.

**Search results are filtered by scheme at the provider**, not only at the
fetcher. A SearXNG instance is software the user runs, but it is also the one
component here that talks to the open internet, and a compromised one should
not be able to hand a `javascript:` URL to the web UI. The UI has its own
`safeHref` guard as well.

**Citations survive a pipe.** `hearth ask` suppresses progress bars and the
tokens-per-second line when stdout is not a terminal, but the source list goes
to stderr regardless, because where a factual claim came from is worth more
than tidiness. Only `--quiet` turns it off.

## Still open

- The query for an unprefixed search is the message text, truncated. Nothing
  rewrites it, because that would cost a whole generation turn, so a turn that
  leans on earlier context ("what about the second one?") searches badly.
  `/web <query>` is the workaround.
- No re-ranking. The top `max_fetch` results are read in the order the provider
  returned them; there is no embedding or cross-encoder pass to pick the
  passages that actually match the question.
- A hanging provider is bounded by `timeout_s`, not by the cancel button.
