# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
uv venv --python 3.12
uv pip install -e .

./run_tests.sh                          # full suite; stubs the models, no GPU or weights needed
./.venv/bin/ruff check src/ tests/ --select F,E9

./.venv/bin/python tests/smoke_real.py  # real models; requires `hearth pull` first
```

`run_tests.sh` covers the HTTP surface, the CLI (spawned as real subprocesses),
cancellation and queueing, the REPL under a pty, and the web UI's markdown
renderer under node.

## Architecture Overview

One daemon owns the models; the CLI and the web UI are both thin HTTP clients
over it. That split is deliberate — it is what lets an SSH session start
instantly instead of loading a 38GB model, and it lets a conversation started
in the browser be continued from the terminal.

```
src/hearth/
  config.py       TOML + HEARTH_* env configuration
  store.py        SQLite threads and messages
  textutil.py     incremental <think> and <tool_call> splitting
  search/         web retrieval; a sibling of engine/ because it touches no model
  engine/
    manager.py    job queue, lazy loading, idle eviction
    text.py       mlx-vlm text generation
    image.py      mflux image generation
  server.py       FastAPI: thread API, SSE, OpenAI shim
  client.py       HTTP client used by the CLI
  cli.py          commands and the interactive REPL
  web/index.html  the web UI (single file, no build step)
```

## Conventions & Patterns

**All GPU work runs on one worker thread** (`engine/manager.py`). MLX is not
safe to run concurrently and two large models racing for unified memory will
wedge the machine, so requests queue rather than collide.

**Retrieval never runs on the worker thread.** `src/hearth/search/` is a
sibling of `engine/`, not a part of it, for this reason: the worker thread is
effectively the GPU lock, and a ten second fetch taken on it stalls every
queued generation behind it. Web search runs in the server's request path,
inside the streaming response (so it is off the event loop too), and a turn
that searches becomes two model jobs with an HTTP round trip in between.

**mflux must generate on the thread that loaded it.** MLX streams are
per-thread; running the denoise loop on a different thread fails with
"There is no Stream(cpu, N)". This is why image progress is reported through an
`emit` callback rather than yielded — yielding would require a second thread.

**Qwen's chat template pre-opens `<think>`.** With reasoning on, the prompt ends
in a bare `<think>` and the model's first token is already inside the block, so
the only tag ever seen is the closing one. `ThinkSplitter(start_in_think=True)`
handles this; the engine signals it with a `start` event.

**Request models are strict** (`extra="forbid"`). A misspelled field should 422,
not be silently ignored — a CLI/server field-name mismatch once made `--think` a
no-op. The OpenAI shim stays permissive on purpose.

**The CLI must never hang.** When a prompt argument is given, stdin is read only
if data is already waiting; a blocking read there hangs forever under launchd,
cron, or any parent that leaks an open stdin.

**Tests bind ephemeral ports** and always pass explicit subprocess stdin. Fixed
ports and inherited stdin were both sources of real flakiness.
