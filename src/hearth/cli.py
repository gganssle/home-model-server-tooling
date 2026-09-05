"""hearth command line interface.

Designed for two shapes of use:
  * interactive  - `hearth chat`, a streaming REPL over SSH
  * scriptable   - `hearth ask "..."` reads stdin and writes plain stdout,
                   so it composes with pipes
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from hearth import config as config_mod
from hearth.client import HearthClient, ServerUnavailable

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local chat + image generation against your own models.",
)

console = Console()
err = Console(stderr=True)


def is_tty() -> bool:
    return sys.stdout.isatty()


def get_client() -> HearthClient:
    return HearthClient(config_mod.load().base_url)


def fail(message: str, code: int = 1) -> None:
    err.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def tilde(path: Path) -> str:
    """Shorten a path under $HOME to ~/..., which keeps it on one line."""
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def image_store_summary(stats: dict[str, Any]) -> str:
    """One line describing the image folder, from the server's own report."""
    where = tilde(Path(stats["dir"]))
    if stats.get("files") is None:
        return f"{where} (unreadable)"
    if not stats["files"]:
        return f"{where} (empty)"
    return f"{where} ({stats['files']} files, {human_size(stats['bytes'])})"


def human_age(ts: float) -> str:
    delta = time.time() - ts
    for limit, unit, div in ((60, "s", 1), (3600, "m", 60), (86400, "h", 3600)):
        if delta < limit:
            return f"{int(delta / div)}{unit}"
    return f"{int(delta / 86400)}d"


# --------------------------------------------------------------------------
# streaming render
# --------------------------------------------------------------------------

def render_stream(
    events: Iterator[dict[str, Any]],
    client: HearthClient,
    show_thinking: bool = True,
    plain: bool = False,
) -> dict[str, Any]:
    """Render an SSE event stream to the terminal.

    Returns the final `done` event. Ctrl-C cancels the generation server-side
    rather than killing the client, so a runaway answer costs you nothing.
    """
    final: dict[str, Any] = {}
    in_thinking = False
    wrote_any = False
    progress_active = False

    try:
        for event in events:
            etype = event.get("type")

            if etype == "queued":
                if not plain:
                    err.print("[dim]waiting for the model (another request is running)...[/dim]")

            elif etype == "status":
                if not plain:
                    err.print(f"[dim]{event['text']}…[/dim]")

            elif etype == "progress":
                if not plain:
                    step, total = event["step"], event["total"]
                    bar_width = 24
                    filled = int(bar_width * step / max(total, 1))
                    bar = "#" * filled + "-" * (bar_width - filled)
                    sys.stderr.write(f"\r  [{bar}] step {step}/{total}")
                    sys.stderr.flush()
                    progress_active = True

            elif etype == "token":
                channel = event.get("channel", "content")
                if channel == "thinking":
                    if not show_thinking or plain:
                        continue
                    if not in_thinking:
                        console.print("\n[dim italic]thinking:[/dim italic]", end=" ")
                        in_thinking = True
                    console.print(f"[dim italic]{event['text']}[/dim italic]", end="")
                else:
                    if in_thinking:
                        console.print("\n")
                        in_thinking = False
                    sys.stdout.write(event["text"])
                    sys.stdout.flush()
                    wrote_any = True

            elif etype == "done":
                if progress_active:
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                final = event
                if wrote_any:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

            elif etype == "cancelled":
                final = {"type": "cancelled"}
                if not plain:
                    err.print("\n[yellow]cancelled[/yellow]")

            elif etype == "error":
                if progress_active:
                    sys.stderr.write("\n")
                fail(event.get("error", "unknown server error"))

    except KeyboardInterrupt:
        try:
            client.cancel()
        except Exception:
            pass
        err.print("\n[yellow]cancelled[/yellow]")
        return {"type": "cancelled"}

    return final


MIME_BY_SUFFIX = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif",
}


def to_data_uri(path: Path) -> str:
    """Read a local image into a data: URI.

    Sending bytes rather than a path means attachments work unchanged when the
    server is on another machine, which is the whole point of HEARTH_HOST.
    """
    import base64

    resolved = path.expanduser()
    if not resolved.is_file():
        fail(f"no such image: {path}")
    mime = MIME_BY_SUFFIX.get(resolved.suffix.lower(), "image/png")
    payload = base64.b64encode(resolved.read_bytes()).decode()
    return f"data:{mime};base64,{payload}"


def image_ref(value: str) -> str:
    """Accept a local file or a name the server already knows.

    A path that exists here is uploaded; anything else is passed through for
    the server to resolve against images it already has.
    """
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return to_data_uri(candidate)
    return value


def show_inline(data: bytes) -> bool:
    """Draw an image in the terminal if it speaks a graphics protocol.

    iTerm2 and WezTerm understand this escape sequence, and it survives SSH,
    which makes image generation genuinely usable from a remote shell. Anywhere
    else this is skipped and we just print the path.
    """
    import base64

    if not is_tty():
        return False
    term = os.environ.get("TERM_PROGRAM", "")
    if term not in ("iTerm.app", "WezTerm"):
        return False
    payload = base64.b64encode(data).decode()
    sys.stdout.write(f"\033]1337;File=inline=1;width=40;preserveAspectRatio=1:{payload}\a\n")
    sys.stdout.flush()
    return True


def print_stats(final: dict[str, Any]) -> None:
    meta = final.get("meta") or {}
    if not meta:
        return
    bits = []
    if meta.get("tokens_per_second"):
        bits.append(f"{meta['tokens_per_second']} tok/s")
    if meta.get("generation_tokens"):
        bits.append(f"{meta['generation_tokens']} tokens")
    if meta.get("elapsed_s"):
        bits.append(f"{meta['elapsed_s']}s")
    if bits:
        err.print(f"[dim]{' · '.join(bits)}[/dim]")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Bind address. Use 0.0.0.0 to expose on your LAN."),
    port: Optional[int] = typer.Option(None, help="Port to listen on."),
    preload: bool = typer.Option(False, "--preload", help="Load the text model at startup."),
) -> None:
    """Run the model server. This is the process that holds the models."""
    cfg = config_mod.load()
    if host:
        cfg.server.host = host
    if port:
        cfg.server.port = port
    config_mod.write_default()

    import logging
    import uvicorn
    from hearth.server import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    application = create_app(cfg)

    if preload:
        err.print("[dim]preloading text model…[/dim]")
        application.state.manager.preload("text")

    err.print(f"[green]hearth[/green] serving on http://{cfg.server.host}:{cfg.server.port}")
    err.print(f"[dim]text : {cfg.text.repo}[/dim]")
    err.print(f"[dim]image: {cfg.image.repo}[/dim]")
    # soft_wrap keeps a long path on one line instead of breaking it mid-path.
    err.print(
        f"[dim]images: {image_store_summary(config_mod.image_store_stats(cfg))}[/dim]",
        soft_wrap=True,
    )
    err.print("[dim]        kept indefinitely - nothing is deleted, even when a "
              "conversation is[/dim]", soft_wrap=True)
    err.print("[dim]        removed, so tidy this folder by hand when it gets "
              "large[/dim]", soft_wrap=True)
    uvicorn.run(application, host=cfg.server.host, port=cfg.server.port, log_level="warning")


def _read_stdin(required: bool, wait: float = 0.5) -> str:
    """Read piped input, without hanging when there is none.

    A blocking read is right when stdin *is* the prompt. But when a prompt was
    also given on the command line, stdin is only supplementary - and anything
    non-interactive (launchd, cron, a parent that leaks its own stdin) hands us
    an open pipe nobody will ever write to or close. Blocking there means the
    command hangs forever with no output, so we wait briefly and move on.
    """
    if sys.stdin.isatty():
        return ""
    if required:
        return sys.stdin.read().strip()
    import select

    try:
        ready, _, _ = select.select([sys.stdin], [], [], wait)
    except (OSError, ValueError):
        return ""
    if not ready:
        return ""
    return sys.stdin.read().strip()


@app.command()
def ask(
    prompt: Optional[str] = typer.Argument(None, help="Prompt. Omit to read stdin."),
    thread: Optional[str] = typer.Option(None, "--thread", "-t", help="Continue a thread (id, prefix, or 'last')."),
    new: bool = typer.Option(False, "--new", "-n", help="Force a fresh throwaway thread."),
    images: Optional[list[Path]] = typer.Option(
        None, "--image", "-i",
        help="Attach an image for the model to look at. Repeatable.",
    ),
    think: Optional[bool] = typer.Option(None, "--think/--no-think", help="Toggle reasoning mode."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Answer only: no stats, no reasoning."),
) -> None:
    """One-shot question. Pipe-friendly: `cat bug.log | hearth ask 'what broke?'`

    Attach images with -i to ask about them:
    `hearth ask -i chart.png "what is the trend?"`
    """
    stdin_text = _read_stdin(required=prompt is None)

    attachments = [to_data_uri(p) for p in (images or [])]

    parts = [p for p in (prompt, stdin_text) if p]
    if not parts and not attachments:
        fail("no prompt given (pass an argument, pipe text in, or attach an image)")
    content = "\n\n".join(parts)

    plain = quiet or not is_tty()

    with get_client() as client:
        try:
            ref = _pick_thread(client, thread, new)
            final = render_stream(
                client.send(ref, content, thinking=think, max_tokens=max_tokens,
                            temperature=temperature, images=attachments or None),
                client,
                show_thinking=not plain,
                plain=plain,
            )
        except ServerUnavailable as exc:
            fail(str(exc))
        if not plain:
            print_stats(final)


def _pick_thread(client: HearthClient, thread: Optional[str], new: bool) -> str:
    """Reuse the named thread, or open a fresh one."""
    if thread and not new:
        return thread
    return client.create_thread()["id"]


@app.command()
def image(
    prompt: str = typer.Argument(..., help="What to draw."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the PNG here."),
    thread: Optional[str] = typer.Option(None, "--thread", "-t", help="Attach to a thread."),
    steps: Optional[int] = typer.Option(None, "--steps"),
    width: Optional[int] = typer.Option(None, "--width"),
    height: Optional[int] = typer.Option(None, "--height"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    negative: Optional[str] = typer.Option(None, "--negative", help="Negative prompt."),
    from_image: Optional[str] = typer.Option(
        None, "--from", help="Start from this image (a local file, or one already in a thread)."
    ),
    strength: Optional[float] = typer.Option(
        None, "--strength",
        help="How far to move from --from, 0-1. Low stays close to the original.",
    ),
    open_after: bool = typer.Option(False, "--open", help="Open the result when done (macOS)."),
) -> None:
    """Generate an image, or vary an existing one with --from.

    `hearth image "the same barn in winter" --from barn.png --strength 0.5`
    """
    with get_client() as client:
        try:
            final = render_stream(
                client.image(
                    prompt, thread_id=thread, steps=steps, width=width,
                    height=height, seed=seed, negative_prompt=negative,
                    init_image=image_ref(from_image) if from_image else None,
                    image_strength=strength,
                ),
                client,
                plain=not is_tty(),
            )
        except ServerUnavailable as exc:
            fail(str(exc))

        if final.get("type") != "done":
            raise typer.Exit(1)

        filename = final["image"]
        data = client.download_image(filename)
        dest = Path(out) if out else Path.cwd() / filename
        dest.write_bytes(data)
        show_inline(data)
        print(str(dest.resolve()))
        meta = final.get("meta", {})
        bits = [
            f"seed {meta.get('seed')}",
            f"{meta.get('steps')} steps",
            f"{meta.get('elapsed_s')}s",
        ]
        if meta.get("from_image"):
            bits.insert(0, f"from {meta['from_image']} @ {meta.get('image_strength')}")
        err.print(f"[dim]{' - '.join(bits)}[/dim]")
        if open_after:
            import subprocess
            subprocess.run(["open", str(dest.resolve())], check=False)


@app.command(name="threads")
def list_threads() -> None:
    """List conversations, newest first."""
    with get_client() as client:
        try:
            threads = client.list_threads()
        except ServerUnavailable as exc:
            fail(str(exc))
    if not threads:
        err.print("[dim]no conversations yet[/dim]")
        return
    if not is_tty():
        for t in threads:
            print(f"{t['id']}\t{t['message_count']}\t{t['title']}")
        return
    table = Table(box=None, pad_edge=False)
    table.add_column("id", style="cyan")
    table.add_column("age", style="dim", justify="right")
    table.add_column("msgs", justify="right", style="dim")
    table.add_column("title")
    for t in threads:
        table.add_row(t["id"], human_age(t["updated_at"]), str(t["message_count"]), t["title"])
    console.print(table)


@app.command()
def show(ref: str = typer.Argument("last", help="Thread id, unique prefix, or 'last'.")) -> None:
    """Print a whole conversation."""
    with get_client() as client:
        try:
            data = client.get_thread(ref)
        except ServerUnavailable as exc:
            fail(str(exc))
        except RuntimeError as exc:
            fail(str(exc))
    thread = data["thread"]
    if is_tty():
        console.print(f"[bold cyan]{thread['title']}[/bold cyan] [dim]({thread['id']})[/dim]\n")
    for m in data["messages"]:
        _print_message(m)


def _print_message(m: dict[str, Any]) -> None:
    role = m["role"]
    if not is_tty():
        print(f"[{role}] {m['content']}")
        return
    label = {"user": "[bold green]you[/bold green]", "assistant": "[bold magenta]model[/bold magenta]"}.get(
        role, f"[bold]{role}[/bold]"
    )
    console.print(f"{label}:")
    for name in (m.get("meta") or {}).get("images", []) or []:
        console.print(f"  [cyan]<attached {name}>[/cyan]")
    if m.get("image"):
        console.print(f"  [cyan]<image {m['image']}>[/cyan]")
    if m["content"]:
        console.print(Markdown(m["content"]))
    console.print()


@app.command(name="rm")
def remove_thread(
    ref: str = typer.Argument(..., help="Thread id or unique prefix."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete a conversation."""
    with get_client() as client:
        try:
            data = client.get_thread(ref)
        except (ServerUnavailable, RuntimeError) as exc:
            fail(str(exc))
        title = data["thread"]["title"]
        if not yes:
            if not typer.confirm(f"delete {data['thread']['id']} ({title})?"):
                raise typer.Exit(0)
        client.delete_thread(ref)
        err.print(f"[dim]deleted {data['thread']['id']}[/dim]")


@app.command()
def search(query: str = typer.Argument(..., help="Substring to look for.")) -> None:
    """Search across all conversations."""
    with get_client() as client:
        try:
            results = client.search(query)
        except ServerUnavailable as exc:
            fail(str(exc))
    if not results:
        err.print("[dim]no matches[/dim]")
        return
    for r in results:
        snippet = " ".join(r["content"].split())
        if len(snippet) > 100:
            idx = snippet.lower().find(query.lower())
            start = max(0, idx - 30)
            snippet = ("…" if start else "") + snippet[start:start + 100] + "…"
        if is_tty():
            console.print(f"[cyan]{r['thread_id']}[/cyan] [dim]{r['role']}[/dim] {snippet}")
        else:
            print(f"{r['thread_id']}\t{r['role']}\t{snippet}")


@app.command()
def status() -> None:
    """Show what is loaded and how much memory it is using."""
    with get_client() as client:
        try:
            st = client.status()
        except ServerUnavailable as exc:
            fail(str(exc))
    if not is_tty():
        print(json.dumps(st, indent=2))
        return
    table = Table(box=None, pad_edge=False)
    table.add_column("", style="dim")
    table.add_column("")
    for kind in ("text", "image"):
        info = st[kind]
        state = "[green]loaded[/green]" if info["loaded"] else "[dim]not loaded[/dim]"
        idle = f" [dim](idle {int(info['idle_s'])}s)[/dim]" if info["loaded"] and info["idle_s"] else ""
        table.add_row(kind, f"{info['repo']}  {state}{idle}")
    mem = st["memory"]
    table.add_row("memory", f"{mem['active_gb']} GB active · {mem['cache_gb']} GB cache · {mem['peak_gb']} GB peak")
    table.add_row("busy", st["busy_with"] or "[dim]idle[/dim]")
    table.add_row("queued", str(st["queue_depth"]))
    table.add_row("threads", str(st.get("threads", 0)))
    if st.get("images"):
        table.add_row("images", image_store_summary(st["images"]))
    console.print(table)


@app.command()
def pull(
    which: str = typer.Argument("all", help="all | text | image"),
) -> None:
    """Download model weights from Hugging Face into the local cache."""
    from huggingface_hub import snapshot_download

    cfg = config_mod.load()
    targets = []
    if which in ("all", "text"):
        targets.append(("text", cfg.text.repo))
    if which in ("all", "image"):
        targets.append(("image", cfg.image.repo))
    if not targets:
        fail("which must be one of: all, text, image")

    for kind, repo in targets:
        err.print(f"[bold]downloading {kind} model:[/bold] {repo}")
        path = snapshot_download(repo_id=repo)
        err.print(f"[green]done[/green] [dim]{path}[/dim]")


@app.command(name="config")
def config_cmd(
    edit: bool = typer.Option(False, "--edit", help="Open the config in $EDITOR."),
    path_only: bool = typer.Option(False, "--path", help="Print the config path and exit."),
) -> None:
    """Show or edit the configuration."""
    path = config_mod.write_default()
    if path_only:
        print(path)
        return
    if edit:
        os.system(f'{os.environ.get("EDITOR", "vi")} "{path}"')
        return
    console.print(f"[dim]{path}[/dim]\n")
    console.print(path.read_text())


@app.command(name="unload")
def unload_cmd(which: str = typer.Argument("all", help="all | text | image")) -> None:
    """Evict models from memory without stopping the server."""
    with get_client() as client:
        try:
            res = client.unload(which)
        except ServerUnavailable as exc:
            fail(str(exc))
    freed = res["unloaded"]
    err.print(f"[dim]unloaded: {', '.join(freed) if freed else 'nothing was loaded'}[/dim]")


@app.command(name="install-service")
def install_service() -> None:
    """Write a launchd plist so the server starts at login and stays up."""
    cfg = config_mod.load()
    label = "com.hearth.server"
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    exe = Path(sys.argv[0]).resolve()
    log_dir = config_mod.DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{log_dir / 'hearth.log'}</string>
    <key>StandardErrorPath</key><string>{log_dir / 'hearth.err.log'}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HEARTH_HOST</key><string>{cfg.server.host}</string>
        <key>HEARTH_PORT</key><string>{cfg.server.port}</string>
    </dict>
</dict>
</plist>
"""
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist)
    console.print(f"wrote [cyan]{plist_path}[/cyan]\n")
    console.print("Load it with:")
    console.print(f"  [bold]launchctl load -w {plist_path}[/bold]")
    console.print(f"Logs: [dim]{log_dir}[/dim]")


def main() -> None:
    # Default to the chat REPL when invoked bare in a terminal.
    app()


# --------------------------------------------------------------------------
# interactive REPL
# --------------------------------------------------------------------------

REPL_HELP = """
[bold]commands[/bold]
  /new [title]        start a new conversation
  /threads            list conversations
  /switch <ref>       jump to another conversation (id, prefix, or 'last')
  /image <prompt>     generate an image in this conversation
  /attach <path>      queue an image for the model to look at
  /detach             drop queued attachments
  /edit <prompt>      redraw the newest image in this conversation
  /think on|off       toggle reasoning mode
  /retry              re-run your last message
  /title <text>       rename this conversation
  /show               reprint the conversation so far
  /status             what is loaded, memory use
  /unload [all|text|image]   free memory
  /help               this list
  /quit               exit  (Ctrl-D also works)

[dim]Ctrl-C stops a running generation without leaving the chat.
Submit a multi-line message with Esc then Enter.[/dim]
"""


@app.command()
def chat(
    ref: Optional[str] = typer.Argument(None, help="Resume a thread (id, prefix, or 'last')."),
    think: bool = typer.Option(False, "--think", help="Start with reasoning mode on."),
) -> None:
    """Interactive chat. This is the one you want over SSH."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

    cfg = config_mod.load()
    client = HearthClient(cfg.base_url)

    try:
        client.status()
    except ServerUnavailable as exc:
        client.close()
        fail(str(exc))

    if ref:
        try:
            thread = client.get_thread(ref)["thread"]
        except RuntimeError as exc:
            client.close()
            fail(str(exc))
        console.print(f"[dim]resuming[/dim] [bold cyan]{thread['title']}[/bold cyan] [dim]({thread['id']})[/dim]")
        for m in client.get_thread(thread["id"])["messages"][-6:]:
            _print_message(m)
    else:
        thread = client.create_thread()
        console.print(f"[dim]new conversation {thread['id']}[/dim]")

    console.print(f"[dim]{cfg.text.repo}[/dim]")
    console.print("[dim]/help for commands, /quit to exit[/dim]\n")

    history_path = config_mod.DATA_DIR / "cli_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory(),
    )

    thinking_on = think
    last_user_message: str | None = None
    pending: list[Path] = []   # images queued by /attach for the next message

    def send(content: str) -> None:
        """Send one turn, with any queued attachments, and render the reply.

        `/image` and `/edit` go through here too: the server routes on the
        leading verb, so the REPL and the web UI behave identically.
        """
        nonlocal last_user_message
        last_user_message = content
        attachments = [to_data_uri(p) for p in pending]
        console.print()
        final = render_stream(
            client.send(thread["id"], content, thinking=thinking_on,
                        images=attachments or None),
            client,
            show_thinking=True,
        )
        if attachments:
            pending.clear()

        if final.get("type") == "done" and final.get("image"):
            data = client.download_image(final["image"])
            dest = Path.cwd() / final["image"]
            dest.write_bytes(data)
            show_inline(data)
            console.print(f"[cyan]image:[/cyan] {dest}")
            meta = final.get("meta", {})
            bits = [f"seed {meta.get('seed')}", f"{meta.get('elapsed_s')}s"]
            if meta.get("from_image"):
                bits.insert(0, f"from {meta['from_image']} @ {meta.get('image_strength')}")
            console.print(f"[dim]{' - '.join(bits)}[/dim]")
        print_stats(final)
        console.print()

    while True:
        try:
            line = session.prompt("you › ", multiline=False).strip()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

        if not line:
            continue

        if not line.startswith("/"):
            send(line)
            continue

        cmd, _, rest = line.partition(" ")
        cmd = cmd.lower()
        rest = rest.strip()

        if cmd in ("/quit", "/exit", "/q"):
            break

        elif cmd == "/help":
            console.print(REPL_HELP)

        elif cmd == "/new":
            thread = client.create_thread(rest or "New conversation")
            console.print(f"[dim]new conversation {thread['id']}[/dim]\n")

        elif cmd == "/threads":
            for t in client.list_threads()[:20]:
                marker = "[green]*[/green]" if t["id"] == thread["id"] else " "
                console.print(f"{marker} [cyan]{t['id']}[/cyan] [dim]{human_age(t['updated_at'])}[/dim]  {t['title']}")
            console.print()

        elif cmd == "/switch":
            if not rest:
                console.print("[yellow]usage: /switch <id|prefix|last>[/yellow]")
                continue
            try:
                thread = client.get_thread(rest)["thread"]
            except RuntimeError as exc:
                console.print(f"[red]{exc}[/red]")
                continue
            console.print(f"[dim]switched to[/dim] [bold cyan]{thread['title']}[/bold cyan]\n")

        elif cmd in ("/image", "/edit"):
            if not rest:
                console.print(f"[yellow]usage: {cmd} <prompt>[/yellow]")
                if cmd == "/edit":
                    console.print("[dim]edits the newest image in this conversation, "
                                  "or one queued with /attach[/dim]")
                continue
            # The server routes on the leading verb, so this is just a message.
            send(line)

        elif cmd == "/attach":
            if not rest:
                if pending:
                    console.print("[dim]queued for the next message:[/dim]")
                    for path in pending:
                        console.print(f"  [cyan]{path}[/cyan]")
                else:
                    console.print("[dim]nothing queued - /attach <path> to add an image[/dim]")
                console.print()
                continue
            candidate = Path(rest).expanduser()
            if not candidate.is_file():
                console.print(f"[red]no such file:[/red] {candidate}\n")
                continue
            pending.append(candidate)
            console.print(
                f"[dim]attached {candidate.name} "
                f"({len(pending)} queued) - it goes with your next message[/dim]\n"
            )

        elif cmd == "/detach":
            count = len(pending)
            pending.clear()
            console.print(f"[dim]cleared {count} attachment(s)[/dim]\n")

        elif cmd == "/think":
            if rest in ("on", "true", "1"):
                thinking_on = True
            elif rest in ("off", "false", "0"):
                thinking_on = False
            else:
                thinking_on = not thinking_on
            console.print(f"[dim]reasoning mode {'on' if thinking_on else 'off'}[/dim]\n")

        elif cmd == "/retry":
            if not last_user_message:
                console.print("[yellow]nothing to retry[/yellow]")
                continue
            send(last_user_message)

        elif cmd == "/title":
            if not rest:
                console.print("[yellow]usage: /title <text>[/yellow]")
                continue
            client.rename_thread(thread["id"], rest)
            thread["title"] = rest
            console.print("[dim]renamed[/dim]\n")

        elif cmd == "/show":
            for m in client.get_thread(thread["id"])["messages"]:
                _print_message(m)

        elif cmd == "/status":
            status()

        elif cmd == "/unload":
            res = client.unload(rest or "all")
            console.print(f"[dim]unloaded: {res['unloaded'] or 'nothing'}[/dim]\n")

        else:
            console.print(f"[yellow]unknown command {cmd}[/yellow] [dim](/help)[/dim]\n")

    client.close()
    console.print("[dim]bye[/dim]")
