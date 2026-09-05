"""Web retrieval for hearth.

This package is a sibling of `engine/`, not a part of it, and the distinction
is load-bearing: nothing here touches a model, and none of it may ever run on
the manager's worker thread. That thread is effectively the GPU lock, and a ten
second fetch taken on it stalls every queued generation behind it for no
reason at all.

The call sites run this from Starlette's threadpool, inside the streaming
response, which is off the event loop and off the worker both.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from hearth.config import SearchConfig
from hearth.search import budget
from hearth.search.fetch import FetchError, Page, UnsafeURL, fetch
from hearth.search.heuristics import should_search
from hearth.search.providers import Provider, Result, SearchError, build_provider

log = logging.getLogger("hearth.search")

__all__ = [
    "Document", "Outcome", "WebSearch", "SearchError", "UnsafeURL", "FetchError",
    "Page", "Result", "Provider", "should_search", "budget", "TOOL_SCHEMA",
]

# Offered to the model under search.autonomous = "tool". The description is the
# policy: it is the only place the model is told when calling this is right.
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web and read the top results. Use this for anything that "
            "happened after your training data ends, anything that changes over "
            "time (prices, versions, weather, who currently holds a position), and "
            "anything you would otherwise have to guess at. Do not use it for "
            "reasoning, writing, or questions about material the user has already "
            "given you."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A standalone search query. Do not use pronouns "
                                   "referring to earlier turns; spell the subject out.",
                }
            },
            "required": ["query"],
        },
    },
}


@dataclass
class Document:
    id: int
    title: str
    url: str
    snippet: str = ""
    text: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "url": self.url,
            "snippet": self.snippet, "text": self.text, "error": self.error,
        }


@dataclass
class Outcome:
    """Everything one retrieval produced, ready to store on a message."""

    query: str
    results: list[Result] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    error: str | None = None
    cancelled: bool = False

    def abandon(self) -> "Outcome":
        """Drop the documents, keeping the record that a search was attempted.

        Called when the user cancels. Whatever was fetched by then is partial
        and unasked-for, and carrying it forward would let a cancelled lookup
        quietly steer every later turn in the thread. The compact line stays,
        so the transcript still says what happened.
        """
        self.cancelled = True
        self.documents = []
        return self

    @property
    def usable(self) -> bool:
        return any(d.text.strip() for d in self.documents)

    def to_meta(self) -> dict[str, Any]:
        """The structured record stored under message.meta['search'].

        The full page text lives here and not in message.content, so that the
        transcript stays readable and so that the history builder can choose,
        per turn, whether to rehydrate it or fall back to the compact line.
        """
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "documents": [d.to_dict() for d in self.documents],
            "error": self.error,
        }

    def compact(self) -> str:
        if self.cancelled:
            return f'[web search for "{self.query}" was cancelled]'
        if self.error:
            return f'[web search for "{self.query}" failed: {self.error}]'
        return budget.compact_line(self.query, [r.to_dict() for r in self.results])

    def sources(self) -> list[dict[str, Any]]:
        return [{"id": d.id, "title": d.title, "url": d.url} for d in self.documents]


Emit = Callable[[dict[str, Any]], None]


class WebSearch:
    """Query a provider, read the top hits, hand back documents."""

    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg
        self._provider: Provider | None = None
        self._provider_error: str | None = None
        try:
            self._provider = build_provider(cfg)
        except SearchError as exc:
            # A bad provider setting must not stop the server from booting; it
            # surfaces when someone actually tries to search.
            self._provider_error = str(exc)
            log.warning("web search disabled: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._provider is not None

    @property
    def unavailable_reason(self) -> str:
        if self._provider_error:
            return self._provider_error
        if not self.cfg.enabled:
            return "web search is off (set search.enabled, or HEARTH_SEARCH=1)"
        return "no search provider is configured (search.provider)"

    def wants_search(self, text: str) -> tuple[bool, str]:
        """Should this message be searched, with nothing else to go on?"""
        if not self.enabled:
            return False, "search unavailable"
        if self.cfg.autonomous != "heuristic":
            return False, f"autonomous={self.cfg.autonomous}"
        return should_search(text)

    def run(
        self,
        query: str,
        emit: Emit | None = None,
        cancel: threading.Event | None = None,
        reason: str | None = None,
    ) -> Outcome:
        """Search, fetch and extract. Never raises; failures land in `error`.

        A failed search should cost the user a note in the transcript, not the
        whole turn - the model can still answer, it just answers from memory.
        """
        query = " ".join((query or "").split())[:400]
        outcome = Outcome(query=query)
        if not self.enabled:
            outcome.error = self.unavailable_reason
            return outcome
        if not query:
            outcome.error = "empty search query"
            return outcome

        def say(event: dict[str, Any]) -> None:
            if emit is not None:
                emit({"type": "search", **event})

        # The reason travels with the event so that "heuristic" mode is
        # inspectable from the client rather than only from the server log.
        say({"phase": "querying", "query": query, "reason": reason})
        try:
            results = self._provider.search(query, self.cfg.max_results)
        except SearchError as exc:
            outcome.error = str(exc)
            say({"phase": "error", "error": outcome.error})
            return outcome

        outcome.results = results
        say({"phase": "results", "results": [r.to_dict() for r in results]})
        if not results:
            return outcome

        if cancel is not None and cancel.is_set():
            return outcome.abandon()

        targets = results[: max(0, self.cfg.max_fetch)]
        say({"phase": "fetching", "urls": [r.url for r in targets]})
        pages = self._fetch_all(targets, cancel)

        for index, (result, page) in enumerate(zip(targets, pages), start=1):
            doc = Document(
                id=index, title=result.title, url=result.url, snippet=result.snippet
            )
            if isinstance(page, Page):
                doc.title = page.title or result.title
                doc.url = page.url
                doc.text = page.text
            elif isinstance(page, Exception):
                doc.error = str(page)
                # The snippet is a poor substitute for the page, but it is real
                # text from the provider and better than dropping the source.
                doc.text = result.snippet
            outcome.documents.append(doc)

        if cancel is not None and cancel.is_set():
            return outcome.abandon()

        say({"phase": "ready", "sources": outcome.sources()})
        return outcome

    def _fetch_all(
        self, targets: list[Result], cancel: threading.Event | None
    ) -> list[Page | Exception]:
        """Fetch in parallel, preserving order.

        Three pages at a second each is three seconds serially and one in
        parallel, and this is plain socket waiting on a threadpool thread -
        none of it contends with the GPU.
        """
        if not targets:
            return []

        def one(result: Result) -> Page | Exception:
            try:
                return fetch(
                    result.url,
                    timeout=self.cfg.timeout_s,
                    max_bytes=self.cfg.max_page_bytes,
                    allow_private=self.cfg.allow_private_hosts,
                    user_agent=self.cfg.user_agent,
                    cancel=cancel,
                )
            except (UnsafeURL, FetchError) as exc:
                log.info("skipping %s: %s", result.url, exc)
                return exc
            except Exception as exc:  # noqa: BLE001 - one bad page is not fatal
                log.warning("unexpected error fetching %s: %s", result.url, exc)
                return exc

        if len(targets) == 1:
            return [one(targets[0])]
        with ThreadPoolExecutor(max_workers=len(targets)) as pool:
            return list(pool.map(one, targets))

    def pack(self, outcome: Outcome) -> str:
        """Render an outcome into the block that goes in front of the model."""
        return budget.pack(
            [d.to_dict() for d in outcome.documents],
            self.cfg.max_context_chars,
            outcome.query,
        )
