"""Search backends, behind one small interface.

Only the query goes to a provider, and with SearXNG that provider can be a
container on the same machine - which is the point. Whatever the backend, the
result is the same normalised list, so nothing above this file knows or cares
which one is configured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from hearth.config import SearchConfig

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class SearchError(RuntimeError):
    """The provider could not be reached, or answered with something unusable."""


@dataclass
class Result:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class Provider(Protocol):
    name: str

    def search(self, query: str, count: int) -> list[Result]: ...


def _clean(value: object) -> str:
    """Providers are inconsistent about nulls and stray markup in titles."""
    import re

    text = str(value or "").strip()
    return re.sub(r"<[^>]+>", "", text)


def _clean_url(value: object) -> str:
    """Return the URL only if it is one we would be willing to open.

    A SearXNG instance is a piece of software the user runs, but it is also the
    one component here that speaks to the open internet, and a compromised or
    merely buggy one should not be able to hand a `javascript:` URL to the web
    UI or a `file:` URL to the fetcher.
    """
    url = _clean(value)
    return url if url.lower().startswith(("http://", "https://")) else ""


class SearxngProvider:
    """A SearXNG instance's JSON API.

    Note that the configured instance is *not* subject to the private-address
    guard that fetching applies: pointing this at 127.0.0.1 is the recommended
    setup, and the user chose the address explicitly. Result URLs the instance
    hands back are a different matter, and are guarded like any other.
    """

    name = "searxng"

    def __init__(self, base_url: str, timeout: float, user_agent: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent

    def search(self, query: str, count: int) -> list[Result]:
        try:
            resp = httpx.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json"},
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise SearchError(
                f"cannot reach the SearXNG instance at {self.base_url}: {exc}"
            ) from exc

        if resp.status_code == 403:
            raise SearchError(
                f"{self.base_url} refused the request. A stock SearXNG only serves "
                "HTML; add 'json' to the formats list in its settings.yml."
            )
        if resp.status_code >= 400:
            raise SearchError(f"{self.base_url} returned {resp.status_code}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise SearchError(f"{self.base_url} did not return JSON") from exc

        out: list[Result] = []
        for item in payload.get("results", []):
            url = _clean_url(item.get("url"))
            if not url:
                continue
            out.append(Result(
                title=_clean(item.get("title")) or url,
                url=url,
                snippet=_clean(item.get("content")),
            ))
            if len(out) >= count:
                break
        return out


class BraveProvider:
    """Brave's Web Search API, for people who would rather not self-host."""

    name = "brave"

    def __init__(self, api_key: str, timeout: float, user_agent: str) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.user_agent = user_agent

    def search(self, query: str, count: int) -> list[Result]:
        try:
            resp = httpx.get(
                BRAVE_ENDPOINT,
                params={"q": query, "count": count},
                headers={
                    "X-Subscription-Token": self.api_key,
                    "Accept": "application/json",
                    "User-Agent": self.user_agent,
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise SearchError(f"cannot reach the Brave search API: {exc}") from exc

        if resp.status_code in (401, 403):
            raise SearchError("Brave rejected the API key (search.brave_api_key)")
        if resp.status_code == 429:
            raise SearchError("Brave rate-limited the request")
        if resp.status_code >= 400:
            raise SearchError(f"Brave returned {resp.status_code}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise SearchError("Brave did not return JSON") from exc

        out: list[Result] = []
        for item in (payload.get("web") or {}).get("results", []):
            url = _clean_url(item.get("url"))
            if not url:
                continue
            out.append(Result(
                title=_clean(item.get("title")) or url,
                url=url,
                snippet=_clean(item.get("description")),
            ))
            if len(out) >= count:
                break
        return out


def build_provider(cfg: SearchConfig) -> Provider | None:
    """Construct the configured provider, or None when search is switched off.

    Misconfiguration raises here rather than at the first query, so the failure
    names the setting instead of surfacing as an empty result set.
    """
    provider = (cfg.provider or "none").strip().lower()
    if not cfg.enabled or provider in ("none", ""):
        return None
    if provider == "searxng":
        if not cfg.searxng_url:
            raise SearchError("search.provider is 'searxng' but search.searxng_url is empty")
        return SearxngProvider(cfg.searxng_url, cfg.timeout_s, cfg.user_agent)
    if provider == "brave":
        if not cfg.brave_api_key:
            raise SearchError(
                "search.provider is 'brave' but no API key is set "
                "(search.brave_api_key, or HEARTH_BRAVE_KEY)"
            )
        return BraveProvider(cfg.brave_api_key, cfg.timeout_s, cfg.user_agent)
    raise SearchError(f"unknown search provider {cfg.provider!r} (searxng, brave, none)")
