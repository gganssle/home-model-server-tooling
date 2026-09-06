"""Fetch a search result and reduce it to text.

Two jobs, and the smaller one is the extraction. The larger one is not
fetching things we should not: this daemon can be bound to a LAN address, and
under the autonomous tier the URL can originate from the model, which in turn
got it from a web page. A fetcher that will follow any URL it is handed turns
the assistant into a proxy for scanning the user's own network.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

# Anything else - PDFs, images, octet-stream - is not worth the tokens.
TEXTUAL = ("text/html", "application/xhtml+xml", "text/plain")

# Structural furniture that is never the thing you came to read.
SKIP_TAGS = frozenset({
    "script", "style", "noscript", "template", "svg", "iframe", "form",
    "nav", "header", "footer", "aside", "menu", "button", "select",
})
BLOCK_TAGS = frozenset({
    "p", "div", "br", "li", "tr", "section", "article", "blockquote", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table", "hr", "figcaption",
})


class UnsafeURL(ValueError):
    """The URL points somewhere we refuse to go."""


class FetchError(RuntimeError):
    """The page could not be retrieved, or was not text."""


@dataclass
class Page:
    url: str          # the URL actually fetched, after redirects
    title: str
    text: str


# ---------------------------------------------------------------- guards

def _address_is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def resolve_guarded(url: str, *, allow_private: bool = False) -> tuple[str, int, str]:
    """Validate a URL and resolve it to one specific address.

    Returns (host, port, ip). Raising rather than returning a flag keeps every
    caller honest: there is no way to fetch without going through this.

    The resolved address is handed back so the request can be pinned to it. A
    guard that only validates the name leaves the gap where DNS answers with a
    public address for the check and a private one microseconds later, when the
    HTTP client does its own independent lookup.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURL(f"refusing non-http(s) URL: {url}")
    if "@" in parsed.netloc:
        raise UnsafeURL("refusing URL with embedded credentials")
    host = parsed.hostname
    if not host:
        raise UnsafeURL(f"no host in URL: {url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURL(f"cannot resolve {host}: {exc}") from exc
    if not infos:
        raise UnsafeURL(f"cannot resolve {host}")

    addresses = [info[4][0] for info in infos]
    if not allow_private:
        # Every answer must be public, not merely the first: a host that
        # resolves to both a public and a private address is a rebinding
        # attempt, not a coincidence.
        for ip in addresses:
            if not _address_is_public(ip):
                raise UnsafeURL(f"refusing {host}: resolves to non-public address {ip}")
    return host, port, addresses[0]


def _pin(url: str, ip: str) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Rewrite a URL to hit one specific address, preserving name-based routing.

    The Host header keeps virtual hosting working and `sni_hostname` keeps TLS
    verifying against the name the user asked for, not the address we pinned.
    """
    parsed = urlsplit(url)
    literal = f"[{ip}]" if ":" in ip else ip
    if parsed.port:
        literal = f"{literal}:{parsed.port}"
    pinned = urlunsplit((parsed.scheme, literal, parsed.path or "/", parsed.query, ""))
    headers = {"Host": parsed.netloc}
    extensions: dict[str, Any] = {}
    if parsed.scheme == "https":
        extensions["sni_hostname"] = parsed.hostname
    return pinned, headers, extensions


# ---------------------------------------------------------------- extraction

class _TextExtractor(HTMLParser):
    """Strip a page down to the prose, without pulling in a parser library.

    Deliberately crude. `trafilatura` does this properly and is used instead
    when it happens to be installed, but the base install already carries mlx
    and mflux and should not also require lxml for a nice-to-have.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        self.parts.append(data)


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract(html: str) -> tuple[str, str]:
    """Return (title, text) for a page of HTML."""
    try:
        import trafilatura  # noqa: PLC0415
    except ImportError:
        pass
    else:
        body = trafilatura.extract(html, include_comments=False, include_tables=True)
        if body:
            meta_title = ""
            parser = _TextExtractor()
            try:
                parser.feed(html)
                meta_title = parser.title
            except Exception:
                pass
            return _tidy(meta_title), _tidy(body)

    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # Malformed markup is the norm, not the exception; keep whatever the
        # parser managed to collect before it gave up.
        pass
    return _tidy(parser.title), _tidy("".join(parser.parts))


# ---------------------------------------------------------------- fetching

def fetch(
    url: str,
    *,
    timeout: float = 10.0,
    max_bytes: int = 2_000_000,
    allow_private: bool = False,
    user_agent: str = "hearth/0.1",
    max_redirects: int = 4,
    cancel: Any = None,
) -> Page:
    """GET a page and return its extracted text.

    Redirects are followed by hand so that every hop is re-checked: a public
    URL that 302s to 169.254.169.254 is the oldest trick there is.
    """
    current = url
    with httpx.Client(follow_redirects=False, timeout=timeout) as client:
        for _ in range(max_redirects + 1):
            if cancel is not None and cancel.is_set():
                raise FetchError("cancelled")

            _, _, ip = resolve_guarded(current, allow_private=allow_private)
            pinned, headers, extensions = _pin(current, ip)
            headers = {
                **headers,
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
                "Accept-Encoding": "gzip, deflate",
            }

            try:
                with client.stream(
                    "GET", pinned, headers=headers, extensions=extensions
                ) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            raise FetchError(f"{current} redirected with no destination")
                        current = urljoin(current, location)
                        continue

                    if resp.status_code >= 400:
                        raise FetchError(f"{current} returned {resp.status_code}")

                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype and not any(ctype.startswith(t) for t in TEXTUAL):
                        raise FetchError(f"{current} is {ctype}, not a text document")

                    chunks: list[bytes] = []
                    total = 0
                    for chunk in resp.iter_bytes():
                        if cancel is not None and cancel.is_set():
                            raise FetchError("cancelled")
                        total += len(chunk)
                        if total > max_bytes:
                            # Take the prefix rather than failing: the top of a
                            # page is usually the part worth reading anyway.
                            chunks.append(chunk)
                            break
                        chunks.append(chunk)
                    body = b"".join(chunks)[:max_bytes]
                    encoding = resp.encoding or "utf-8"
            except httpx.HTTPError as exc:
                raise FetchError(f"{current}: {exc}") from exc

            html = body.decode(encoding, errors="replace")
            title, text = extract(html)
            return Page(url=current, title=title, text=text)

    raise FetchError(f"{url}: too many redirects")
