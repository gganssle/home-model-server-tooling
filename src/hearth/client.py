"""Thin HTTP client used by the CLI.

Keeping this separate from the CLI means an SSH session never imports mlx or
touches a model; it just speaks HTTP to the box running the daemon.
"""
from __future__ import annotations

import json
from typing import Any, Iterator

import httpx


class ServerUnavailable(RuntimeError):
    pass


class HearthClient:
    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HearthClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ServerUnavailable(
                f"cannot reach the hearth server at {self.base_url}.\n"
                f"Start it with:  hearth serve\n"
                f"(or point elsewhere with HEARTH_HOST / HEARTH_PORT)"
            ) from exc
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise RuntimeError(f"{resp.status_code}: {detail}")
        return resp.json()

    def _stream(self, method: str, path: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        try:
            with self._client.stream(method, path, **kwargs) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    detail = resp.text
                    try:
                        detail = json.loads(detail).get("detail", detail)
                    except Exception:
                        pass
                    raise RuntimeError(f"{resp.status_code}: {detail}")
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        return
                    yield json.loads(payload)
        except httpx.ConnectError as exc:
            raise ServerUnavailable(
                f"cannot reach the hearth server at {self.base_url}.\n"
                f"Start it with:  hearth serve"
            ) from exc

    # ---------------- api ----------------

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/api/status")

    def list_threads(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/threads")["threads"]

    def create_thread(self, title: str = "New conversation") -> dict[str, Any]:
        return self._request("POST", "/api/threads", json={"title": title})

    def get_thread(self, ref: str) -> dict[str, Any]:
        return self._request("GET", f"/api/threads/{ref}")

    def rename_thread(self, ref: str, title: str) -> dict[str, Any]:
        return self._request("PATCH", f"/api/threads/{ref}", json={"title": title})

    def delete_thread(self, ref: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/threads/{ref}")

    def search(self, query: str) -> list[dict[str, Any]]:
        return self._request("GET", "/api/search", params={"q": query})["results"]

    def cancel(self) -> dict[str, Any]:
        return self._request("POST", "/api/cancel")

    def preload(self, which: str) -> dict[str, Any]:
        return self._request("POST", f"/api/models/{which}/preload")

    def unload(self, which: str = "all") -> dict[str, Any]:
        return self._request("POST", "/api/models/unload", params={"which": which})

    def send(
        self,
        ref: str,
        content: str,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        images: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        body: dict[str, Any] = {"content": content}
        for key, value in (
            ("thinking", thinking), ("max_tokens", max_tokens),
            ("temperature", temperature), ("images", images),
        ):
            if value is not None:
                body[key] = value
        return self._stream("POST", f"/api/threads/{ref}/messages", json=body)

    def image(self, prompt: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        body = {"prompt": prompt, **{k: v for k, v in kwargs.items() if v is not None}}
        return self._stream("POST", "/api/images", json=body)

    def download_image(self, filename: str) -> bytes:
        resp = self._client.get(f"/api/images/{filename}")
        resp.raise_for_status()
        return resp.content
