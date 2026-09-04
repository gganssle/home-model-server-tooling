"""SQLite-backed conversation store.

Both the CLI and the web UI talk to the same server process, which owns this
database, so a thread started in the browser can be continued over SSH and
vice versa. WAL mode is on so reads never block the writer.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    -- Relative filename under the image directory when this message carries a
    -- generated or attached image.
    image       TEXT,
    -- JSON blob: model id, token counts, timings, generation params.
    meta        TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_thread
    ON messages(thread_id, created_at);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Message:
    id: str
    thread_id: str
    role: str
    content: str
    created_at: float
    image: str | None = None
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "image": self.image,
            "meta": self.meta or {},
        }


@dataclass
class Thread:
    id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
        }


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because the server touches this from the
        # request loop and from the generation worker thread; every call below
        # goes through a short-lived transaction so that is safe.
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._db.commit()

    # ---------------- threads ----------------

    def create_thread(self, title: str = "New conversation") -> Thread:
        now = time.time()
        tid = new_id("t")
        with self._db:
            self._db.execute(
                "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?,?,?,?)",
                (tid, title, now, now),
            )
        return Thread(id=tid, title=title, created_at=now, updated_at=now)

    def list_threads(self, limit: int = 200) -> list[Thread]:
        rows = self._db.execute(
            """
            SELECT t.*, COUNT(m.id) AS message_count
            FROM threads t LEFT JOIN messages m ON m.thread_id = t.id
            GROUP BY t.id ORDER BY t.updated_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            Thread(r["id"], r["title"], r["created_at"], r["updated_at"], r["message_count"])
            for r in rows
        ]

    def get_thread(self, thread_id: str) -> Thread | None:
        row = self._db.execute(
            """
            SELECT t.*, COUNT(m.id) AS message_count
            FROM threads t LEFT JOIN messages m ON m.thread_id = t.id
            WHERE t.id = ? GROUP BY t.id
            """,
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        return Thread(row["id"], row["title"], row["created_at"], row["updated_at"], row["message_count"])

    def resolve_thread(self, ref: str) -> Thread | None:
        """Look up a thread by full id, by unique id prefix, or by 'last'.

        Prefix matching is what makes the CLI tolerable to type: `hearth show
        t_9f2` rather than the full identifier.
        """
        if ref in ("last", "latest", "-"):
            threads = self.list_threads(limit=1)
            return threads[0] if threads else None
        exact = self.get_thread(ref)
        if exact:
            return exact
        rows = self._db.execute(
            "SELECT id FROM threads WHERE id LIKE ? LIMIT 2", (f"{ref}%",)
        ).fetchall()
        if len(rows) == 1:
            return self.get_thread(rows[0]["id"])
        return None

    def rename_thread(self, thread_id: str, title: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE threads SET title=?, updated_at=? WHERE id=?",
                (title, time.time(), thread_id),
            )

    def touch_thread(self, thread_id: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE threads SET updated_at=? WHERE id=?", (time.time(), thread_id)
            )

    def delete_thread(self, thread_id: str) -> bool:
        with self._db:
            cur = self._db.execute("DELETE FROM threads WHERE id=?", (thread_id,))
        return cur.rowcount > 0

    # ---------------- messages ----------------

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        image: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Message:
        now = time.time()
        mid = new_id("m")
        with self._db:
            self._db.execute(
                "INSERT INTO messages (id, thread_id, role, content, created_at, image, meta)"
                " VALUES (?,?,?,?,?,?,?)",
                (mid, thread_id, role, content, now, image, json.dumps(meta) if meta else None),
            )
            self._db.execute("UPDATE threads SET updated_at=? WHERE id=?", (now, thread_id))
        return Message(mid, thread_id, role, content, now, image, meta)

    def get_messages(self, thread_id: str, limit: int | None = None) -> list[Message]:
        sql = "SELECT * FROM messages WHERE thread_id=? ORDER BY created_at ASC"
        params: tuple[Any, ...] = (thread_id,)
        rows = self._db.execute(sql, params).fetchall()
        msgs = [
            Message(
                r["id"], r["thread_id"], r["role"], r["content"], r["created_at"],
                r["image"], json.loads(r["meta"]) if r["meta"] else None,
            )
            for r in rows
        ]
        if limit is not None and len(msgs) > limit:
            msgs = msgs[-limit:]
        return msgs

    def delete_message(self, message_id: str) -> bool:
        with self._db:
            cur = self._db.execute("DELETE FROM messages WHERE id=?", (message_id,))
        return cur.rowcount > 0

    def last_message(self, thread_id: str, role: str | None = None) -> Message | None:
        msgs = self.get_messages(thread_id)
        for m in reversed(msgs):
            if role is None or m.role == role:
                return m
        return None

    def search(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.execute(
            """
            SELECT m.id, m.thread_id, m.role, m.content, m.created_at, t.title
            FROM messages m JOIN threads t ON t.id = m.thread_id
            WHERE m.content LIKE ? ORDER BY m.created_at DESC LIMIT ?
            """,
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._db.close()
