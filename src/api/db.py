"""SQLite storage for posts (one short-lived connection per call; safe with background tasks)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    declared_species TEXT NOT NULL,          -- cat | dog (what the poster says)
    media_type TEXT NOT NULL,                -- image | video
    ext TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | published | review | rejected | error
    reasons TEXT NOT NULL DEFAULT '[]',      -- JSON list of reason codes
    detected_species TEXT,                   -- cat | dog (what the classifier says)
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_status ON posts(status);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def create_post(self, title: str, description: str, species: str, media_type: str,
                    ext: str) -> int:
        with closing(self._conn()) as c, c:
            cur = c.execute(
                "INSERT INTO posts (title, description, declared_species, media_type, ext, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (title, description, species, media_type, ext, datetime.now(UTC).isoformat()))
            return int(cur.lastrowid)

    def get(self, post_id: int) -> dict | None:
        with closing(self._conn()) as c:
            row = c.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
        return self._row(row) if row else None

    def set_moderation(self, post_id: int, status: str, reasons: list[str],
                       detected_species: str | None) -> None:
        with closing(self._conn()) as c, c:
            c.execute("UPDATE posts SET status = ?, reasons = ?, detected_species = ? "
                      "WHERE id = ?", (status, json.dumps(reasons), detected_species, post_id))

    def set_status_if(self, post_id: int, expected: str, new: str) -> bool:
        """Compare-and-set, so a post can only be resolved once."""
        with closing(self._conn()) as c, c:
            cur = c.execute("UPDATE posts SET status = ? WHERE id = ? AND status = ?",
                            (new, post_id, expected))
            return cur.rowcount == 1

    def list_posts(self, status: str, species: str | None = None, q: str | None = None,
                   limit: int = 100) -> list[dict]:
        sql, args = "SELECT * FROM posts WHERE status = ?", [status]
        if species:
            sql += " AND (detected_species = ? OR (detected_species IS NULL AND "\
                   "declared_species = ?))"
            args += [species, species]
        if q:
            esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            sql += " AND (title LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\')"
            args += [f"%{esc}%"] * 2
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with closing(self._conn()) as c:
            return [self._row(r) for r in c.execute(sql, args).fetchall()]

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["reasons"] = json.loads(d["reasons"])
        return d
