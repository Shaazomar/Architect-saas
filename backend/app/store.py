"""Durable job store (SQLite, WAL). Swap for Postgres by reimplementing this
module behind the same five functions; nothing else in the app touches SQL.

Job artifacts (uploaded image, exported GLB) live on disk under
``data/jobs/<uuid>/`` — ids are server-generated UUIDs, never user input, so
no user-controlled value ever reaches a filesystem path.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

VALID_STATUSES = ("queued", "processing", "done", "failed")


@dataclass
class Job:
    id: str
    status: str
    created_at: float
    error: str | None
    result: dict[str, Any] | None

    @property
    def dir(self) -> Path:
        return settings.data_dir / "jobs" / self.id


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(settings.data_dir / "jobs.db", check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                error TEXT,
                result TEXT
            )"""
        )
        _conn.commit()
    return _conn


def create_job() -> Job:
    job = Job(id=str(uuid.uuid4()), status="queued", created_at=time.time(), error=None, result=None)
    with _lock:
        _db().execute(
            "INSERT INTO jobs (id, status, created_at) VALUES (?, ?, ?)",
            (job.id, job.status, job.created_at),
        )
        _db().commit()
    job.dir.mkdir(parents=True, exist_ok=True)
    return job


def update_job(job_id: str, status: str, error: str | None = None, result: dict[str, Any] | None = None) -> None:
    assert status in VALID_STATUSES
    with _lock:
        _db().execute(
            "UPDATE jobs SET status=?, error=?, result=? WHERE id=?",
            (status, error, json.dumps(result) if result is not None else None, job_id),
        )
        _db().commit()


def get_job(job_id: str) -> Job | None:
    with _lock:
        row = _db().execute(
            "SELECT id, status, created_at, error, result FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    return Job(
        id=row[0],
        status=row[1],
        created_at=row[2],
        error=row[3],
        result=json.loads(row[4]) if row[4] else None,
    )


def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
