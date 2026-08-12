from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

MAX_TOOL_CALLS = 5
DEFAULT_SESSION_TTL_SECONDS = 30 * 60

_PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?previous\s+instructions",
    r"(?i)ignore\s+(all\s+)?prior\s+instructions",
    r"(?i)reveal\s+(the\s+)?(system|developer)\s+prompt",
    r"(?i)override\s+(security|rules|constraints)",
    r"(?i)bypass\s+(security|validation|guardrails)",
    r"(?i)system\s+prompt",
    r"(?i)developer\s+prompt",
]


def sanitize_input(value: Any, max_len: int | None = None) -> str:
    """Reduce prompt-injection language and keep user text compact."""
    text = str(value or "").strip()
    if not text:
        return ""
    for pattern in _PROMPT_INJECTION_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_len is not None:
        text = text[:max_len]
    return text


def validate_search_request(
    *,
    query: str = "",
    category: str = "",
    price_min: float = 0.0,
    price_max: float = 1_000_000.0,
    min_rating: float = 0.0,
    limit: int = 10,
) -> dict[str, Any]:
    """Validate request fields before invoking the search/LLM stack."""
    cleaned_query = sanitize_input(query, 200)
    cleaned_category = sanitize_input(category, 100)
    if len(cleaned_query) > 200:
        raise ValueError("query must be <= 200 characters")
    if len(cleaned_category) > 100:
        raise ValueError("category must be <= 100 characters")
    if not 0.0 <= float(price_min) <= 1_000_000_000.0:
        raise ValueError("price_min must be >= 0")
    if not 0.0 <= float(price_max) <= 1_000_000_000.0:
        raise ValueError("price_max must be >= 0")
    if float(price_min) > float(price_max):
        raise ValueError("price_min must be <= price_max")
    if not 0.0 <= float(min_rating) <= 5.0:
        raise ValueError("rating must be between 0 and 5")
    if int(limit) < 1 or int(limit) > 20:
        raise ValueError("limit must be between 1 and 20")
    return {
        "query": cleaned_query,
        "category": cleaned_category,
        "price_min": float(price_min),
        "price_max": float(price_max),
        "min_rating": float(min_rating),
        "limit": int(limit),
    }


def ensure_safe_path(path: str | None) -> bool:
    """Reject traversalish and sensitive paths before any filesystem access."""
    if not path:
        return False
    candidate = str(path).strip()
    if candidate.startswith(("/", "~")):
        return False
    parts = PurePosixPath(candidate).parts
    if ".." in parts:
        return False
    sensitive = {".env", "SKILL.md", "etc", "passwd"}
    if any(token in candidate.lower() for token in [".env", "skill.md", "etc/passwd", ".."]):
        return False
    if any(part in sensitive for part in parts):
        return False
    return True


class SessionStore:
    """In-memory session store with short-lived TTL and no persistent profiles."""

    def __init__(self, ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create_session(self, data: dict[str, Any] | None = None) -> str:
        session_id = uuid.uuid4().hex
        payload = {"data": dict(data or {}), "expires_at": time.time() + self.ttl_seconds}
        with self._lock:
            self._store[session_id] = payload
        return session_id

    def get_session(self, session_id: str | None) -> dict[str, Any] | None:
        if not session_id:
            return None
        with self._lock:
            value = self._store.get(session_id)
            if not value:
                return None
            if time.time() > value["expires_at"]:
                self._store.pop(session_id, None)
                return None
            value["expires_at"] = time.time() + self.ttl_seconds
            return dict(value["data"])

    def update_session(self, session_id: str | None, data: dict[str, Any]) -> bool:
        if not session_id:
            return False
        with self._lock:
            if session_id not in self._store:
                return False
            self._store[session_id]["data"] = dict(data)
            self._store[session_id]["expires_at"] = time.time() + self.ttl_seconds
            return True


class AuditLogger:
    """Minimal non-PII execution audit trail backed by SQLite."""

    def __init__(self, db_path: str | Path = "backend/audit.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_events (
                    trace_id TEXT,
                    session_id TEXT,
                    timestamp TEXT,
                    status TEXT,
                    step TEXT,
                    latency_ms INTEGER,
                    error TEXT,
                    product_ids TEXT,
                    validation_result TEXT,
                    recommendation_result TEXT
                )
                """
            )
            conn.commit()

    def record_event(
        self,
        *,
        trace_id: str,
        session_id: str,
        status: str,
        step: str,
        latency_ms: int = 0,
        error: str | None = None,
        product_ids: list[int] | None = None,
        validation_result: str | None = None,
        recommendation_result: str | None = None,
    ) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO agent_events (
                        trace_id, session_id, timestamp, status, step, latency_ms,
                        error, product_ids, validation_result, recommendation_result
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace_id,
                        session_id,
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        status,
                        step,
                        int(latency_ms),
                        error or "",
                        json.dumps(product_ids or []),
                        validation_result or "",
                        recommendation_result or "",
                    ),
                )
                conn.commit()

    def create_trace(self, session_id: str) -> str:
        return f"trace-{uuid.uuid4().hex}"


session_store = SessionStore()
audit_logger = AuditLogger()


def safe_error_response(message: str, *, status_code: int = 400) -> dict[str, Any]:
    return {"detail": message, "status_code": status_code}
