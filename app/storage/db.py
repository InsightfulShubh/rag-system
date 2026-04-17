import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

# Database file stored in data/ directory
DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "chat.db")


def _get_conn() -> sqlite3.Connection:
    """
    Open a new SQLite connection with row_factory so rows behave like dicts.
    Each function call opens and closes its own connection (simple, stateless).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows row["column"] access
    return conn


def init_db() -> None:
    """
    Create tables if they don't already exist.
    Called once at application startup from main.py.

    Tables:
        sessions  — one row per conversation
        messages  — one row per message, linked to a session
    """
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,       -- 'user' or 'assistant'
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
        """)
        conn.commit()
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Session functions
# ──────────────────────────────────────────────

def create_session() -> dict:
    """
    Insert a new session row and return it.

    Returns:
        dict with keys: id, created_at
    """
    session_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, created_at),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": session_id, "created_at": created_at}


def get_sessions() -> list[dict]:
    """
    Return all sessions ordered by most recent first.

    Returns:
        list of dicts with keys: id, created_at
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def delete_session(session_id: str) -> bool:
    """
    Delete a session and all its messages (CASCADE handles messages).

    Returns:
        True if a session was deleted, False if session_id not found
    """
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Message functions
# ──────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str) -> dict:
    """
    Insert a single message row and return it.

    Args:
        session_id: the session this message belongs to
        role:       'user' or 'assistant'
        content:    message text

    Returns:
        dict with keys: id, session_id, role, content, created_at
    """
    message_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (message_id, session_id, role, content, created_at),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": created_at,
    }


def get_messages(session_id: str, limit: int = 6) -> list[dict]:
    """
    Load the last `limit` messages for a session (HISTORY_LIMIT = 6).
    Returned in chronological order (oldest first) so they slot directly
    into the LLM messages list.

    Args:
        session_id: session to load history for
        limit:      max messages to return (default 6 per requirements)

    Returns:
        list of dicts with keys: id, session_id, role, content, created_at
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, session_id, role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        # Reverse so oldest message comes first (correct order for LLM)
        return [dict(row) for row in reversed(rows)]
    finally:
        conn.close()


def session_exists(session_id: str) -> bool:
    """
    Check whether a session_id exists in the database.
    Used by routes to return 404 cleanly.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()
