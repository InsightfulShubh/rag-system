import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "chat.db")


def _get_conn() -> sqlite3.Connection:
    """Open a SQLite connection with row_factory and foreign key enforcement."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they don't already exist. Called once at startup."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
        """)
        conn.commit()
    finally:
        conn.close()


def create_session() -> dict:
    """Insert a new session row and return it as a dict."""
    created_at = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "INSERT INTO sessions (created_at) VALUES (?)",
            (created_at,),
        )
        conn.commit()
        session_id = cursor.lastrowid
    finally:
        conn.close()
    return {"id": session_id, "created_at": created_at}


def get_sessions() -> list[dict]:
    """Return all sessions ordered by most recent first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def delete_session(session_id: int) -> bool:
    """Delete a session and its messages. Returns True if found, False if not."""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def save_message(session_id: int, role: str, content: str) -> dict:
    """Insert a message row and return it as a dict."""
    created_at = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, created_at),
        )
        conn.commit()
        message_id = cursor.lastrowid
    finally:
        conn.close()
    return {
        "id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": created_at,
    }


def get_messages(session_id: int, limit: int = 6) -> list[dict]:
    """Return the last `limit` messages in chronological order (oldest first)."""
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
        # Reverse to get chronological (oldest-first) order for LLM context
        return [dict(row) for row in reversed(rows)]
    finally:
        conn.close()


def session_exists(session_id: int) -> bool:
    """Return True if the session_id exists in the database."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()
