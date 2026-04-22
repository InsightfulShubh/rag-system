from app.storage import db


class SessionService:
    """
    Business logic for session management.
    Thin layer over db.py — no SQL here, just orchestration.
    Routes stay even thinner by calling this instead of db directly.
    """

    def create_session(self) -> dict:
        """Create a new conversation session."""
        return db.create_session()

    def get_sessions(self) -> list[dict]:
        """Return all sessions, newest first."""
        return db.get_sessions()

    def delete_session(self, session_id: int) -> bool:
        """
        Delete a session and all its messages.
        Returns True if deleted, False if session not found.
        """
        return db.delete_session(session_id)