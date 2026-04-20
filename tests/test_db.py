"""
Unit tests for app/storage/db.py — session and message CRUD.

Uses an isolated in-memory SQLite database for every test via monkeypatch,
so no real data/chat.db is touched and tests run fully offline.
"""
import sqlite3
import pytest


# ---------------------------------------------------------------------------
# Fixture — patch DB_PATH to an in-memory SQLite DB for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """
    Replace the on-disk chat.db path with a fresh temp-directory DB
    for each test. autouse=True means every test in this module gets it.
    """
    db_file = str(tmp_path / "test_chat.db")
    import app.storage.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", db_file)
    db_mod.init_db()   # create tables in the temp DB
    return db_file


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:

    def test_tables_exist_after_init(self, tmp_path, monkeypatch):
        import app.storage.db as db_mod
        conn = sqlite3.connect(db_mod.DB_PATH)
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "sessions" in tables
        assert "messages" in tables

    def test_init_is_idempotent(self):
        """Calling init_db() twice must not raise."""
        import app.storage.db as db_mod
        db_mod.init_db()  # second call
        # no assertion needed — just must not raise


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

class TestSessionCrud:

    def test_create_session_returns_id_and_created_at(self):
        import app.storage.db as db_mod
        session = db_mod.create_session()
        assert "id" in session
        assert "created_at" in session
        assert len(session["id"]) == 36   # UUID4 format

    def test_create_session_persists_to_db(self):
        import app.storage.db as db_mod
        session = db_mod.create_session()
        sessions = db_mod.get_sessions()
        ids = [s["id"] for s in sessions]
        assert session["id"] in ids

    def test_get_sessions_returns_newest_first(self):
        import app.storage.db as db_mod
        s1 = db_mod.create_session()
        s2 = db_mod.create_session()
        sessions = db_mod.get_sessions()
        # s2 was created last, should appear first (ORDER BY created_at DESC)
        assert sessions[0]["id"] == s2["id"]
        assert sessions[1]["id"] == s1["id"]

    def test_get_sessions_empty_when_none_exist(self):
        import app.storage.db as db_mod
        assert db_mod.get_sessions() == []

    def test_delete_session_returns_true_when_found(self):
        import app.storage.db as db_mod
        session = db_mod.create_session()
        result = db_mod.delete_session(session["id"])
        assert result is True

    def test_delete_session_returns_false_when_not_found(self):
        import app.storage.db as db_mod
        result = db_mod.delete_session("00000000-0000-0000-0000-000000000000")
        assert result is False

    def test_delete_session_removes_it_from_list(self):
        import app.storage.db as db_mod
        session = db_mod.create_session()
        db_mod.delete_session(session["id"])
        ids = [s["id"] for s in db_mod.get_sessions()]
        assert session["id"] not in ids

    def test_session_exists_true_for_existing(self):
        import app.storage.db as db_mod
        s = db_mod.create_session()
        assert db_mod.session_exists(s["id"]) is True

    def test_session_exists_false_for_missing(self):
        import app.storage.db as db_mod
        assert db_mod.session_exists("no-such-id") is False


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------

class TestMessageCrud:

    def test_save_message_returns_correct_fields(self):
        import app.storage.db as db_mod
        s = db_mod.create_session()
        msg = db_mod.save_message(s["id"], "user", "Hello")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"
        assert msg["session_id"] == s["id"]
        assert "id" in msg
        assert "created_at" in msg

    def test_save_multiple_messages_all_persisted(self):
        import app.storage.db as db_mod
        s = db_mod.create_session()
        db_mod.save_message(s["id"], "user", "Question 1")
        db_mod.save_message(s["id"], "assistant", "Answer 1")
        db_mod.save_message(s["id"], "user", "Question 2")
        msgs = db_mod.get_messages(s["id"], limit=10)
        assert len(msgs) == 3

    def test_get_messages_returns_chronological_order(self):
        import app.storage.db as db_mod
        s = db_mod.create_session()
        db_mod.save_message(s["id"], "user", "first")
        db_mod.save_message(s["id"], "assistant", "second")
        msgs = db_mod.get_messages(s["id"], limit=10)
        assert msgs[0]["content"] == "first"
        assert msgs[1]["content"] == "second"

    def test_get_messages_respects_limit(self):
        import app.storage.db as db_mod
        s = db_mod.create_session()
        for i in range(10):
            db_mod.save_message(s["id"], "user", f"msg {i}")
        msgs = db_mod.get_messages(s["id"], limit=6)
        assert len(msgs) == 6

    def test_get_messages_limit_returns_most_recent(self):
        """With limit=2 on 4 messages, we should get the last 2 (most recent)."""
        import app.storage.db as db_mod
        s = db_mod.create_session()
        for i in range(4):
            db_mod.save_message(s["id"], "user", f"msg {i}")
        msgs = db_mod.get_messages(s["id"], limit=2)
        # The last two messages are msg 2 and msg 3
        contents = [m["content"] for m in msgs]
        assert "msg 2" in contents
        assert "msg 3" in contents

    def test_get_messages_empty_for_new_session(self):
        import app.storage.db as db_mod
        s = db_mod.create_session()
        assert db_mod.get_messages(s["id"]) == []

    def test_delete_session_cascades_to_messages(self):
        """Deleting a session must also delete all its messages (FK CASCADE)."""
        import app.storage.db as db_mod
        s = db_mod.create_session()
        db_mod.save_message(s["id"], "user", "should be deleted")
        db_mod.delete_session(s["id"])
        # Messages for the deleted session must be gone
        msgs = db_mod.get_messages(s["id"])
        assert msgs == []

    def test_messages_from_different_sessions_are_isolated(self):
        import app.storage.db as db_mod
        s1 = db_mod.create_session()
        s2 = db_mod.create_session()
        db_mod.save_message(s1["id"], "user", "session1 msg")
        db_mod.save_message(s2["id"], "user", "session2 msg")
        msgs1 = db_mod.get_messages(s1["id"], limit=10)
        msgs2 = db_mod.get_messages(s2["id"], limit=10)
        assert len(msgs1) == 1
        assert len(msgs2) == 1
        assert msgs1[0]["content"] == "session1 msg"
        assert msgs2[0]["content"] == "session2 msg"
