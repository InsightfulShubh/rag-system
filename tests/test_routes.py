"""
Unit tests for HTTP routes. Tests endpoint logic without mocking.

Non-LLM dependent tests only (health check, session CRUD).
LLM-dependent tests (query, chat messages) are in test_integration.py.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage import db


client = TestClient(app)


class TestHealthCheck:

    def test_health_returns_200_ok(self):
        """Health check endpoint returns 200 with status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestSessionEndpoints:

    def test_create_session_returns_201(self):
        """POST /api/sessions — create new session. Returns 201 Created."""
        response = client.post("/api/sessions")

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert "created_at" in data
        assert isinstance(data["id"], int)
        assert data["id"] > 0

    def test_get_sessions_returns_list(self):
        """GET /api/sessions — list all sessions."""
        # Create a session first
        client.post("/api/sessions")

        response = client.get("/api/sessions")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should have at least the one we just created
        assert len(data) >= 1
        assert all("id" in item and "created_at" in item for item in data)

    def test_delete_session_returns_204_when_found(self):
        """DELETE /api/sessions/{id} — delete session. Returns 204 No Content."""
        # Create a session
        create_resp = client.post("/api/sessions")
        session_id = create_resp.json()["id"]

        # Delete it
        response = client.delete(f"/api/sessions/{session_id}")

        assert response.status_code == 204

        # Verify it's gone
        get_resp = client.get("/api/sessions")
        assert session_id not in [s["id"] for s in get_resp.json()]

    def test_delete_session_returns_404_when_not_found(self):
        """DELETE /api/sessions/{id} — return 404 if session doesn't exist."""
        response = client.delete("/api/sessions/99999")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_send_message_returns_404_if_session_not_found(self):
        """POST /api/sessions/{id}/messages — return 404 if session doesn't exist."""
        response = client.post(
            "/api/sessions/99999/messages",
            json={"message": "test"},
        )

        assert response.status_code == 404

    def test_get_messages_returns_404_if_session_not_found(self):
        """GET /api/sessions/{id}/messages — return 404 if session doesn't exist."""
        response = client.get("/api/sessions/99999/messages")

        assert response.status_code == 404

    def test_get_messages_empty_for_new_session(self):
        """GET /api/sessions/{id}/messages — new session has no messages."""
        # Create a session
        create_resp = client.post("/api/sessions")
        session_id = create_resp.json()["id"]

        # Get messages
        response = client.get(f"/api/sessions/{session_id}/messages")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
