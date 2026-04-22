"""
Integration tests — real LLM and embedding API calls. No mocking.

Credentials are read from .env (loaded automatically via app.config).
All tests are skipped gracefully when no API key is configured.

Cost: minimal — one embedding call + one LLM call per test run.
"""
import pytest
from app.config import settings


# ---------------------------------------------------------------------------
# Module-level skip guard
# ---------------------------------------------------------------------------

def _api_key_available() -> bool:
    """Return True if at least one provider key is set in the environment."""
    if settings.use_azure_openai:
        return bool(settings.azure_api_key)
    return bool(settings.openai_api_key)


# Applied to every test in this file via pytestmark
pytestmark = pytest.mark.skipif(
    not _api_key_available(),
    reason=(
        "No API key found in .env. "
        "Set OPENAI_API_KEY (or AZURE_API_KEY + USE_AZURE_OPENAI=true) to run integration tests."
    ),
)


# ---------------------------------------------------------------------------
# Embedding API
# ---------------------------------------------------------------------------

class TestEmbeddingApi:

    def test_get_embedding_returns_list_of_floats(self):
        """Real API call — verify embedding is returned as a non-empty list of floats."""
        from app.utils.embedding import get_embedding
        result = get_embedding("test")
        assert isinstance(result, list), "embedding must be a list"
        assert len(result) > 0, "embedding must be non-empty"
        assert all(isinstance(v, float) for v in result), "all values must be floats"

    def test_get_embedding_returns_correct_dimension(self):
        """text-embedding-3-small returns 1536-dimensional vectors."""
        from app.utils.embedding import get_embedding
        result = get_embedding("hello")
        assert len(result) == 1536, f"expected 1536 dims, got {len(result)}"

    def test_different_texts_produce_different_embeddings(self):
        """Semantically different texts must produce different vectors."""
        from app.utils.embedding import get_embedding
        emb_a = get_embedding("machine learning")
        emb_b = get_embedding("ancient history")
        assert emb_a != emb_b, "distinct texts must produce distinct embeddings"


# ---------------------------------------------------------------------------
# search() — retrieval without LLM (used by tool calling)
# ---------------------------------------------------------------------------

class TestSearchIntegration:

    def test_search_returns_required_keys(self):
        """search() must always return 'context' and 'sources' — even if KB is empty."""
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        result = svc.search("What is machine learning?")
        assert "context" in result, "search result must contain 'context'"
        assert "sources" in result, "search result must contain 'sources'"

    def test_search_context_is_string(self):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        result = svc.search("deep learning")
        assert isinstance(result["context"], str)

    def test_search_sources_is_list(self):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        result = svc.search("neural networks")
        assert isinstance(result["sources"], list)

    def test_search_sources_are_strings_when_present(self):
        """If any sources are returned, they must be strings (file names)."""
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        result = svc.search("supervised learning")
        for source in result["sources"]:
            assert isinstance(source, str), f"source must be a string, got {type(source)}"


# ---------------------------------------------------------------------------
# retrieve_and_answer() — full RAG pipeline with LLM
# ---------------------------------------------------------------------------

class TestRetrieveAndAnswer:

    def test_returns_answer_and_sources(self):
        """Full pipeline: embed → retrieve → LLM → answer. Verify response structure."""
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        result = svc.retrieve_and_answer("What is machine learning?")
        assert "answer" in result, "result must contain 'answer'"
        assert "sources" in result, "result must contain 'sources'"

    def test_answer_is_non_empty_string(self):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        result = svc.retrieve_and_answer("Explain supervised learning briefly.")
        assert isinstance(result["answer"], str)
        assert len(result["answer"].strip()) > 0, "answer must not be blank"

    def test_sources_is_list(self):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        result = svc.retrieve_and_answer("What is deep learning?")
        assert isinstance(result["sources"], list)

    def test_handles_empty_kb_gracefully(self):
        """When no documents are ingested, should return a clear 'no data' answer."""
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        result = svc.retrieve_and_answer("What is the meaning of life?")
        # Either a real answer (KB exists) or the "no documents" fallback — both are valid
        assert isinstance(result["answer"], str)
        assert len(result["answer"].strip()) > 0


# ---------------------------------------------------------------------------
# HTTP endpoint tests — real LLM calls via API
# ---------------------------------------------------------------------------

class TestQueryEndpointIntegration:

    def test_query_endpoint_returns_200_with_answer(self):
        """POST /api/query — real LLM call. Returns 200 with answer + sources."""
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)

    def test_query_endpoint_handles_empty_kb(self):
        """POST /api/query — gracefully handles empty knowledge base."""
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "any?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0


class TestSessionChatIntegration:

    def test_send_message_makes_real_llm_call(self):
        """POST /api/sessions/{id}/messages — real LLM call. Returns 200 with answer."""
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        # Create a session
        create_resp = client.post("/api/sessions")
        session_id = create_resp.json()["id"]

        # Send a message (triggers real LLM call)
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)

    def test_get_messages_returns_conversation_history(self):
        """GET /api/sessions/{id}/messages — retrieve stored conversation."""
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        # Create a session
        create_resp = client.post("/api/sessions")
        session_id = create_resp.json()["id"]

        # Send a message
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "hi"},
        )

        # Get messages
        response = client.get(f"/api/sessions/{session_id}/messages")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2  # user message + assistant response
        assert data[0]["role"] == "user"
        assert data[1]["role"] == "assistant"

    def test_multi_turn_conversation(self):
        """Multi-turn chat — verify history is preserved across turns."""
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        # Create a session
        create_resp = client.post("/api/sessions")
        session_id = create_resp.json()["id"]

        # Turn 1
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "first"},
        )

        # Turn 2
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"message": "second"},
        )

        # Get history
        response = client.get(f"/api/sessions/{session_id}/messages")
        data = response.json()

        # Should have: user1, assistant1, user2, assistant2
        assert len(data) >= 4
        assert data[0]["content"] == "first"
        assert data[2]["content"] == "second"
