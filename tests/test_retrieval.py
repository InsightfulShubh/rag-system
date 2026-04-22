"""
Unit tests for the two-stage retrieval logic (no LLM, no API calls required).

All tests use pre-built mock embeddings written to a temp directory so the
suite can run fully offline.
"""
import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures — build / teardown a minimal on-disk embedding store
# ---------------------------------------------------------------------------

FILE_EMBEDDINGS = {
    "finance.txt": {"embedding": [1.0, 0.0, 0.0]},
    "python.txt":  {"embedding": [0.0, 1.0, 0.0]},
    "history.txt": {"embedding": [0.0, 0.0, 1.0]},
}

CHUNK_EMBEDDINGS = {
    "finance.txt": [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
    "python.txt":  [[0.1, 0.85, 0.05], [0.0, 0.9, 0.1]],
    "history.txt": [[0.05, 0.0, 0.95], [0.0, 0.1, 0.9]],
}


@pytest.fixture()
def mock_embeddings(tmp_path, monkeypatch):
    """
    Write mock embedding files to a temp directory and patch the paths that
    VectorStore reads so RetrievalService uses them instead of real data.
    """
    # Write files.json
    files_json = tmp_path / "files.json"
    files_json.write_text(json.dumps(FILE_EMBEDDINGS))

    # Write per-file chunk JSONs
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    for fname, embeddings in CHUNK_EMBEDDINGS.items():
        chunk_data = [{"text": f"chunk from {fname}", "embedding": e} for e in embeddings]
        # VectorStore._chunk_path() appends .json to the full filename → finance.txt.json
        (chunks_dir / f"{fname}.json").write_text(json.dumps(chunk_data))

    # Patch VectorStore paths before RetrievalService is instantiated
    import app.storage.vector_store as vs
    monkeypatch.setattr(vs, "FILES_PATH", str(files_json))
    monkeypatch.setattr(vs, "CHUNKS_DIR", str(chunks_dir))

    return tmp_path


# ---------------------------------------------------------------------------
# Stage 1 — file-level retrieval
# ---------------------------------------------------------------------------

class TestStage1FileRetrieval:

    def test_top_file_is_finance_for_finance_query(self, mock_embeddings):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        query_emb = [0.95, 0.05, 0.0]   # points strongly toward finance
        top_files = svc._get_top_files(query_emb, top_k=2)
        assert top_files[0] == "finance.txt", "finance.txt should rank first"

    def test_second_file_is_python_for_finance_query(self, mock_embeddings):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        query_emb = [0.95, 0.05, 0.0]
        top_files = svc._get_top_files(query_emb, top_k=2)
        assert top_files[1] == "python.txt", "python.txt should rank second"

    def test_top_file_is_history_for_history_query(self, mock_embeddings):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        query_emb = [0.0, 0.05, 0.95]   # points toward history
        top_files = svc._get_top_files(query_emb, top_k=1)
        assert top_files[0] == "history.txt"

    def test_returns_at_most_top_k_files(self, mock_embeddings):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        top_files = svc._get_top_files([1.0, 0.0, 0.0], top_k=2)
        assert len(top_files) == 2

    def test_returns_empty_list_when_no_embeddings(self, tmp_path, monkeypatch):
        import app.storage.vector_store as vs
        empty = tmp_path / "empty.json"
        empty.write_text("{}")
        monkeypatch.setattr(vs, "FILES_PATH", str(empty))
        monkeypatch.setattr(vs, "CHUNKS_DIR", str(tmp_path))
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        assert svc._get_top_files([1.0, 0.0, 0.0]) == []


# ---------------------------------------------------------------------------
# Stage 2 — chunk-level retrieval
# ---------------------------------------------------------------------------

class TestStage2ChunkRetrieval:

    def test_chunks_returned_only_from_selected_files(self, mock_embeddings):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        query_emb = [0.95, 0.05, 0.0]
        top_chunks = svc._get_top_chunks(query_emb, ["finance.txt"], top_k=5)
        for chunk in top_chunks:
            assert chunk["file_name"] == "finance.txt"

    def test_chunk_count_respects_top_k(self, mock_embeddings):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        query_emb = [0.95, 0.05, 0.0]
        top_chunks = svc._get_top_chunks(query_emb, ["finance.txt", "python.txt"], top_k=3)
        assert len(top_chunks) <= 3

    def test_chunks_sorted_by_descending_score(self, mock_embeddings):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        query_emb = [0.95, 0.05, 0.0]
        top_chunks = svc._get_top_chunks(query_emb, ["finance.txt", "python.txt"], top_k=4)
        scores = [c["score"] for c in top_chunks]
        assert scores == sorted(scores, reverse=True), "chunks must be sorted best-first"

    def test_chunk_has_required_keys(self, mock_embeddings):
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        top_chunks = svc._get_top_chunks([0.9, 0.1, 0.0], ["finance.txt"], top_k=2)
        for chunk in top_chunks:
            assert "text" in chunk
            assert "file_name" in chunk
            assert "score" in chunk

    def test_best_finance_chunk_has_higher_score_than_best_python_chunk(self, mock_embeddings):
        """finance chunks are closer to [0.95, 0.05, 0.0] than python chunks."""
        from app.services.retrieval import RetrievalService
        svc = RetrievalService()
        query_emb = [0.95, 0.05, 0.0]
        chunks = svc._get_top_chunks(query_emb, ["finance.txt", "python.txt"], top_k=4)
        finance_scores = [c["score"] for c in chunks if c["file_name"] == "finance.txt"]
        python_scores  = [c["score"] for c in chunks if c["file_name"] == "python.txt"]
        assert max(finance_scores) > max(python_scores)
