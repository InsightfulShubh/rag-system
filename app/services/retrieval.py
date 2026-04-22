from app.storage.vector_store import VectorStore
from app.utils.embedding import get_embedding
from app.utils.similarity import cosine_similarity
from app.services.llm import LLMService
import numpy as np


class RetrievalService:
    """Two-stage RAG retrieval: file-level cosine similarity (Stage 1), then chunk-level (Stage 2)."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_service = LLMService()

    def retrieve_and_answer(self, query: str) -> dict:
        """Run the full RAG pipeline: embed → retrieve → LLM answer. Returns {answer, sources}."""
        query_embedding = get_embedding(query)
        top_files = self._get_top_files(query_embedding, top_k=2)

        if not top_files:
            return {"answer": "No documents have been ingested yet.", "sources": []}

        top_chunks = self._get_top_chunks(query_embedding, top_files, top_k=5)
        answer = self.llm_service.generate_answer(query, top_chunks)
        sources = list(dict.fromkeys(chunk["file_name"] for chunk in top_chunks))

        return {"answer": answer, "sources": sources}

    def _get_top_files(self, query_embedding: list[float], top_k: int = 2) -> list[str]:
        """Rank all files by cosine similarity to the query. Returns top_k file names."""
        file_embeddings = self.vector_store.load_all_file_embeddings()

        if not file_embeddings:
            return []

        file_names = list(file_embeddings.keys())

        # Vectorized cosine similarity over all files at once
        matrix = np.array(list(file_embeddings.values()))
        query_vec = np.array(query_embedding)

        dot_products = matrix @ query_vec
        norms_matrix = np.linalg.norm(matrix, axis=1)
        norm_query = np.linalg.norm(query_vec)
        scores = dot_products / (norms_matrix * norm_query + 1e-10)

        # O(n) partial selection, then sort only top_k results
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [file_names[i] for i in top_indices]

    def _get_top_chunks(
        self,
        query_embedding: list[float],
        file_names: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """Rank chunks from the selected files by similarity. Returns top_k across all files."""
        all_scored_chunks = []

        for file_name in file_names:
            chunks = self.vector_store.load_chunks(file_name)

            for chunk in chunks:
                score = cosine_similarity(query_embedding, chunk["embedding"])
                all_scored_chunks.append({
                    "text": chunk["text"],
                    "file_name": file_name,
                    "score": score,
                })

        # Sort all chunks from both files by descending score
        all_scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return all_scored_chunks[:top_k]

    def search(self, query: str) -> dict:
        """Retrieval-only pipeline (no LLM). Returns {context, sources}. Used by the search_kb tool."""
        query_embedding = get_embedding(query)
        top_files = self._get_top_files(query_embedding, top_k=2)

        if not top_files:
            return {"context": "", "sources": []}

        top_chunks = self._get_top_chunks(query_embedding, top_files, top_k=5)

        context = "\n\n".join(
            f"[Source: {c['file_name']}]\n{c['text']}" for c in top_chunks
        )
        sources = list(dict.fromkeys(c["file_name"] for c in top_chunks))
        return {"context": context, "sources": sources}
