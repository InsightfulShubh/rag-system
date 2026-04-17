from app.storage.vector_store import VectorStore
from app.utils.embedding import get_embedding
from app.utils.similarity import cosine_similarity
from app.services.llm import LLMService
import numpy as np


class RetrievalService:
    """
    Implements two-stage retrieval (the core RAG logic).

    Stage 1 - File-Level Retrieval:
        - Load ALL file embeddings into memory (~12MB, fast)
        - Compute cosine similarity between query and every file embedding
        - Select top 2 files  ← enforces the "max 2 files in context" constraint

    Stage 2 - Chunk-Level Retrieval:
        - Load chunks ONLY from the 2 selected files (lazy load)
        - Rank all chunks by cosine similarity to query
        - Select top K chunks to build LLM context window

    Why two stages instead of searching all chunks directly:
        - Searching all chunks across 2000 files = potentially 40,000+ comparisons
        - Stage 1 narrows to 2 files first (~2000 comparisons, very fast)
        - Stage 2 then searches only ~40 chunks (2 files × ~20 chunks each)
        - Total: ~2040 comparisons instead of 40,000+
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_service = LLMService()

    def retrieve_and_answer(self, query: str) -> dict:
        """
        Full query pipeline: embed → retrieve files → retrieve chunks → LLM answer.

        Args:
            query: natural language question from the user

        Returns:
            dict with keys:
                - answer (str): LLM-generated answer
                - sources (list[str]): file names used as context
        """
        # Step 1: Embed the query (single API call)
        query_embedding = get_embedding(query)

        # Step 2 (Stage 1): Find the top 2 most relevant files
        top_files = self._get_top_files(query_embedding, top_k=2)

        if not top_files:
            return {"answer": "No documents have been ingested yet.", "sources": []}

        # Step 3 (Stage 2): Find the top K chunks from those 2 files
        top_chunks = self._get_top_chunks(query_embedding, top_files, top_k=5)

        # Step 4: Call LLM with context chunks + query, get answer
        answer = self.llm_service.generate_answer(query, top_chunks)

        # Step 5: Collect unique source file names for the response
        sources = list(dict.fromkeys(chunk["file_name"] for chunk in top_chunks))

        return {"answer": answer, "sources": sources}

    def _get_top_files(self, query_embedding: list[float], top_k: int = 2) -> list[str]:
        """
        Stage 1: rank ALL files by cosine similarity, return top_k file names.

        Performance approach — two optimisations over a naive Python loop:

        1. Numpy vectorized cosine similarity (vs calling cosine_similarity() in a loop):
           - Build a matrix of shape (n_files × 1536)
           - Compute ALL dot products in one matrix–vector multiply: matrix @ query_vec
           - Divide by norms in one vectorized operation
           - Result: C-speed computation instead of 2000 Python function calls

        2. np.argpartition instead of full sort (O(n) vs O(n log n)):
           - We only need top 2 out of 2000 files
           - argpartition rearranges so top_k largest are at the end — O(n) average
           - Then sort only those top_k elements — O(k log k), negligible
           - Full sort would be O(n log n) = O(2000 × 11) unnecessarily

        Args:
            query_embedding: embedding vector of the query
            top_k: number of files to select (default: 2)

        Returns:
            list of file names sorted by descending similarity score
        """
        file_embeddings = self.vector_store.load_all_file_embeddings()

        if not file_embeddings:
            return []

        file_names = list(file_embeddings.keys())

        # Stack all file embeddings into a 2D matrix: shape (n_files, dims)
        matrix = np.array(list(file_embeddings.values()))   # (2000, 1536)
        query_vec = np.array(query_embedding)                # (1536,)

        # Vectorized cosine similarity for ALL files in one call:
        #   dot_products = matrix @ query_vec  →  (2000,)
        #   norms_matrix = ||each row of matrix||  →  (2000,)
        #   norm_query   = ||query_vec||  →  scalar
        #   scores = dot_products / (norms_matrix * norm_query)  →  (2000,)
        dot_products = matrix @ query_vec
        norms_matrix = np.linalg.norm(matrix, axis=1)
        norm_query = np.linalg.norm(query_vec)
        scores = dot_products / (norms_matrix * norm_query + 1e-10)  # 1e-10 avoids /0

        # O(n) partial selection — only guarantees top_k are at the end, unsorted
        top_indices = np.argpartition(scores, -top_k)[-top_k:]

        # Sort only the top_k candidates (O(k log k), k=2 → negligible)
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [file_names[i] for i in top_indices]

    def _get_top_chunks(
        self,
        query_embedding: list[float],
        file_names: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Stage 2: rank chunks from selected files by cosine similarity, return top_k.

        Why we merge chunks from both files before ranking:
            - If file A has scores [0.9, 0.85, 0.7] and file B has [0.88, 0.6, 0.5]
            - Merging → [0.9, 0.88, 0.85, 0.7, 0.6, 0.5]
            - Top 5 = best chunks regardless of which file they came from
            - This is better than "top K/2 from each file" which forces equal
              representation even when one file is much more relevant

        Args:
            query_embedding: embedding vector of the query
            file_names: list of file names selected in Stage 1
            top_k: number of chunks to return across all selected files

        Returns:
            list of top_k chunk dicts, each with keys:
                - text (str): the chunk text to inject into LLM context
                - file_name (str): source file name (used as citation in response)
                - score (float): cosine similarity score (for debugging/transparency)
        """
        all_scored_chunks = []

        for file_name in file_names:
            # Lazy load — reads only this file's chunk JSON, not all chunks
            chunks = self.vector_store.load_chunks(file_name)

            for chunk in chunks:
                score = cosine_similarity(query_embedding, chunk["embedding"])
                all_scored_chunks.append({
                    "text": chunk["text"],
                    "file_name": file_name,
                    "score": score,
                })

        # Sort all chunks from both files together by descending score
        all_scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return all_scored_chunks[:top_k]
