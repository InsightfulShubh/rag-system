from app.storage.vector_store import VectorStore
from app.utils.embedding import get_embedding
from app.utils.similarity import cosine_similarity
from app.services.llm import LLMService


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

        How it works:
            - files.json is loaded entirely into memory (it's small, ~12MB)
            - cosine_similarity() compares the query vector to each file's vector
            - sorted() with reverse=True gives highest-similarity files first

        Args:
            query_embedding: embedding vector of the query
            top_k: number of files to select (default: 2, per architecture constraint)

        Returns:
            list of file names sorted by descending similarity score
            e.g. ["report.txt", "policy.txt"]
        """
        # Load all file-level embeddings: {file_name: embedding_vector}
        file_embeddings = self.vector_store.load_all_file_embeddings()

        if not file_embeddings:
            return []

        # Score each file against the query
        scores = [
            (file_name, cosine_similarity(query_embedding, embedding))
            for file_name, embedding in file_embeddings.items()
        ]

        # Sort descending by score, take top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [file_name for file_name, _ in scores[:top_k]]

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
