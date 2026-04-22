import os

import numpy as np

from app.storage.document_reader import DocumentReader
from app.storage.vector_store import VectorStore
from app.utils.chunking import split_into_chunks
from app.utils.embedding import get_embeddings_batch


class IngestionService:
    """Orchestrates the full ingestion pipeline: read → chunk → embed → store."""

    def __init__(self):
        self.doc_reader = DocumentReader()
        self.vector_store = VectorStore()

    def ingest(self, file_path: str) -> int:
        """Ingest a file into the knowledge base. Returns number of chunks created."""
        file_name = self.doc_reader.get_file_name(file_path)
        text = self.doc_reader.read_file(file_path)

        chunks_text = split_into_chunks(text)

        if not chunks_text:
            return 0

        embeddings = get_embeddings_batch(chunks_text)

        file_embedding = self._mean_embedding(embeddings)

        chunks = [
            {"text": text, "embedding": emb}
            for text, emb in zip(chunks_text, embeddings)
        ]
        self.vector_store.save_chunks(file_name, chunks)
        self.vector_store.save_file_embedding(file_name, file_embedding)

        return len(chunks)

    def ingest_directory(self, dir_path: str) -> dict:
        """
        Ingest all text files found in a directory.

        Processes files sequentially. Each file uses batch embedding
        (all its chunks embedded in one OpenAI call), so total API calls
        = number of files (not number of chunks).

        Files that fail (e.g. unreadable, empty) are logged in 'errors'
        and processing continues with the remaining files.

        Args:
            dir_path: path to the directory containing text files

        Returns:
            dict with keys:
                - files_processed (int): number of successfully ingested files
                - total_chunks (int): total chunks across all files
                - results (list[dict]): [{"file": name, "chunks": n}, ...]
                - errors (list[dict]): [{"file": name, "error": msg}, ...]

        Raises:
            NotADirectoryError: if dir_path does not point to a valid directory
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        # Collect all files in the directory (non-recursive, text files only)
        all_files = [
            os.path.join(dir_path, f)
            for f in sorted(os.listdir(dir_path))
            if os.path.isfile(os.path.join(dir_path, f))
        ]

        results = []
        errors = []
        total_chunks = 0

        for file_path in all_files:
            file_name = self.doc_reader.get_file_name(file_path)
            try:
                chunks_count = self.ingest(file_path)
                results.append({"file": file_name, "chunks": chunks_count})
                total_chunks += chunks_count
            except Exception as e:
                errors.append({"file": file_name, "error": str(e)})

        return {
            "files_processed": len(results),
            "total_chunks": total_chunks,
            "results": results,
            "errors": errors,
        }

    def _mean_embedding(self, embeddings: list[list[float]]) -> list[float]:
        """Return the element-wise mean of a list of embedding vectors."""
        return np.array(embeddings).mean(axis=0).tolist()
