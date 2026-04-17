import numpy as np

from app.storage.document_reader import DocumentReader
from app.storage.vector_store import VectorStore
from app.utils.chunking import split_into_chunks
from app.utils.embedding import get_embeddings_batch


class IngestionService:
    """
    Orchestrates the full ingestion pipeline for a single file.

    Pipeline steps:
        1. Read file content from disk          (DocumentReader)
        2. Split content into overlapping chunks (chunking util)
        3. Embed all chunks in one API call      (embedding util - batch)
        4. Compute file-level embedding          (math: mean of chunk embeddings)
        5. Save chunk embeddings to storage      (VectorStore)
        6. Save file-level embedding to storage  (VectorStore)

    Dependencies wired in __init__ so they can be replaced in tests.
    """

    def __init__(self):
        self.doc_reader = DocumentReader()
        self.vector_store = VectorStore()

    def ingest(self, file_path: str) -> int:
        """
        Ingest a file into the knowledge base.

        Args:
            file_path: path to the raw text file (absolute or relative)

        Returns:
            int: number of chunks created and stored
        """
        # Step 1: Read file content from disk
        # file_name ("report.txt") becomes the key in all storage
        file_name = self.doc_reader.get_file_name(file_path)
        text = self.doc_reader.read_file(file_path)

        # Step 2: Split text into overlapping chunks
        # Each chunk is ~500 chars with 50-char overlap between consecutive chunks
        chunks_text = split_into_chunks(text)

        if not chunks_text:
            return 0

        # Step 3: Embed all chunks in a single OpenAI API call (batch)
        # Returns list of vectors: embeddings[i] corresponds to chunks_text[i]
        embeddings = get_embeddings_batch(chunks_text)

        # Step 4: Compute file-level embedding = mean of all chunk embeddings
        # This single vector represents the "overall topic" of the file
        # Used in Stage 1 retrieval to pick the top 2 most relevant files
        file_embedding = self._mean_embedding(embeddings)

        # Step 5: Build chunk dicts and persist to storage
        # Each chunk stores its text (for LLM context) and embedding (for similarity)
        chunks = [
            {"text": text, "embedding": emb}
            for text, emb in zip(chunks_text, embeddings)
        ]
        self.vector_store.save_chunks(file_name, chunks)

        # Step 6: Persist the file-level embedding
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
        import os

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        # Collect all files in the directory (non-recursive, text files only)
        all_files = [
            os.path.join(dir_path, f)
            for f in sorted(os.listdir(dir_path))   # sorted for deterministic order
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
                # Log failure but continue processing remaining files
                errors.append({"file": file_name, "error": str(e)})

        return {
            "files_processed": len(results),
            "total_chunks": total_chunks,
            "results": results,
            "errors": errors,
        }

    def _mean_embedding(self, embeddings: list[list[float]]) -> list[float]:
        """
        Compute the element-wise mean of a list of embedding vectors.

        Why mean for file-level embedding:
            - Each chunk represents a piece of the file's content
            - Averaging all chunk vectors gives a centroid that captures
              the overall topic of the file
            - Example: a file about "Python web frameworks" would have chunks
              covering FastAPI, Django, Flask — the mean vector lands somewhere
              in the middle of all those concepts, representing the file as a whole

        Args:
            embeddings: list of embedding vectors (one per chunk)

        Returns:
            list[float]: single averaged embedding vector
        """
        # np.mean across axis=0 computes element-wise mean in one vectorised call:
        #   np.array(embeddings) → shape (n_chunks, 1536)
        #   .mean(axis=0)        → shape (1536,)  — average across chunk dimension
        #   .tolist()            → back to plain Python list for JSON serialisation
        return np.array(embeddings).mean(axis=0).tolist()
