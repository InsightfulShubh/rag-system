import json
import os

CHUNKS_DIR = "data/embeddings/chunks"   # one JSON file per document
FILES_PATH = "data/embeddings/files.json"


class VectorStore:
    """
    Manages persistence of chunk and file embeddings as JSON files.

    Why JSON instead of a database:
        - Simple, human-readable, no external dependencies
        - Fine for 2000 files (our target scale)
        - File embeddings (~2000 entries) fit easily in memory
        - Chunk embeddings are loaded lazily per file, not all at once

    Storage layout:
        data/embeddings/
            files.json                  ← all file-level embeddings (loaded at startup)
            chunks/
                report.txt.json         ← chunks for report.txt only
                policy.txt.json         ← chunks for policy.txt only

        files.json:
            {
                "report.txt": {"embedding": [0.001, -0.022, ...]},
                "policy.txt": {"embedding": [0.091,  0.011, ...]}
            }

        chunks/report.txt.json:
            [
                {"text": "Company was founded...", "embedding": [0.012, ...]},
                {"text": "Revenue grew by 20%...", "embedding": [-0.034, ...]}
            ]

    Why one chunk file per document (not one big chunks.json):
        - chunks.json for 2000 files could be ~490MB
          (2000 files × 20 chunks × 1536 floats × 8 bytes)
        - Loading all 490MB just to get 1 file's chunks is wasteful
        - Per-file approach: load only the few KB you actually need
    """

    def _load_json(self, path: str) -> dict | list:
        """
        Read a JSON file from disk and return its contents.
        Returns an empty dict if the file doesn't exist yet.

        Args:
            path: path to the JSON file

        Returns:
            dict or list: parsed JSON content, or {} if file not found
        """
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json(self, path: str, data: dict | list) -> None:
        """
        Write a dict or list to a JSON file on disk.
        Creates parent directories if they don't exist.

        Args:
            path: path to write the JSON file
            data: dict or list to serialize and write
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _chunk_path(self, file_name: str) -> str:
        """
        Build the path to a file's dedicated chunk JSON file.

        Args:
            file_name: e.g. "report.txt"

        Returns:
            str: e.g. "data/embeddings/chunks/report.txt.json"
        """
        return os.path.join(CHUNKS_DIR, f"{file_name}.json")

    def save_chunks(self, file_name: str, chunks: list[dict]) -> None:
        """
        Persist chunk embeddings for a given file (upsert behaviour).
        Writes to its own dedicated file: chunks/<file_name>.json

        Upsert means:
            - If file already exists → overwrite it (re-ingesting replaces old data)
            - If file is new → create it

        Args:
            file_name: name of the source file (e.g. "report.txt")
            chunks: list of dicts, each with 'text' (str) and 'embedding' (list[float])
        """
        self._save_json(self._chunk_path(file_name), chunks)

    def load_chunks(self, file_name: str) -> list[dict]:
        """
        Load chunk embeddings for a specific file only.

        Reads only chunks/<file_name>.json — no other file is touched.
        This is the key efficiency win: 1 file's chunks (~few KB)
        instead of the entire chunks store (~490MB).

        Args:
            file_name: name of the file whose chunks to load (e.g. "report.txt")

        Returns:
            list of chunk dicts with keys 'text' and 'embedding'
            Empty list if file_name not found.
        """
        path = self._chunk_path(file_name)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_file_embedding(self, file_name: str, embedding: list[float]) -> None:
        """
        Persist the file-level embedding (mean of all chunk embeddings).

        This is the embedding used in Stage 1 retrieval to select
        the top 2 most relevant files before looking at chunks.

        Args:
            file_name: name of the source file (e.g. "report.txt")
            embedding: averaged embedding vector (list of 1536 floats)
        """
        data = self._load_json(FILES_PATH)
        data[file_name] = {"embedding": embedding}  # upsert
        self._save_json(FILES_PATH, data)

    def load_all_file_embeddings(self) -> dict[str, list[float]]:
        """
        Load ALL file-level embeddings into memory at once.

        Why load all at once:
            - Stage 1 requires comparing query against EVERY file
            - files.json is small: 2000 files × 1536 floats × 4 bytes ≈ 12MB
            - Fits comfortably in RAM, fast to scan with cosine similarity

        Returns:
            dict mapping file_name -> embedding vector
            e.g. {"report.txt": [0.012, ...], "policy.txt": [-0.034, ...]}
        """
        data = self._load_json(FILES_PATH)
        return {file_name: entry["embedding"] for file_name, entry in data.items()}
