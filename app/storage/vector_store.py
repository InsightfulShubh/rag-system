import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_DIR = str(_PROJECT_ROOT / "data" / "embeddings" / "chunks")
FILES_PATH = str(_PROJECT_ROOT / "data" / "embeddings" / "files.json")


class VectorStore:
    """JSON-backed store for chunk and file-level embeddings. One file per document for lazy loading."""

    def _load_json(self, path: str) -> dict | list:
        """Read JSON from disk. Returns {} if the file doesn't exist or is empty."""
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)

    def _save_json(self, path: str, data: dict | list) -> None:
        """Write data to a JSON file, creating parent directories as needed."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _chunk_path(self, file_name: str) -> str:
        return os.path.join(CHUNKS_DIR, f"{file_name}.json")

    def save_chunks(self, file_name: str, chunks: list[dict]) -> None:
        """Persist chunk embeddings for a file (overwrites existing data)."""
        self._save_json(self._chunk_path(file_name), chunks)

    def load_chunks(self, file_name: str) -> list[dict]:
        """Load chunk embeddings for a single file. Returns [] if not found."""
        path = self._chunk_path(file_name)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_file_embedding(self, file_name: str, embedding: list[float]) -> None:
        """Persist the file-level embedding (mean of chunk embeddings). Upserts."""
        data = self._load_json(FILES_PATH)
        data[file_name] = {"embedding": embedding}  # upsert
        self._save_json(FILES_PATH, data)

    def load_all_file_embeddings(self) -> dict[str, list[float]]:
        """Load all file-level embeddings. Used in Stage 1 to rank files by similarity."""
        data = self._load_json(FILES_PATH)
        return {file_name: entry["embedding"] for file_name, entry in data.items()}
