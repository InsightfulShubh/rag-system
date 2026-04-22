import os


class DocumentReader:
    """Reads raw text documents from disk."""

    def read_file(self, file_path: str) -> str:
        """Read and return the full text content of a file (UTF-8, replace errors).

        Raises:
            FileNotFoundError: if the file does not exist at file_path
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def get_file_name(self, file_path: str) -> str:
        """Extract the base file name from a path (used as the storage key)."""
        return os.path.basename(file_path)
