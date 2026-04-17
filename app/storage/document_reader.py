import os


class DocumentReader:
    """
    Reads raw text documents from disk.

    Responsibility:
        - Read text files from any path on disk
        - Extract file name from a path

    Note:
        We accept any file_path (absolute or relative) from the caller.
        The ingestion service decides where files live; DocumentReader just reads them.
    """

    def read_file(self, file_path: str) -> str:
        """
        Read and return the full text content of a file.

        Why utf-8 with errors='replace':
            - Most text files are UTF-8
            - errors='replace' prevents crashes on files with unexpected
              byte sequences (e.g., a stray Windows-1252 character)
            - Replaced chars become '?' which is acceptable for RAG use

        Args:
            file_path: path to the file (absolute or relative)

        Returns:
            str: full text content of the file

        Raises:
            FileNotFoundError: if the file does not exist at file_path
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def get_file_name(self, file_path: str) -> str:
        """
        Extract the base file name from a path.

        Used as the unique key for storing/retrieving embeddings.
        
        Examples:
            "/data/raw/report.txt"      → "report.txt"
            "data/raw/notes.txt"        → "notes.txt"
            "C:\\docs\\policy.txt"      → "policy.txt"

        Args:
            file_path: full or relative file path

        Returns:
            str: base file name including extension (e.g., "report.txt")
        """
        return os.path.basename(file_path)
