from pydantic import BaseModel


# --- Ingest ---

class IngestRequest(BaseModel):
    file_path: str


class IngestResponse(BaseModel):
    chunks_processed: int


# --- Ingest Directory ---

class IngestDirectoryRequest(BaseModel):
    dir_path: str   # path to directory containing text files (e.g. "data/raw")


class IngestDirectoryResponse(BaseModel):
    files_processed: int
    total_chunks: int
    results: list[dict]  # [{"file": "report.txt", "chunks": 12}, ...]
    errors: list[dict]   # [{"file": "bad.txt", "error": "..."}, ...] — files that failed


# --- Query ---

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
