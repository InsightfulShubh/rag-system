from pydantic import BaseModel


# --- Ingest ---

class IngestRequest(BaseModel):
    file_path: str


class IngestResponse(BaseModel):
    chunks_processed: int


# --- Ingest Directory ---
from pathlib import Path as _Path
_DEFAULT_RAW_DIR = str(_Path(__file__).resolve().parent.parent / "data" / "raw")

class IngestDirectoryRequest(BaseModel):
    dir_path: str = _DEFAULT_RAW_DIR   # defaults to <project>/data/raw


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


# --- Chat ---

class MessageRequest(BaseModel):
    message: str      # the user's message text


class MessageResponse(BaseModel):
    answer: str           # LLM-generated answer
    sources: list[str]    # source files used to answer (from search_kb)


class MessageRecord(BaseModel):
    id: str
    session_id: str
    role: str             # 'user' or 'assistant'
    content: str
    created_at: str
