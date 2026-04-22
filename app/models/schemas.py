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
    dir_path: str = _DEFAULT_RAW_DIR


class IngestDirectoryResponse(BaseModel):
    files_processed: int
    total_chunks: int
    results: list[dict]
    errors: list[dict]


# --- Query ---

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# --- Chat ---

class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    answer: str
    sources: list[str]


class MessageRecord(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    created_at: str
