from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    IngestRequest, IngestResponse,
    IngestDirectoryRequest, IngestDirectoryResponse,
)
from app.services.ingestion import IngestionService

router = APIRouter()
ingestion_service = IngestionService()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(request: IngestRequest):
    """
    Ingest a single file into the knowledge base.

    - Reads the file from disk at the given file_path
    - Splits into overlapping chunks (~500 chars, 50 overlap)
    - Embeds all chunks in one OpenAI API call
    - Computes file-level embedding (mean of chunks)
    - Saves embeddings to disk storage

    Returns number of chunks created.
    """
    try:
        chunks_processed = ingestion_service.ingest(request.file_path)
        return IngestResponse(chunks_processed=chunks_processed)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/ingest-all", response_model=IngestDirectoryResponse)
async def ingest_directory(request: IngestDirectoryRequest):
    """
    Ingest all files in a directory into the knowledge base.

    - Iterates all files in dir_path (non-recursive)
    - Each file is processed with batch embedding (1 API call per file)
    - Failed files are reported in 'errors' but do not stop processing

    Returns summary: files processed, total chunks, per-file results, errors.
    """
    try:
        result = ingestion_service.ingest_directory(request.dir_path)
        return IngestDirectoryResponse(**result)
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
