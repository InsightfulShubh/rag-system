from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    IngestRequest, IngestResponse,
    IngestDirectoryRequest, IngestDirectoryResponse,
)
from app.services.ingestion import IngestionService

router = APIRouter(tags=["Knowledge Base — Setup & Ingestion"])
ingestion_service = IngestionService()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(request: IngestRequest):
    """Ingest a single file into the knowledge base. Returns chunk count."""
    try:
        chunks_processed = ingestion_service.ingest(request.file_path)
        return IngestResponse(chunks_processed=chunks_processed)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/ingest-all", response_model=IngestDirectoryResponse)
async def ingest_directory(request: IngestDirectoryRequest):
    """Ingest all files in a directory. Failed files are reported but do not stop processing."""
    try:
        result = ingestion_service.ingest_directory(request.dir_path)
        return IngestDirectoryResponse(**result)
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
