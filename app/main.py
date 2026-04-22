from fastapi import FastAPI
from app.routes import ingest, query, session
from app.storage.db import init_db


app = FastAPI(
    title="RAG System — Chat Agent v2",
    description=(
        "Simple Retrieval-Augmented Generation API. "
        "Uses two-stage retrieval: file-level cosine similarity to select top 2 files, "
        "then chunk-level cosine similarity to build LLM context."
    ),
    version="2.0.0",
)


@app.on_event("startup")
def startup():
    init_db()

app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(session.router, prefix="/api")


@app.get("/health", tags=["System"])
def health_check():
    """Returns 200 OK when the server is running."""
    return {"status": "ok"}
