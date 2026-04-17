# RAG System — From Scratch (No Frameworks)

A lightweight **Retrieval-Augmented Generation (RAG)** system built with FastAPI and the OpenAI API — no LangChain, no vector database, no ORM. Pure Python.

---

## Overview

Given a knowledge base of text files, the system answers user questions by finding the most relevant content and generating a grounded answer using an LLM. The design enforces strict context limits: **at most 2 files** may be passed to the LLM at once, making retrieval accuracy critical.

---

## Features

- **Two-stage retrieval** — file-level cosine similarity narrows the search, then chunk-level retrieval picks the best passages
- **No RAG frameworks** — built directly on the OpenAI API
- **Dual provider support** — works with OpenAI (public) or Azure OpenAI (enterprise), switchable via a single `.env` flag
- **Batch embeddings** — all chunks of a file embedded in a single API call during ingestion
- **Numpy-optimised similarity** — matrix multiply + `argpartition` for O(n) top-k selection
- **Per-file chunk storage** — only the 2 selected files' chunks are loaded per query (memory efficient)
- **Source attribution** — every answer includes the source file names
- **Hallucination guard** — system prompt instructs the LLM to say "I don't know" if the answer isn't in the context

---

## Architecture

```
Query
  │
  ▼
[Embed query]
  │
  ▼
Stage 1 — File-level retrieval
  Compare query embedding against ALL file embeddings (numpy matrix op)
  → Select top 2 most relevant files
  │
  ▼
Stage 2 — Chunk-level retrieval
  Load chunks only from selected 2 files
  Compare query against all chunks → Select top 5 chunks
  │
  ▼
[Build prompt: system grounding + context chunks + user question]
  │
  ▼
[LLM → Answer + Sources]
```

---

## Project Structure

```
rag-system/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Settings loaded from .env
│   ├── clients/
│   │   └── llm_client.py        # Client factory (OpenAI / Azure OpenAI)
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── utils/
│   │   ├── similarity.py        # Cosine similarity
│   │   ├── chunking.py          # Sliding window text chunker
│   │   └── embedding.py        # OpenAI embedding wrappers
│   ├── storage/
│   │   ├── document_reader.py   # Read raw text files from disk
│   │   └── vector_store.py      # Load/save embeddings as JSON
│   ├── services/
│   │   ├── ingestion.py         # Ingestion pipeline orchestrator
│   │   ├── retrieval.py         # Two-stage retrieval logic
│   │   └── llm.py              # LLM prompt building + chat completion
│   └── routes/
│       ├── ingest.py            # /api/ingest, /api/ingest-all
│       └── query.py             # /api/query
├── data/
│   ├── raw/                     # Knowledge base text files (input)
│   └── embeddings/
│       ├── files.json           # All file-level embeddings
│       └── chunks/<name>.json   # Per-file chunk embeddings (lazy-loaded)
├── .env                         # Local config (not committed)
├── .env-sample                  # Config template
├── requirements.txt
└── start-server.ps1             # Helper script to restart server
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd rag-system
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env-sample` to `.env` and fill in credentials:

```bash
cp .env-sample .env
```

```env
# Choose provider: false = OpenAI, true = Azure OpenAI (DIAL)
USE_AZURE_OPENAI=false

# OpenAI (default)
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini

# Azure OpenAI (set USE_AZURE_OPENAI=true to activate)
AZURE_API_KEY=your-azure-dial-key
AZURE_ENDPOINT=https://*****.***.****.***
AZURE_API_VERSION=2024-02-01
AZURE_DEPLOYMENT_NAME=gpt-4
AZURE_EMBEDDING_MODEL=text-embedding-3-small-1
```

> **Note:** For local development and testing the public OpenAI API was used. For internal evaluation, the organisation-provided **Azure OpenAI** endpoint was used — switchable with no code changes.

### 3. Add knowledge base files

Place `.txt` files in `data/raw/`.

### 4. Start the server

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/ingest` | Ingest a single file |
| `POST` | `/api/ingest-all` | Ingest all files in a directory |
| `POST` | `/api/query` | Ask a question against the knowledge base |

### POST `/api/ingest`
```json
{ "file_path": "data/raw/FastAPI_Framework.txt" }
```

### POST `/api/ingest-all`
```json
{ "dir_path": "data/raw" }
```

### POST `/api/query`
```json
{ "query": "What is the difference between supervised and unsupervised learning?" }
```
**Response:**
```json
{
  "answer": "Supervised learning involves training the algorithm on labeled data, where the correct output is provided for each input. It is commonly used for tasks like classification and regression, such as image recognition and predicting continuous values.\n\nUnsupervised learning, on the other hand, involves training the algorithm on unlabeled data, where the algorithm must identify patterns and relationships in the data on its own. It is commonly used for tasks like clustering and dimensionality reduction, such as customer segmentation and data visualization.",
  "sources": ["Machine_Learning.txt", "Deep_Learning.txt"]
}
```

> **Why POST for `/query`?** Query strings sent as GET parameters have URL length limits and get cached by browsers/proxies. POST allows clean JSON bodies of any size and prevents caching of expensive LLM calls.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API framework | FastAPI + Uvicorn |
| LLM & Embeddings | OpenAI API (`gpt-4o-mini`, `text-embedding-3-small`) |
| Enterprise LLM | Azure OpenAI (`gpt-4`) |
| Vector similarity | NumPy (cosine similarity, matrix multiply) |
| Storage | JSON files (no database) |
| Config | python-dotenv |
| Validation | Pydantic v2 |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Two-stage retrieval | Satisfies the "max 2 files in context" constraint efficiently |
| Per-file chunk JSON | Only load chunks for selected files, not the entire KB |
| Batch embeddings | One API call per file during ingestion instead of N calls per chunk |
| `np.argpartition` | O(n) top-k selection vs O(n log n) sorting |
| No vector database | Keeps the system self-contained with zero infrastructure dependencies |
| Provider flag in `.env` | Switch between OpenAI and Azure DIAL without any code changes |

---

## Assumptions & Simplifications

The following trade-offs were made intentionally for simplicity and time constraints. Each point notes the production-grade alternative.

| Area | What was done | Production alternative |
|------|--------------|------------------------|
| **Embedding storage** | Embeddings saved as plain JSON files | Binary formats (pickle, numpy `.npy`, FAISS index) are faster to load and more compact, but JSON is human-readable and zero-dependency |
| **File-level embedding** | Mean of all chunk embeddings used to represent a file | LLM-generated summary of the file could be embedded instead — better semantic representation, especially for long or varied documents. Mean embedding is fast and free (no extra API call) |
| **Chunk strategy** | Fixed-size sliding window (character-based) | Semantic/sentence-aware chunking (e.g. split on paragraphs, headers) would produce more coherent chunks |
| **Top-k fixed values** | `top_k=2` files and `top_k=5` chunks are hardcoded defaults | Should be configurable per-query or tuned via evaluation |
| **No re-ingestion check** | Re-ingesting a file overwrites existing embeddings silently | A production system should track file hashes to skip unchanged files and detect deletions |
| **No authentication** | API endpoints are open with no auth | Should add API key header or OAuth2 for any shared/deployed instance |
| **Sequential ingestion** | Files ingested one by one in `ingest_directory()` | Parallel ingestion with `asyncio` or `ThreadPoolExecutor` would be faster for large KBs |
| **In-memory file embeddings** | All file embeddings loaded into memory at query time | Fine for small KBs; for large KBs an ANN index (FAISS, Hnswlib) is needed |
| **Single-turn queries** | No conversation history | A production system would maintain a session context window for follow-up questions |

---

## Future Improvements

- Pluggable vector store (FAISS, ChromaDB)
- LLM-based file summarization for better file-level embeddings
- Hybrid search (keyword + semantic / BM25 + dense)
- Streaming responses
- Re-ranking with a cross-encoder
- File hash tracking to skip re-ingestion of unchanged files
- Authentication on API endpoints
- Parallel ingestion for large knowledge bases