# RAG System — From Scratch (No Frameworks)

A lightweight **Retrieval-Augmented Generation (RAG)** system built with FastAPI and the OpenAI API — no LangChain, no vector database, no ORM. Pure Python.

Supports both **single-turn queries** and **multi-turn chat sessions** with conversation history, OpenAI function/tool calling, and SQLite-persisted sessions.

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
- **Chat sessions** — multi-turn conversations with full history persisted in SQLite
- **Conversation history** — last N messages (configurable, default 6) passed as context on every turn
- **OpenAI tool/function calling** — LLM decides when to call `search_kb`; backend executes the retrieval and feeds results back
- **Force tool usage** — configurable flag to always invoke `search_kb` on every turn (default: `true`)

---

## Architecture

### Single-turn query (`/api/query`)

```
User question
  │
  ▼
[Embed query]  ──► OpenAI Embeddings API
  │
  ▼
Stage 1 — File-level retrieval
  Vectorized cosine similarity (NumPy) against all file embeddings
  → Select top 2 most relevant files
  │
  ▼
Stage 2 — Chunk-level retrieval
  Load chunks only from the 2 selected files
  Cosine similarity across ~40 chunks → Select top 5
  │
  ▼
[Build prompt: system grounding + context chunks + user question]
  │
  ▼
[LLM → Answer + Sources]
```

### Multi-turn chat session (`/api/sessions/{id}/messages`)

```
User message
  │
  ▼
Save to SQLite  ──────────────────────────────────────────────────┐
  │                                                               │
  ▼                                                               │
Load last 6 messages from SQLite (conversation history)           │
  │                                                               │
  ▼                                                               │
LLM call #1  ──── with search_kb tool definition ────►  LLM      │
              tool_choice = "required" (FORCE_TOOL_USAGE=true)    │
                                │                                 │
                    returns tool_call {query: "..."}              │
                                │                                 │
  ◄─────────────── execute_tool_call() ──────────────────────────┘
  │
  ▼
search_kb(query)
  → get_embedding(query)
  → Stage 1 (file-level cosine similarity)
  → Stage 2 (chunk-level cosine similarity)
  → {context, sources}
  │
  ▼
Feed tool result back as "tool" role message
  │
  ▼
LLM call #2  (tool_choice="none") ──► Final answer text
  │
  ▼
Save assistant message to SQLite
  │
  ▼
Return {answer, sources}
```

**Memory optimization:**
- File embeddings are loaded into memory at query time for fast Stage 1 retrieval
- Chunk embeddings are lazily loaded per selected file — only 2 files' chunks loaded per query

---

## Project Structure

```
rag-system/
├── app/
│   ├── main.py                    # FastAPI app entry point, DB init on startup
│   ├── config.py                  # All settings loaded from .env
│   ├── clients/
│   │   └── llm_client.py          # Client factory (OpenAI / Azure OpenAI)
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response schemas
│   ├── utils/
│   │   ├── similarity.py          # Cosine similarity helper
│   │   ├── chunking.py            # Sliding window text chunker
│   │   └── embedding.py           # OpenAI embedding wrappers (single + batch)
│   ├── storage/
│   │   ├── document_reader.py     # Read raw .txt files from disk
│   │   ├── vector_store.py        # Load/save embeddings as JSON (absolute paths)
│   │   └── db.py                  # SQLite: sessions + messages (chat history)
│   ├── services/
│   │   ├── ingestion.py           # Ingestion pipeline orchestrator
│   │   ├── retrieval.py           # Two-stage retrieval + search() for tool use
│   │   ├── llm.py                 # LLM prompt building + chat completion
│   │   ├── tools.py               # OpenAI tool spec + execute_tool_call()
│   │   ├── session_service.py     # Session CRUD (create/list/delete)
│   │   └── chat_service.py        # Full tool-calling chat loop
│   └── routes/
│       ├── ingest.py              # POST /api/ingest, /api/ingest-all
│       ├── query.py               # POST /api/query  (single-turn)
│       └── session.py             # /api/sessions + /{id}/messages (chat)
├── data/
│   ├── raw/                       # Knowledge base .txt files (input)
│   ├── chat.db                    # SQLite database (sessions + messages)
│   └── embeddings/
│       ├── files.json             # All file-level embeddings
│       └── chunks/<name>.json     # Per-file chunk embeddings (lazy-loaded)
├── Dockerfile                     # Multi-stage production build
├── .dockerignore
├── .env                           # Local secrets (not committed)
├── .env-sample                    # Config template
├── requirements.txt               # Direct dependencies only
└── rag-system.postman_collection.json
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
# ── Provider ─────────────────────────────────────────────────
# false = OpenAI (default), true = Azure OpenAI (DIAL)
USE_AZURE_OPENAI=false

# ── OpenAI (used when USE_AZURE_OPENAI=false) ─────────────────
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini

# ── Azure OpenAI / DIAL (used when USE_AZURE_OPENAI=true) ─────
AZURE_API_KEY=your-azure-dial-key
AZURE_ENDPOINT=https://ai-****.**.***.com
AZURE_API_VERSION=2024-02-01
AZURE_DEPLOYMENT_NAME=gpt-4
AZURE_EMBEDDING_MODEL=text-embedding-3-small-1

# ── LLM settings ─────────────────────────────────────────────
LLM_TEMPERATURE=0          # 0 = deterministic (recommended for RAG)

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# ── Chat agent ────────────────────────────────────────────────
HISTORY_LIMIT=6            # messages passed as context to LLM per turn
FORCE_TOOL_USAGE=true      # true = always call search_kb; false = LLM decides
# SYSTEM_PROMPT=...        # optional override of the default system prompt
```

> **Note:** For local development, the public OpenAI API was used. For internal evaluation, the organisation-provided **Azure OpenAI (DIAL)** endpoint was used — switchable with no code changes.

### 3. Add knowledge base files

Place `.txt` files in `data/raw/`.

### 4. Start the server

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### 5. Ingest documents (first run only)

```bash
# Ingest all files in data/raw/ in one call
curl -X POST http://localhost:8000/api/ingest-all
```

After ingestion, `data/embeddings/files.json` and `data/embeddings/chunks/` will be populated.

---

## Docker

```bash
# Build
docker build -t rag-system:latest .

# Run (pass secrets as env vars — never bake .env into the image)
docker run -p 8000:8000 \
  -e USE_AZURE_OPENAI=true \
  -e AZURE_API_KEY=your_key \
  -e AZURE_ENDPOINT=https://*******.**.****.com \
  -e AZURE_DEPLOYMENT_NAME=gpt-4 \
  -e AZURE_EMBEDDING_MODEL=text-embedding-3-small-1 \
  rag-system:latest
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/ingest` | Ingest a single file |
| `POST` | `/api/ingest-all` | Ingest all files in a directory |
| `POST` | `/api/query` | Single-turn RAG query |
| `POST` | `/api/sessions` | Create a new chat session |
| `GET` | `/api/sessions` | List all sessions |
| `DELETE` | `/api/sessions/{id}` | Delete a session and all its messages |
| `POST` | `/api/sessions/{id}/messages` | Send a message and get a reply (multi-turn chat) |
| `GET` | `/api/sessions/{id}/messages` | Get full conversation history |

### POST `/api/ingest`
```json
{ "file_path": "data/raw/FastAPI_Framework.txt" }
```

### POST `/api/ingest-all`
```json
{}
```
> Body is optional — defaults to `data/raw/`. Pass `{"dir_path": "/custom/path"}` to override.

### POST `/api/query` — single-turn
```json
{ "query": "What is the difference between supervised and unsupervised learning?" }
```
**Response:**
```json
{
  "answer": "Supervised learning involves training on labeled data... Unsupervised learning identifies patterns in unlabeled data...",
  "sources": ["Machine_Learning.txt", "Deep_Learning.txt"]
}
```

### POST `/api/sessions` — create session
**Response (201):**
```json
{ "id": "f725ce69-c439-49c2-adab-37813e848b9e", "created_at": "2026-04-17T10:30:00" }
```

### POST `/api/sessions/{id}/messages` — chat turn

Full tool-calling loop on every request:
1. User message saved to SQLite
2. Last `HISTORY_LIMIT` messages loaded as context
3. LLM called with `search_kb` tool
4. LLM fires tool → two-stage RAG retrieval runs
5. Tool result (context + sources) fed back to LLM
6. LLM generates grounded answer
7. Assistant message saved to SQLite

```json
{ "message": "What is supervised learning?" }
```
**Response:**
```json
{
  "answer": "Supervised learning uses labeled data to train a model to predict outputs...",
  "sources": ["Machine_Learning.txt"]
}
```

### GET `/api/sessions/{id}/messages` — conversation history
```json
[
  { "id": 1, "session_id": "...", "role": "user",      "content": "What is supervised learning?", "created_at": "..." },
  { "id": 2, "session_id": "...", "role": "assistant", "content": "Supervised learning uses...",   "created_at": "..." }
]
```

> **Why POST for `/query`?** GET parameters have URL length limits and get cached. POST allows clean JSON bodies of any size and prevents caching of expensive LLM calls.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API framework | FastAPI + Uvicorn |
| LLM & Embeddings | OpenAI API (`gpt-4o-mini`, `text-embedding-3-small`) |
| Enterprise LLM | Azure OpenAI / DIAL (`gpt-4`, `text-embedding-3-small-1`) |
| Tool/function calling | OpenAI function calling (chat completions) |
| Vector similarity | NumPy (cosine similarity, matrix multiply) |
| Chat persistence | SQLite (`sqlite3` stdlib, no ORM) |
| Embedding storage | JSON files (no vector database) |
| Config | python-dotenv |
| Validation | Pydantic v2 |
| Containerisation | Docker (multi-stage build, non-root user) |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Two-stage retrieval | Satisfies the "max 2 files in context" constraint efficiently — O(n) file search then O(k) chunk search |
| Per-file chunk JSON | Only load chunks for selected files, not the entire KB |
| Batch embeddings | One API call per file during ingestion instead of N calls per chunk |
| `np.argpartition` | O(n) top-k selection vs O(n log n) full sort |
| No vector database | Self-contained, zero infrastructure dependencies |
| Absolute paths in `vector_store.py` | Works correctly regardless of which directory uvicorn is launched from |
| Provider flag in `.env` | Switch between OpenAI and Azure DIAL without any code changes |
| OpenAI tool/function calling | LLM decides what to search for and when — more flexible than always calling RAG |
| `force_tool_usage=true` default | Ensures KB is always consulted, preventing LLM from answering purely from training data |
| `tool_choice="none"` on second LLM call | After tool result is in context, forces the LLM to write the final answer without looping |
| SQLite for chat history | Zero infrastructure, built-in to Python stdlib, sufficient for development and small deployments |
| `HISTORY_LIMIT=6` | Last 6 messages give the LLM enough context for follow-up questions without exceeding token limits |
| Multi-stage Docker build | Compiler (`gcc`) used to build packages stays in builder stage only — smaller final image |
| Non-root Docker user | OWASP security best practice — process has no write access outside `/app` |

---

## Assumptions & Simplifications

| Area | What was done | Production alternative |
|------|--------------|------------------------|
| **Embedding storage** | Plain JSON files | Binary formats (pickle, numpy `.npy`, FAISS index) — faster, more compact |
| **File-level embedding** | Mean of all chunk embeddings | LLM-generated file summary embedding — better semantic representation |
| **Chunk strategy** | Fixed-size sliding window (character-based) | Sentence/paragraph-aware chunking for more coherent chunks |
| **Top-k values** | `top_k=2` files, `top_k=5` chunks (defaults) | Configurable per-query or tuned via evaluation |
| **No re-ingestion check** | Re-ingesting a file silently overwrites | Track file hashes; skip unchanged files, detect deletions |
| **No authentication** | All endpoints are open | API key header or OAuth2 for any shared/deployed instance |
| **Sequential ingestion** | Files ingested one by one | Parallel ingestion with `asyncio` or `ThreadPoolExecutor` |
| **In-memory file embeddings** | All file embeddings loaded per query | ANN index (FAISS, Hnswlib) for large KBs |
| **SQLite for chat** | Single-file DB, no ORM | PostgreSQL + async ORM (SQLAlchemy) for multi-instance deployments |
| **Single `search_kb` tool** | One retrieval tool per turn | Multiple specialised tools (e.g. `search_kb`, `get_file`, `summarise`) |

---

## Future Improvements

- Evaluation framework (precision, LLM-as-judge metrics)
- Pluggable vector store (FAISS, ChromaDB)
- LLM-based file summarization for better file-level embeddings
- Hybrid search (BM25 + dense)
- Streaming responses (`text/event-stream`)
- Re-ranking with a cross-encoder
- File hash tracking to skip re-ingestion of unchanged files
- Authentication on API endpoints
- Parallel ingestion for large knowledge bases
- Session expiry / TTL for chat history
