# RAG System

A lightweight RAG API built with FastAPI and the OpenAI API. No LangChain, no vector database, no ORM. Supports single-turn queries and multi-turn chat sessions with tool calling and SQLite-persisted history.

---

## Overview

Given a directory of `.txt` files, the system answers questions by finding the most relevant content and generating a grounded LLM answer. At most 2 files are passed to the LLM per query — retrieval accuracy is the critical constraint.

---

## Architecture

**Two-stage retrieval:**
1. **Stage 1 — File level:** Vectorized cosine similarity (NumPy) across all file embeddings → select top 2 files
2. **Stage 2 — Chunk level:** Load only those 2 files' chunks → cosine similarity → select top 5 chunks

**Multi-turn chat (tool calling):**
1. Save user message → load last `HISTORY_LIMIT` messages from SQLite
2. LLM call #1 with `search_kb` tool (`tool_choice="required"`)
3. Execute retrieval → feed context back as `tool` role message
4. LLM call #2 (`tool_choice="none"`) → final grounded answer
5. Save assistant message → return `{answer, sources}`

---

## Grounding Guarantee

All responses are grounded in the knowledge base:

- `FORCE_TOOL_USAGE=true` ensures the LLM always calls `search_kb`
- The model is forced to use retrieved context, significantly reducing reliance on pretrained knowledge
- If relevant context is not found, the model is instructed to return "I don't know"

This prevents hallucination and ensures traceability of answers.

---

## How to Run

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd rag-system
uv sync                  # create venv + install deps
cp .env-sample .env      # fill in API key
```

**Start the server:**
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

**Ingest documents (first run):**
```bash
# Place .txt files in data/raw/, then:
curl -X POST http://localhost:8000/api/ingest-all
```

**Run tests:**
```bash
uv sync --all-extras     # install pytest
uv run pytest
```
**Note:** Tests use real LLM calls configured via environment variables. No mock responses are used, as per task requirements.

**Docker:**
```bash
docker build -t rag-system:latest .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key rag-system:latest
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/ingest` | Ingest a single file |
| `POST` | `/api/ingest-all` | Ingest all files in `data/raw/` |
| `POST` | `/api/query` | (Optional) Single-turn query for debugging; main interface is session-based |
| `POST` | `/api/sessions` | Create a chat session |
| `GET` | `/api/sessions` | List sessions |
| `DELETE` | `/api/sessions/{id}` | Delete session and messages |
| `POST` | `/api/sessions/{id}/messages` | Send a message, get a reply |
| `GET` | `/api/sessions/{id}/messages` | Get conversation history |

**Key env vars** (see `.env-sample` for full list):
```env
USE_AZURE_OPENAI=false        # true to use Azure DIAL instead
OPENAI_API_KEY=...
FORCE_TOOL_USAGE=true         # always call search_kb (recommended)
HISTORY_LIMIT=6               # messages passed as LLM context per turn
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Two-stage retrieval | File-level similarity narrows search space to 2 files, then chunk-level retrieval selects the most relevant passages |
| JSON embedding storage | No external infra; read-heavy workload at this scale; one file per doc for lazy loading |
| `np.argpartition` | O(n) top-k selection vs O(n log n) full sort |
| No vector database | Self-contained, zero infrastructure dependencies |
| `FORCE_TOOL_USAGE=true` | Ensures KB is consulted for every query, improving grounding and reducing reliance on pretrained knowledge |
| `tool_choice="none"` (2nd call) | Forces a text answer after tool result is in context, ending the loop |
| SQLite + `INTEGER AUTOINCREMENT` | Zero infrastructure, stdlib only; AUTOINCREMENT gives clean integer session IDs |
| Absolute paths in `vector_store.py` | Works from any working directory uvicorn is launched from |
| Provider flag in `.env` | Switch OpenAI ↔ Azure DIAL with no code changes |
| No LangChain | Full control over retrieval logic, prompt construction, and tool loop; 7 direct dependencies total |

---

## Why JSON for Embeddings?

Embeddings are stored as JSON files to keep the system simple and self-contained:

- No external infrastructure required
- Read-heavy workload suits file-based storage
- Enables lazy loading (only selected files are loaded per query)

For production-scale systems, binary formats or vector indexes would be preferred.

---

## Evaluation (Optional)

Retrieval quality can be evaluated using metrics like  precision and llm as judge validation of LLM responses.

---

## Limitations

- Brute-force similarity search does not scale to very large datasets
- JSON-based embedding storage is not optimal for high-volume systems
- Retrieval quality depends on chunking strategy and embedding model performance

For large-scale systems, this can be improved using ANN indexes (e.g., FAISS) or vector databases.

---

## Scaling Path
This ensures the system remains modular and can evolve without major architectural changes.
For large knowledge bases, the retrieval layer can be replaced with a vector index (e.g., FAISS) or a vector database without changing the API or overall architecture.

---

## Constraints Followed

- Max 2 files passed as context to LLM per query
- No RAG frameworks (no LangChain, LlamaIndex, etc.)
- No vector database (ChromaDB, Pinecone, etc.)
- No vector index libraries (e.g., FAISS)
- No ORM (raw `sqlite3` only)
- Grounded answers only — `FORCE_TOOL_USAGE=true` + system prompt instructs LLM to say "I don't know" if answer is not in context
- Non-root Docker user (OWASP best practice)

---


## Future Improvements

- Replace brute-force similarity with an ANN index (e.g., FAISS) for large-scale retrieval
- Move from JSON embedding storage to a vector database (e.g., ChromaDB or Postgres + pgvector) for persistence and filtering
- Introduce hybrid search (keyword + semantic) to improve retrieval accuracy
- Add result re-ranking (cross-encoder) for higher answer quality
- Cache frequent queries and embeddings to reduce latency