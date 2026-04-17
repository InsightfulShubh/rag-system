# RAG System - Copilot Instructions

## Objective
Implement a simple Retrieval-Augmented Generation (RAG) system using FastAPI and OpenAI APIs without using external frameworks like LangChain.

---

## Constraints
- Do NOT use LangChain or similar libraries
- Do NOT use ORM (e.g., SQLAlchemy)
- Use only OpenAI API for embeddings and LLM
- No UI required, only API endpoints
- Only 2 files can be used in LLM context

---

## Project Structure

rag-system/
│
├── app/
│   ├── main.py                # FastAPI entrypoint
│
│   ├── routes/
│   │   ├── ingest.py          # /api/ingest endpoint
│   │   └── query.py           # /api/query endpoint
│
│   ├── services/
│   │   ├── ingestion.py       # file processing + embedding generation
│   │   ├── retrieval.py       # core RAG logic (file + chunk retrieval)
│   │   └── llm.py             # OpenAI LLM calls
│
│   ├── storage/
│   │   ├── vector_store.py    # save/load embeddings
│   │   └── file_store.py      # file handling utilities
│
│   ├── utils/
│   │   ├── embedding.py       # OpenAI embedding wrapper
│   │   ├── chunking.py        # text splitting logic
│   │   └── similarity.py      # cosine similarity
│
│   └── models/
│       └── schemas.py         # request/response models (optional)
│
├── data/
│   ├── raw/                   # input text files
│   └── embeddings/            # stored embeddings (JSON)
│
├── .env
├── requirements.txt
└── README.md

---

## Architecture Overview

### Two-Stage Retrieval Strategy

#### Stage 1: File-Level Retrieval
- Each file has an embedding
- File embedding = average of chunk embeddings
- Retrieve top 2 files using cosine similarity

#### Stage 2: Chunk-Level Retrieval
- Load chunks only from selected files
- Rank chunks using cosine similarity
- Select top chunks for LLM context

---

## Data Flow

### Ingestion Pipeline
1. Read file from disk
2. Split into chunks (chunk_size ~500, overlap ~50)
3. Generate embeddings using OpenAI API
4. Store:
   - Chunk embeddings
   - File embedding (mean of chunks)

### Query Pipeline
1. Convert query to embedding
2. Retrieve top 2 files using file embeddings
3. Load chunks from selected files
4. Rank chunks and select top K
5. Send context + query to OpenAI LLM
6. Return answer + source files

---

## Storage Strategy

- Raw files → stored on disk (`data/raw`)
- Chunk embeddings → stored in JSON (`data/embeddings/chunks.json`)
- File embeddings → stored in JSON (`data/embeddings/files.json`)
- File embeddings loaded into memory at runtime
- Chunk embeddings loaded only for selected files

---

## Key Implementation Details

- Use cosine similarity for vector comparison
- Embedding model: `text-embedding-3-small`
- LLM model: `gpt-4o-mini`
- Keep implementation simple and modular

---

## API Design

### POST /api/ingest
- Input: file_path
- Output: number of chunks processed

### POST /api/query
- Input: query
- Output:
  - answer
  - sources (list of file names)

---

## Coding Guidelines

- Keep API routes thin
- Place all business logic in `services/`
- Do NOT introduce unnecessary dependencies
- Avoid over-engineering
- Use simple file-based storage instead of database

---

## Future Improvements (DO NOT IMPLEMENT NOW)

- Replace brute-force search with FAISS
- Add hybrid search (keyword + semantic)
- Add caching for frequent queries