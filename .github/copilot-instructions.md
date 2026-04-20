# RAG Chat Agent System - Copilot Instructions

## Objective
Extend the existing RAG system into a chat-based QnA agent API using FastAPI and OpenAI APIs.

The system must support:
- Chat sessions
- Conversation history
- Tool/function calling for knowledge base retrieval
- SQLite-based persistence

---

## Constraints
- Do NOT use LangChain or similar frameworks
- Do NOT use ORM (e.g., SQLAlchemy)
- Use only OpenAI API
- Knowledge base must be file-based
- Maximum 2 files allowed in LLM context
- Keep implementation simple and modular

---

## Architecture Overview

### Previous System
User → RAG → LLM → Answer

### Updated System

User → API → Session (SQLite)
             ↓
        LLM (tool calling)
             ↓
        search_kb()  ← existing RAG logic
             ↓
        LLM → Answer
             ↓
        Store messages

---

## Core Components

### 1. Session Management
- Each session represents a conversation
- Store sessions and messages in SQLite

Tables:

sessions:
- id (TEXT, primary key)
- created_at (timestamp)

messages:
- id (TEXT)
- session_id (TEXT)
- role (user / assistant)
- content (TEXT)
- created_at (timestamp)

---

### 2. Conversation Context

- Load last N messages from DB
- Pass history to LLM

Example:

messages = [
  {"role": "system", ...},
  *history,
  {"role": "user", query}
]

---

### 3. Tool / Function Calling (CRITICAL)

LLM must call a tool to retrieve knowledge.

Define tool:

search_kb(query: string)

Flow:
1. Send user query to LLM with tool definition
2. LLM decides to call tool
3. Backend executes search_kb (existing RAG)
4. Return context to LLM
5. LLM generates final answer

---

### 4. RAG Integration

- Keep existing retrieval logic unchanged
- Wrap it inside a function:

def search_kb(query: str) -> str:
    return context_chunks

---

### 5. Storage Strategy

- SQLite → sessions + messages
- JSON → embeddings (keep existing)

---

## API Design

### Sessions

POST /sessions  
GET /sessions  
DELETE /sessions/{id}

---

### Messages

POST /sessions/{id}/messages  
GET /sessions/{id}/messages  

---

## Data Flow

1. User sends message
2. Store user message
3. Load session history
4. Send to LLM with tool definition
5. If tool called:
    - execute search_kb
    - send result back to LLM
6. LLM generates final answer
7. Store assistant message
8. Return response

---

## Coding Guidelines

- Keep RAG logic in `services/retrieval.py`
- Add DB logic in new `storage/db.py`
- Keep API routes thin
- Do NOT rewrite existing RAG logic
- Extend, do not replace

---

## Future Improvements (Optional)

- WebSocket for real-time updates
- Docker containerization
- pytest for API tests