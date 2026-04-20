from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.retrieval import RetrievalService

router = APIRouter(tags=["Single-Turn Query — Testing/Debugging Only"])
retrieval_service = RetrievalService()


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using two-stage RAG retrieval.

    Pipeline:
        1. Embed the query (1 OpenAI embedding call)
        2. Stage 1: rank all files by cosine similarity, pick top 2
        3. Stage 2: rank chunks from top 2 files, pick top 5
        4. Send context + query to LLM (1 OpenAI chat completion call)
        5. Return answer + source file names

    Total OpenAI calls per query: 2 (1 embedding + 1 chat completion)
    """
    result = retrieval_service.retrieve_and_answer(request.query)
    return QueryResponse(answer=result["answer"], sources=result["sources"])
