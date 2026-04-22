from fastapi import APIRouter
from app.models.schemas import QueryRequest, QueryResponse
from app.services.retrieval import RetrievalService

router = APIRouter(tags=["Single-Turn Query — Testing/Debugging Only"])
retrieval_service = RetrievalService()


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Single-turn RAG query: embed → top-2 files → top-5 chunks → LLM answer.
    Returns answer and source file names. Use sessions for multi-turn chat.
    """
    result = retrieval_service.retrieve_and_answer(request.query)
    return QueryResponse(answer=result["answer"], sources=result["sources"])
