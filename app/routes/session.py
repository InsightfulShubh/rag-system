from fastapi import APIRouter, HTTPException
from app.services.session_service import SessionService
from app.services.chat_service import ChatService
from app.models.schemas import MessageRequest, MessageResponse, MessageRecord
from app.storage import db

router = APIRouter(prefix="/sessions", tags=["sessions"])
session_service = SessionService()
chat_service = ChatService()


@router.post("", status_code=201)
def create_session():
    """
    Create a new conversation session.

    Returns:
        {id, created_at} — the new session
    """
    return session_service.create_session()


@router.get("")
def get_sessions():
    """
    List all sessions, newest first.

    Returns:
        list of {id, created_at}
    """
    return session_service.get_sessions()


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str):
    """
    Delete a session and all its messages.

    Returns:
        204 No Content on success
        404 if session not found
    """
    deleted = session_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@router.post("/{session_id}/messages", response_model=MessageResponse)
def send_message(session_id: str, request: MessageRequest):
    """
    Send a user message and get the assistant's reply.

    Full flow (handled inside ChatService):
        1. Save user message to DB
        2. Load last 6 messages as history
        3. Send to LLM with search_kb tool
        4. LLM calls tool → RAG retrieval
        5. LLM generates final answer
        6. Save assistant message to DB
        7. Return answer + sources

    Returns:
        {answer, sources}
    """
    # Guard: return 404 if session doesn't exist
    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    result = chat_service.chat(session_id, request.message)
    return MessageResponse(answer=result["answer"], sources=result["sources"])


@router.get("/{session_id}/messages", response_model=list[MessageRecord])
def get_messages(session_id: str):
    """
    Return the full conversation history for a session
    (all messages, oldest first).

    Returns:
        list of {id, session_id, role, content, created_at}
    """
    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    # Pass a high limit to get full history for display
    # (HISTORY_LIMIT=6 only applies when building LLM context)
    return db.get_messages(session_id, limit=1000)
