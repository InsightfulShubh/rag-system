from fastapi import APIRouter, HTTPException
from app.services.session_service import SessionService
from app.services.chat_service import ChatService
from app.models.schemas import MessageRequest, MessageResponse, MessageRecord
from app.storage import db

router = APIRouter(prefix="/sessions", tags=["Multi-Turn Chat — Sessions & History"])
session_service = SessionService()
chat_service = ChatService()


@router.post("", status_code=201)
def create_session():
    """Create a new conversation session. Returns {id, created_at}."""
    return session_service.create_session()


@router.get("")
def get_sessions():
    """List all sessions, newest first."""
    return session_service.get_sessions()


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: int):
    """Delete a session and all its messages. Returns 404 if not found."""
    deleted = session_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@router.post("/{session_id}/messages", response_model=MessageResponse)
def send_message(session_id: int, request: MessageRequest):
    """Send a user message and return the assistant's reply with sources."""
    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    result = chat_service.chat(session_id, request.message)
    return MessageResponse(answer=result["answer"], sources=result["sources"])


@router.get("/{session_id}/messages", response_model=list[MessageRecord])
def get_messages(session_id: int):
    """Return the full conversation history for a session (all messages, oldest first)."""
    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return db.get_messages(session_id, limit=1000)
