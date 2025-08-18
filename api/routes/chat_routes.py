# api/routes/chat_routes.py

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime

from database.connection import get_db
from api.middleware.auth import get_current_user, get_current_active_user
from api.controllers.chat_controller import get_chat_controller
from database.models import User

router = APIRouter(prefix="/chat", tags=["chat"])

# Request/Response Models
class ChatSettings(BaseModel):
    """Chat configuration settings."""
    llm_model: str = Field(default="openai-gpt35", description="LLM model to use")
    embedding_model: str = Field(default="hf-mpnet-base-v2", description="Embedding model for RAG")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response randomness")
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="Maximum response tokens")
    use_rag: bool = Field(default=True, description="Whether to use RAG for context")
    search_type: str = Field(default="semantic", description="Type of search (semantic, hybrid, basic)")
    top_k_documents: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")

class CreateSessionRequest(BaseModel):
    """Request to create a new chat session."""
    settings: Optional[ChatSettings] = None

class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""
    message: str = Field(..., min_length=1, max_length=4000, description="Message content")
    session_id: Optional[str] = None
    settings: Optional[ChatSettings] = None

class SessionResponse(BaseModel):
    """Response for session operations."""
    session_id: str
    created_at: str
    settings: Dict[str, Any]

class MessageResponse(BaseModel):
    """Response for message operations."""
    id: str
    type: str
    content: str
    timestamp: str
    model: Optional[str] = None
    sources: Optional[list] = None

class ChatHistoryResponse(BaseModel):
    """Response for chat history."""
    session_id: str
    messages: list
    total_messages: int
    created_at: str
    last_activity: str
    settings: Dict[str, Any]

class SessionListResponse(BaseModel):
    """Response for listing sessions."""
    sessions: list
    total_sessions: int

class ModelListResponse(BaseModel):
    """Response for available models."""
    llm_models: list
    embedding_models: list

@router.get("/models", response_model=ModelListResponse)
async def get_available_models(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get available LLM and embedding models."""
    try:
        controller = get_chat_controller(current_user.id)
        models = await controller.get_available_models(current_user, db)
        return ModelListResponse(**models)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models: {str(e)}"
        )

@router.post("/sessions", response_model=SessionResponse)
async def create_chat_session(
    request: CreateSessionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new chat session."""
    try:
        settings = request.settings.dict() if request.settings else {}
        controller = get_chat_controller(current_user.id)
        session_info = await controller.create_chat_session(
            user=current_user,
            settings=settings,
            db=db
        )
        return SessionResponse(**session_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )

@router.get("/sessions", response_model=SessionListResponse)
async def list_chat_sessions(
    current_user: User = Depends(get_current_active_user)
):
    """List active chat sessions for the current user."""
    try:
        sessions_info = await get_chat_controller(current_user.id).list_chat_sessions(current_user)
        return SessionListResponse(**sessions_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}"
        )

@router.get("/sessions/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user)
):
    """Get chat history for a specific session."""
    try:
        history = await get_chat_controller(current_user.id).get_chat_history(
            session_id=session_id,
            user=current_user,
            limit=limit,
            db=db
        )
        return ChatHistoryResponse(**history)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat history: {str(e)}"
        )

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a chat session."""
    try:
        result = await get_chat_controller(current_user.id).delete_chat_session(
            session_id=session_id,
            user=current_user,
            db=db
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
        )

@router.post("/stream")
async def stream_chat_response(
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Stream chat response using Server-Sent Events."""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            settings = request.settings.dict() if request.settings else {}
            session_info = await get_chat_controller(current_user.id).create_chat_session(
                user=current_user,
                settings=settings,
                db=db
            )
            session_id = session_info["session_id"]
        
        # Create streaming response
        async def event_stream():
            try:
                async for chunk in get_chat_controller(current_user.id).process_chat_message(
                    session_id=session_id,
                    message=request.message,
                    user=current_user,
                    db=db
                ):
                    yield chunk
            except Exception as e:
                error_data = {
                    "type": "error",
                    "error": str(e)
                }
                yield f"data: {str(error_data)}\n\n"
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )

@router.post("/message")
async def send_chat_message(
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Send a chat message and get a complete response (non-streaming)."""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            settings = request.settings.dict() if request.settings else {}
            session_info = await get_chat_controller(current_user.id).create_chat_session(
                user=current_user,
                settings=settings,
                db=db
            )
            session_id = session_info["session_id"]
        
        # Collect streaming response
        response_content = ""
        sources = []
        
        async for chunk in get_chat_controller(current_user.id).process_chat_message(
            session_id=session_id,
            message=request.message,
            user=current_user,
            db=db
        ):
            if chunk.startswith("data: "):
                try:
                    import json
                    data = json.loads(chunk[6:])
                    if data.get("type") == "content":
                        response_content += data.get("content", "")
                    elif data.get("type") == "sources":
                        sources = data.get("sources", [])
                except:
                    pass
        
        return {
            "session_id": session_id,
            "response": response_content,
            "sources": sources,
            "timestamp": str(datetime.utcnow())
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )

# Model management routes
@router.post("/models/health-check")
async def run_model_health_check(
    current_user: User = Depends(get_current_active_user)
):
    """Run health check on available models."""
    try:
        from vector_db.embedding_model_registry import get_embedding_model_registry
        
        registry = get_embedding_model_registry()
        health_results = await registry.health_check_models()
        
        return {
            "health_check_results": health_results,
            "timestamp": str(datetime.utcnow())
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/models/embedding/recommendations")
async def get_embedding_model_recommendations(
    use_case: str = "general",
    max_results: int = 3,
    current_user: User = Depends(get_current_active_user)
):
    """Get recommended embedding models for a specific use case."""
    try:
        from vector_db.embedding_model_registry import get_embedding_model_registry
        
        registry = get_embedding_model_registry()
        recommendations = registry.get_recommended_models(use_case, max_results)
        
        return {
            "use_case": use_case,
            "recommendations": [
                {
                    "model_id": model.model_id,
                    "display_name": model.display_name,
                    "provider": model.provider.value,
                    "description": model.description,
                    "quality_score": model.quality_score,
                    "performance_tier": model.performance_tier
                }
                for model in recommendations
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )

# Session cleanup endpoint (admin only)
@router.post("/admin/cleanup-sessions")
async def cleanup_old_sessions(
    max_age_hours: int = 24,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Clean up old inactive chat sessions (admin only)."""
    try:
        # Check if user is admin
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        await get_chat_controller(current_user.id).cleanup_old_sessions(max_age_hours, db)
        
        return {
            "message": f"Cleaned up sessions older than {max_age_hours} hours",
            "timestamp": str(datetime.utcnow())
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup failed: {str(e)}"
        )