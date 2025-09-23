# api/routes/chat_routes.py

from typing import Dict, Any, Optional
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from database.connection import get_db
from api.middleware.auth import get_current_user, get_current_active_user
from api.controllers.chat_controller import get_chat_controller
from database.models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# Request/Response Models
class ChatSettings(BaseModel):
    """Chat configuration settings."""
    llm_model: str = Field(default="openai-gpt35", description="LLM model to use")
    embedding_model: str = Field(default="hf-minilm-l6-v2", description="Embedding model for RAG")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response randomness")
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="Maximum response tokens")
    use_rag: bool = Field(default=True, description="Whether to use RAG for context")
    search_type: str = Field(default="semantic", description="Type of search (semantic, hybrid, basic)")
    top_k_documents: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")

    # Search filter settings (missing fields that frontend sends)
    tags: Optional[list] = Field(default=[], description="Tags to filter by")
    tag_match_mode: str = Field(default="any", description="Tag matching mode (any, all, exact)")
    exclude_tags: Optional[list] = Field(default=[], description="Tags to exclude")
    file_type: Optional[list] = Field(default=[], description="File types to filter by")
    language: Optional[str] = Field(default=None, description="Language filter")
    languages: Optional[list] = Field(default=[], description="Languages filter (array)")
    is_public: Optional[bool] = Field(default=None, description="Public/private filter")
    min_score: Optional[float] = Field(default=None, description="Minimum similarity score")
    max_results: int = Field(default=20, description="Maximum number of results")
    file_size_range: Optional[list] = Field(default=None, description="File size range filter")
    date_range: Optional[dict] = Field(default=None, description="Date range filter")

    # Reranker settings
    enable_reranking: Optional[bool] = Field(default=False, description="Enable reranking")
    reranker_model: Optional[str] = Field(default=None, description="Reranker model to use")
    rerank_score_weight: Optional[float] = Field(default=0.5, description="Reranker score weight")
    min_rerank_score: Optional[float] = Field(default=None, description="Minimum rerank score")

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
        logger.info(f"Creating session for user {current_user.id}")
        logger.debug(f"Raw request.settings: {request.settings}")

        settings = request.settings.model_dump() if request.settings else {}
        logger.info(f"Processed settings: {settings}")

        # Log specifically the embedding model setting
        embedding_model = settings.get('embedding_model', 'NOT_SET')
        logger.info(f"Embedding model in processed settings: '{embedding_model}'")

        controller = get_chat_controller(current_user.id)
        session_info = await controller.create_chat_session(
            user=current_user,
            settings=settings,
            db=db
        )

        logger.info(f"Session created: {session_info['session_id']}")
        logger.debug(f"Final session settings: {session_info.get('settings', {})}")

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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
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
        logger.info(f"Processing message for user {current_user.id}")
        logger.debug(f"Request settings: {request.settings}")

        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            logger.info(f"Creating new session with settings")
            settings = request.settings.model_dump() if request.settings else {}
            embedding_model = settings.get('embedding_model', 'NOT_SET')
            logger.info(f"Embedding model for new session: '{embedding_model}'")

            session_info = await get_chat_controller(current_user.id).create_chat_session(
                user=current_user,
                settings=settings,
                db=db
            )
            session_id = session_info["session_id"]
        else:
            logger.info(f"Using existing session: {session_id}")
            if request.settings:
                embedding_model = request.settings.embedding_model if hasattr(request.settings, 'embedding_model') else 'NOT_SET'
                logger.info(f"Settings provided for existing session (will be applied): embedding_model='{embedding_model}'")
        
        # Create streaming response
        async def event_stream():
            try:
                # Prepare settings override if provided
                settings_override = None
                if request.settings:
                    settings_override = request.settings.model_dump()
                    logger.debug(f"Passing settings override: {settings_override}")

                async for chunk in get_chat_controller(current_user.id).process_chat_message(
                    session_id=session_id,
                    message=request.message,
                    user=current_user,
                    db=db,
                    settings_override=settings_override
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
            settings = request.settings.model_dump() if request.settings else {}
            session_info = await get_chat_controller(current_user.id).create_chat_session(
                user=current_user,
                settings=settings,
                db=db
            )
            session_id = session_info["session_id"]
        
        # Collect streaming response
        response_content = ""
        sources = []

        # Prepare settings override if provided
        settings_override = None
        if request.settings:
            settings_override = request.settings.model_dump()

        async for chunk in get_chat_controller(current_user.id).process_chat_message(
            session_id=session_id,
            message=request.message,
            user=current_user,
            db=db,
            settings_override=settings_override
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
            "timestamp": str(datetime.now(timezone.utc))
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
            "timestamp": str(datetime.now(timezone.utc))
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
            "timestamp": str(datetime.now(timezone.utc))
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup failed: {str(e)}"
        )