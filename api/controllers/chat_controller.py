# api/controllers/chat_controller.py

from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import json
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from sqlalchemy import and_, desc

from database.models import User, Document, DocumentChunk
from llm import create_model_manager_with_defaults
from vector_db.enhanced_embedding_manager import EnhancedEmbeddingManager
from vector_db.embedding_model_registry import get_embedding_model_registry, EmbeddingProvider
from vector_db.enhanced_search_engine import EnhancedSearchEngine
from database.connection import get_db
from utils.security.audit_logger import log_user_action

logger = logging.getLogger(__name__)

class ChatSession:
    """Chat session management."""
    def __init__(self, session_id: str, user_id: int, settings: Dict[str, Any]):
        self.session_id = session_id
        self.user_id = user_id
        self.settings = settings
        self.messages: List[Dict[str, Any]] = []
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()

class ChatController:
    """Controller for chat functionality with RAG support."""
    
    def __init__(self):
        self.model_manager = create_model_manager_with_defaults()
        self.embedding_registry = get_embedding_model_registry()
        self.search_engine = None  # Will be initialized per request
        self.active_sessions: Dict[str, ChatSession] = {}
        
    async def get_available_models(self, user: User) -> Dict[str, Any]:
        """Get available LLM and embedding models."""
        try:
            # Get registered LLM models
            llm_models = []
            for model_id in self.model_manager.list_registered_models():
                config = self.model_manager.get_model_config(model_id)
                if config:
                    llm_models.append({
                        "id": model_id,
                        "name": config.model_name,
                        "display_name": config.model_name,
                        "provider": self._get_provider_from_model_id(model_id),
                        "description": f"Max tokens: {config.max_tokens}, Temperature: {config.temperature}",
                        "max_tokens": config.max_tokens,
                        "temperature": config.temperature
                    })
            
            # Get registered embedding models
            embedding_models = []
            available_embedding_models = self.embedding_registry.list_models()
            
            for model in available_embedding_models:
                embedding_models.append({
                    "id": model.model_id,
                    "name": model.model_name,
                    "display_name": model.display_name,
                    "provider": model.provider.value,
                    "description": model.description,
                    "embedding_dimension": model.embedding_dimension,
                    "performance_tier": model.performance_tier,
                    "quality_score": model.quality_score,
                    "use_cases": model.use_cases,
                    "language_support": model.language_support,
                    "model_size_mb": model.model_size_mb,
                    "memory_requirements_mb": model.memory_requirements_mb,
                    "gpu_required": model.gpu_required,
                    "api_cost_per_1k_tokens": model.api_cost_per_1k_tokens
                })
            
            return {
                "llm_models": llm_models,
                "embedding_models": embedding_models
            }
            
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get available models: {str(e)}"
            )

    def _get_provider_from_model_id(self, model_id: str) -> str:
        """Extract provider name from model ID."""
        if model_id.startswith("openai-"):
            return "openai"
        elif model_id.startswith("gemini-"):
            return "gemini"
        elif model_id.startswith("ollama-"):
            return "ollama"
        elif model_id.startswith("lmstudio-"):
            return "lmstudio"
        else:
            return "unknown"

    async def create_chat_session(
        self,
        user: User,
        settings: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Create a new chat session."""
        try:
            session_id = str(uuid.uuid4())
            
            # Validate settings
            validated_settings = await self._validate_chat_settings(settings, user, db)
            
            # Create session
            session = ChatSession(session_id, user.id, validated_settings)
            self.active_sessions[session_id] = session
            
            # Log session creation
            await log_user_action(
                user_id=user.id,
                action="chat_session_created",
                resource_id=session_id,
                details={"settings": validated_settings},
                db=db
            )
            
            return {
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
                "settings": validated_settings
            }
            
        except Exception as e:
            logger.error(f"Failed to create chat session: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create chat session: {str(e)}"
            )

    async def _validate_chat_settings(
        self,
        settings: Dict[str, Any],
        user: User,
        db: Session
    ) -> Dict[str, Any]:
        """Validate and normalize chat settings."""
        # Default settings
        default_settings = {
            "llm_model": "openai-gpt35",
            "embedding_model": "hf-mpnet-base-v2",
            "temperature": 0.7,
            "max_tokens": 2048,
            "use_rag": True,
            "search_type": "semantic",
            "top_k_documents": 5
        }
        
        # Merge with provided settings
        validated_settings = {**default_settings, **settings}
        
        # Validate LLM model
        if validated_settings["llm_model"] not in self.model_manager.list_registered_models():
            # Fallback to first available model
            available_models = self.model_manager.list_registered_models()
            if available_models:
                validated_settings["llm_model"] = available_models[0]
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No LLM models available"
                )
        
        # Validate embedding model
        embedding_model = self.embedding_registry.get_model(validated_settings["embedding_model"])
        if not embedding_model:
            # Fallback to first available model
            available_models = self.embedding_registry.list_models()
            if available_models:
                validated_settings["embedding_model"] = available_models[0].model_id
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No embedding models available"
                )
        
        # Validate numeric parameters
        validated_settings["temperature"] = max(0.0, min(2.0, float(validated_settings["temperature"])))
        validated_settings["max_tokens"] = max(1, min(8192, int(validated_settings["max_tokens"])))
        validated_settings["top_k_documents"] = max(1, min(20, int(validated_settings["top_k_documents"])))
        
        # Validate search type
        if validated_settings["search_type"] not in ["semantic", "hybrid", "basic"]:
            validated_settings["search_type"] = "semantic"
        
        return validated_settings

    async def process_chat_message(
        self,
        session_id: str,
        message: str,
        user: User,
        db: Session
    ) -> AsyncGenerator[str, None]:
        """Process a chat message and yield streaming responses."""
        try:
            # Get session
            session = self.active_sessions.get(session_id)
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )
            
            # Verify session ownership
            if session.user_id != user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to chat session"
                )
            
            # Update session activity
            session.last_activity = datetime.utcnow()
            
            # Add user message to session
            user_message = {
                "id": str(uuid.uuid4()),
                "type": "user",
                "content": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            session.messages.append(user_message)
            
            # Initialize search engine if using RAG
            search_results = []
            if session.settings["use_rag"]:
                search_results = await self._perform_rag_search(
                    message, session.settings, user, db
                )
                
                # Send sources information
                if search_results:
                    sources_data = {
                        "type": "sources",
                        "sources": [
                            {
                                "document_id": result["document_id"],
                                "filename": result["filename"],
                                "chunk_id": result["chunk_id"],
                                "similarity_score": result["similarity_score"],
                                "text_snippet": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
                            }
                            for result in search_results[:session.settings["top_k_documents"]]
                        ]
                    }
                    yield f"data: {json.dumps(sources_data)}\n\n"
            
            # Prepare context for LLM
            context = self._prepare_llm_context(message, search_results, session)
            
            # Generate response using selected LLM
            response_content = ""
            async for chunk in self.model_manager.generate_with_fallback(
                prompt=context,
                primary_model_id=session.settings["llm_model"],
                stream=True
            ):
                if isinstance(chunk, str):
                    response_content += chunk
                    chunk_data = {
                        "type": "content",
                        "content": chunk
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Add assistant message to session
            assistant_message = {
                "id": str(uuid.uuid4()),
                "type": "assistant",
                "content": response_content,
                "timestamp": datetime.utcnow().isoformat(),
                "model": session.settings["llm_model"],
                "sources": search_results[:session.settings["top_k_documents"]] if search_results else []
            }
            session.messages.append(assistant_message)
            
            # Send completion signal
            completion_data = {
                "type": "done",
                "message_id": assistant_message["id"]
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            
            # Log the interaction
            await log_user_action(
                user_id=user.id,
                action="chat_message_processed",
                resource_id=session_id,
                details={
                    "message_length": len(message),
                    "response_length": len(response_content),
                    "rag_used": session.settings["use_rag"],
                    "sources_count": len(search_results),
                    "model_used": session.settings["llm_model"]
                },
                db=db
            )
            
        except Exception as e:
            logger.error(f"Failed to process chat message: {str(e)}")
            error_data = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    async def _perform_rag_search(
        self,
        query: str,
        settings: Dict[str, Any],
        user: User,
        db: Session
    ) -> List[Dict[str, Any]]:
        """Perform RAG search to find relevant documents."""
        try:
            # Create embedding manager for the selected model
            embedding_model = self.embedding_registry.get_model(settings["embedding_model"])
            if not embedding_model:
                logger.warning(f"Embedding model {settings['embedding_model']} not found")
                return []
            
            # Create appropriate embedding manager
            if embedding_model.provider == EmbeddingProvider.HUGGINGFACE:
                embedding_manager = EnhancedEmbeddingManager.create_huggingface_manager(
                    embedding_model.model_name
                )
            elif embedding_model.provider == EmbeddingProvider.OLLAMA:
                embedding_manager = EnhancedEmbeddingManager.create_ollama_manager(
                    model_name=embedding_model.model_name
                )
            elif embedding_model.provider == EmbeddingProvider.OPENAI:
                from config import get_settings
                settings_config = get_settings()
                if not settings_config.OPENAI_API_KEY:
                    logger.warning("OpenAI API key not configured")
                    return []
                embedding_manager = EnhancedEmbeddingManager.create_openai_manager(
                    api_key=settings_config.OPENAI_API_KEY,
                    model_name=embedding_model.model_name
                )
            else:
                logger.warning(f"Unsupported embedding provider: {embedding_model.provider}")
                return []
            
            # Initialize search engine
            if not self.search_engine:
                from vector_db.storage_manager import VectorStorageManager
                storage_manager = VectorStorageManager()
                self.search_engine = EnhancedSearchEngine(storage_manager, embedding_manager)
            
            # Perform search
            search_results = await self.search_engine.search_with_context(
                query=query,
                search_type=settings["search_type"],
                user_id=user.id,
                top_k=settings["top_k_documents"] * 2,  # Get more results for filtering
                db=db
            )
            
            # Format results
            formatted_results = []
            for result in search_results:
                # Get document info
                document = db.query(Document).filter(
                    and_(
                        Document.id == result.get("document_id"),
                        Document.user_id == user.id,
                        Document.is_deleted == False
                    )
                ).first()
                
                if document:
                    formatted_results.append({
                        "document_id": document.id,
                        "filename": document.filename,
                        "chunk_id": result.get("chunk_id", ""),
                        "text": result.get("text", ""),
                        "similarity_score": result.get("similarity_score", 0.0),
                        "metadata": result.get("metadata", {})
                    })
            
            return formatted_results[:settings["top_k_documents"]]
            
        except Exception as e:
            logger.error(f"RAG search failed: {str(e)}")
            return []

    def _prepare_llm_context(
        self,
        current_message: str,
        search_results: List[Dict[str, Any]],
        session: ChatSession
    ) -> str:
        """Prepare context for LLM including RAG results and conversation history."""
        # Start with system message
        context_parts = [
            "You are a helpful AI assistant with access to document knowledge.",
            "Use the provided document context to answer questions accurately.",
            "If you cannot find relevant information in the documents, say so clearly.",
            ""
        ]
        
        # Add document context if available
        if search_results:
            context_parts.append("Relevant document excerpts:")
            for i, result in enumerate(search_results, 1):
                context_parts.append(f"{i}. From '{result['filename']}':")
                context_parts.append(f"   {result['text']}")
                context_parts.append("")
        
        # Add recent conversation history (last 4 messages)
        recent_messages = session.messages[-4:] if len(session.messages) > 4 else session.messages
        if recent_messages:
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                role = "User" if msg["type"] == "user" else "Assistant"
                context_parts.append(f"{role}: {msg['content']}")
            context_parts.append("")
        
        # Add current question
        context_parts.append(f"Current question: {current_message}")
        context_parts.append("")
        context_parts.append("Please provide a helpful and accurate response:")
        
        return "\n".join(context_parts)

    async def get_chat_history(
        self,
        session_id: str,
        user: User,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get chat history for a session."""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )
            
            if session.user_id != user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to chat session"
                )
            
            # Return recent messages
            messages = session.messages[-limit:] if len(session.messages) > limit else session.messages
            
            return {
                "session_id": session_id,
                "messages": messages,
                "total_messages": len(session.messages),
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "settings": session.settings
            }
            
        except Exception as e:
            logger.error(f"Failed to get chat history: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get chat history: {str(e)}"
            )

    async def delete_chat_session(
        self,
        session_id: str,
        user: User,
        db: Session
    ) -> Dict[str, Any]:
        """Delete a chat session."""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )
            
            if session.user_id != user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to chat session"
                )
            
            # Remove session
            del self.active_sessions[session_id]
            
            # Log session deletion
            await log_user_action(
                user_id=user.id,
                action="chat_session_deleted",
                resource_id=session_id,
                details={"message_count": len(session.messages)},
                db=db
            )
            
            return {
                "session_id": session_id,
                "deleted": True,
                "message_count": len(session.messages)
            }
            
        except Exception as e:
            logger.error(f"Failed to delete chat session: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete chat session: {str(e)}"
            )

    async def list_chat_sessions(self, user: User) -> Dict[str, Any]:
        """List active chat sessions for a user."""
        try:
            user_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if session.user_id == user.id:
                    user_sessions.append({
                        "session_id": session_id,
                        "created_at": session.created_at.isoformat(),
                        "last_activity": session.last_activity.isoformat(),
                        "message_count": len(session.messages),
                        "settings": session.settings
                    })
            
            # Sort by last activity
            user_sessions.sort(key=lambda x: x["last_activity"], reverse=True)
            
            return {
                "sessions": user_sessions,
                "total_sessions": len(user_sessions)
            }
            
        except Exception as e:
            logger.error(f"Failed to list chat sessions: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list chat sessions: {str(e)}"
            )

    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old inactive sessions."""
        try:
            current_time = datetime.utcnow()
            sessions_to_remove = []
            
            for session_id, session in self.active_sessions.items():
                age = (current_time - session.last_activity).total_seconds() / 3600
                if age > max_age_hours:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
            
            logger.info(f"Cleaned up {len(sessions_to_remove)} old chat sessions")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {str(e)}")

# Global instance
chat_controller = ChatController()