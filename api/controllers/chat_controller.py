# api/controllers/chat_controller.py

from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import logging
import json
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from sqlalchemy import and_, or_, desc

from database.models import User, Document, DocumentChunk, ChatSessionModel, RegisteredModel
from llm.factory import create_model_manager_with_registered_models_sync
from vector_db.embedding_manager import EnhancedEmbeddingManager
from vector_db.embedding_model_registry import get_embedding_model_registry, EmbeddingProvider
from vector_db.search_engine import EnhancedSearchEngine
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
    
    def __init__(self, user_id: Optional[int] = None):
        self.model_manager = create_model_manager_with_registered_models_sync(user_id)
        self.embedding_registry = get_embedding_model_registry()
        self.search_engine = None  # Will be initialized per request
        self.active_sessions: Dict[str, ChatSession] = {}
    
    async def _save_session_to_db(self, session: ChatSession, db: Session):
        """Save session to database for persistence."""
        try:
            # Check if session already exists in database
            db_session = db.query(ChatSessionModel).filter(
                ChatSessionModel.session_id == session.session_id
            ).first()
            
            if db_session:
                # Update existing session
                db_session.set_settings(session.settings)
                db_session.update_activity()
                db_session.message_count = len(session.messages)
                logger.debug(f"Updated existing session {session.session_id} in database")
            else:
                # Create new session
                db_session = ChatSessionModel(
                    session_id=session.session_id,
                    user_id=session.user_id,
                    message_count=len(session.messages),
                    total_tokens_used=0
                )
                db_session.set_settings(session.settings)
                db.add(db_session)
                logger.debug(f"Created new session {session.session_id} in database")
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to save session to database: {str(e)}")
            db.rollback()
    
    async def _load_session_from_db(self, session_id: str, db: Session) -> Optional[ChatSession]:
        """Load session from database and restore to memory."""
        try:
            db_session = db.query(ChatSessionModel).filter(
                and_(
                    ChatSessionModel.session_id == session_id,
                    ChatSessionModel.is_active == True
                )
            ).first()
            
            if not db_session:
                logger.debug(f"Session {session_id} not found in database")
                return None
            
            if not db_session.is_valid():
                logger.debug(f"Session {session_id} found but expired or invalid")
                return None
            
            # Restore session to memory
            session = ChatSession(
                session_id=db_session.session_id,
                user_id=db_session.user_id,
                settings=db_session.get_settings()
            )
            
            # Update timestamps based on database
            session.created_at = db_session.created_at.replace(tzinfo=None)
            session.last_activity = db_session.last_activity.replace(tzinfo=None)
            
            # Note: Messages are not persisted, so we start with empty message list
            # This is intentional - we keep session state but not full conversation history
            
            logger.info(f"Restored session {session_id} from database for user {db_session.user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session from database: {str(e)}")
            return None
    
    async def _try_recover_session(self, session_id: str, user: User, db: Session) -> Optional[ChatSession]:
        """Try to recover a session from database if not found in memory."""
        logger.info(f"[CHAT] Attempting to recover session {session_id} from database")
        
        try:
            # Load session from database
            session = await self._load_session_from_db(session_id, db)
            if not session:
                return None
            
            # Verify session belongs to user
            if session.user_id != user.id:
                logger.warning(f"Session {session_id} belongs to user {session.user_id}, not {user.id}")
                return None
            
            # Add to active sessions
            self.active_sessions[session_id] = session
            
            logger.info(f"[CHAT] Successfully recovered session {session_id} for user {user.id}")
            return session
            
        except Exception as e:
            logger.error(f"[CHAT] Failed to recover session {session_id}: {str(e)}")
            return None
    
    async def _cleanup_expired_sessions_db(self, db: Session, max_age_hours: int = 24):
        """Clean up expired sessions from database."""
        try:
            from datetime import timedelta
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            # Find expired sessions
            expired_sessions = db.query(ChatSessionModel).filter(
                or_(
                    ChatSessionModel.last_activity < cutoff_time,
                    ChatSessionModel.expires_at < datetime.utcnow(),
                    ChatSessionModel.is_active == False
                )
            ).all()
            
            # Deactivate expired sessions
            for session in expired_sessions:
                session.deactivate()
                
                # Remove from memory if present
                if session.session_id in self.active_sessions:
                    del self.active_sessions[session.session_id]
            
            db.commit()
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions from database")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            db.rollback()
        
    async def get_available_models(self, user: User, db: Session = None) -> Dict[str, Any]:
        """Get available LLM and embedding models."""
        try:
            # Get registered LLM models directly from PostgreSQL database
            llm_models = []
            
            if db is not None:
                try:
                    # Query registered models for this user and public models
                    query = db.query(RegisteredModel).filter(
                        and_(
                            or_(
                                RegisteredModel.user_id == user.id,  # User's own models
                                RegisteredModel.is_public == True    # Public models
                            ),
                            RegisteredModel.is_active == True,       # Only active models
                            RegisteredModel.deleted_at.is_(None),   # Not soft-deleted
                            RegisteredModel.supports_embeddings == False  # LLM models only
                        )
                    ).order_by(RegisteredModel.fallback_priority.asc().nullslast()).all()
                    
                    logger.info(f"Found {len(query)} registered LLM models in database for user {user.id}")
                    
                    for model in query:
                        try:
                            # Parse model configuration from JSON
                            import json
                            config_data = json.loads(model.config_json) if model.config_json else {}
                            
                            llm_models.append({
                                "model_id": f"registered_{model.id}",
                                "model_name": model.model_name,
                                "display_name": model.display_name or model.name,
                                "provider": model.provider.value,
                                "description": model.description or f"Registered {model.provider.value} model",
                                "max_tokens": model.max_tokens or config_data.get("max_tokens", 2048),
                                "context_window": model.context_window or config_data.get("context_window", 4096),
                                "supports_streaming": model.supports_streaming,
                                "supports_embeddings": model.supports_embeddings,
                                "capabilities": ["text_generation", "chat"],
                                "usage_count": model.usage_count,
                                "success_rate": model.success_rate,
                                "is_public": model.is_public,
                                "owner_id": model.user_id
                            })
                            
                        except Exception as model_error:
                            logger.warning(f"Failed to process registered model {model.id}: {str(model_error)}")
                            continue
                            
                except Exception as db_error:
                    logger.error(f"Failed to query registered models from database: {str(db_error)}")
                    
            # Fallback: try model manager if database query failed or no db session
            if not llm_models:
                logger.info("No models from database, trying model manager as fallback")
                registered_model_ids = self.model_manager.list_registered_models()
                
                for model_id in registered_model_ids:
                    config = self.model_manager.get_model_config(model_id)
                    if config:
                        llm_models.append({
                            "model_id": model_id,
                            "model_name": config.model_name,
                            "display_name": config.model_name,
                            "provider": self._get_provider_from_model_id(model_id),
                            "description": f"Max tokens: {config.max_tokens}, Temperature: {config.temperature}",
                            "max_tokens": config.max_tokens,
                            "context_window": config.context_window,
                            "supports_streaming": True,
                            "supports_embeddings": False,
                            "capabilities": ["text_generation", "chat"]
                        })
            
            # Get registered embedding models
            embedding_models = []
            available_embedding_models = self.embedding_registry.list_models()
            
            for model in available_embedding_models:
                embedding_models.append({
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "display_name": model.display_name,
                    "provider": model.provider.value,
                    "description": model.description,
                    "embedding_dimension": model.embedding_dimension,
                    "max_tokens": model.max_input_length,
                    "context_window": model.max_input_length,
                    "supports_streaming": False,
                    "supports_embeddings": True,
                    "capabilities": ["embeddings", "semantic_search"],
                    "performance_tier": model.performance_tier,
                    "quality_score": model.quality_score,
                    "use_cases": model.use_cases,
                    "language_support": model.language_support,
                    "model_size_mb": model.model_size_mb,
                    "memory_requirements_mb": model.memory_requirements_mb,
                    "gpu_required": model.gpu_required,
                    "api_cost_per_1k_tokens": model.api_cost_per_1k_tokens
                })
            
            logger.info(f"Returning {len(llm_models)} LLM models and {len(embedding_models)} embedding models")
            
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

    async def _ensure_model_registered(self, model_id: str, user: User, db: Session) -> bool:
        """Ensure a database model is registered in the model manager for execution."""
        try:
            # If it's not a database model, assume it's already registered
            if not model_id.startswith("registered_"):
                return model_id in self.model_manager.list_registered_models()
            
            # If already registered in model manager, we're good
            if model_id in self.model_manager.list_registered_models():
                logger.debug(f"Model {model_id} already registered in model manager")
                return True
            
            # Extract database ID from model_id
            try:
                db_id = int(model_id.replace("registered_", ""))
            except ValueError:
                logger.error(f"Invalid registered model ID format: {model_id}")
                return False
            
            # Query database for model details
            db_model = db.query(RegisteredModel).filter(
                and_(
                    RegisteredModel.id == db_id,
                    RegisteredModel.is_active == True,
                    RegisteredModel.deleted_at.is_(None),
                    or_(
                        RegisteredModel.user_id == user.id,
                        RegisteredModel.is_public == True
                    )
                )
            ).first()
            
            if not db_model:
                logger.error(f"Database model {model_id} not found or not accessible")
                return False
            
            # Parse model configuration
            import json
            try:
                config_data = json.loads(db_model.config_json) if db_model.config_json else {}
                provider_config = json.loads(db_model.provider_config_json) if db_model.provider_config_json else {}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse model configuration for {model_id}: {e}")
                return False
            
            # Create ModelConfig for registration
            from llm.base.models import ModelConfig
            model_config = ModelConfig(
                model_name=db_model.model_name,
                max_tokens=db_model.max_tokens or config_data.get("max_tokens", 2048),
                temperature=config_data.get("temperature", 0.7),
                top_p=config_data.get("top_p", 1.0),
                presence_penalty=config_data.get("presence_penalty", 0.0),
                frequency_penalty=config_data.get("frequency_penalty", 0.0),
                context_window=db_model.context_window or config_data.get("context_window", 4096),
                api_base=config_data.get("api_base"),
                api_key=provider_config.get("api_key"),
                stop_sequences=config_data.get("stop_sequences"),
                repeat_penalty=config_data.get("repeat_penalty"),
                top_k=config_data.get("top_k"),
                provider_config=provider_config
            )
            
            # Register model in model manager
            provider_name = db_model.provider.value.lower()
            self.model_manager.register_model(
                model_id=model_id,
                provider_name=provider_name,
                config=model_config,
                provider_kwargs=provider_config,
                fallback_priority=db_model.fallback_priority
            )
            
            logger.info(f"Successfully registered database model {model_id} ({provider_name}) in model manager")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register database model {model_id}: {str(e)}")
            return False

    async def _get_available_llm_model_ids(self, user: User, db: Session) -> List[str]:
        """Get list of available LLM model IDs for validation."""
        try:
            # Get from database first
            if db is not None:
                query = db.query(RegisteredModel).filter(
                    and_(
                        or_(
                            RegisteredModel.user_id == user.id,  # User's own models
                            RegisteredModel.is_public == True    # Public models
                        ),
                        RegisteredModel.is_active == True,       # Only active models
                        RegisteredModel.deleted_at.is_(None),   # Not soft-deleted
                        RegisteredModel.supports_embeddings == False  # LLM models only
                    )
                ).all()
                
                model_ids = [f"registered_{model.id}" for model in query]
                if model_ids:
                    logger.debug(f"Found {len(model_ids)} LLM models from database: {model_ids}")
                    return model_ids
            
            # Fallback to model manager
            manager_models = self.model_manager.list_registered_models()
            if manager_models:
                logger.debug(f"Found {len(manager_models)} LLM models from model manager: {manager_models}")
                return manager_models
                
            # Final fallback - provide default model
            logger.warning("No LLM models found anywhere, providing default")
            return ["default-llm"]
            
        except Exception as e:
            logger.error(f"Error getting available LLM models: {str(e)}")
            return ["default-llm"]

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
        elif model_id.startswith("registered_"):
            return "registered"
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
            
            # Save session to database for persistence
            await self._save_session_to_db(session, db)
            
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
        # Get available LLM models first
        available_llm_models = await self._get_available_llm_model_ids(user, db)
        
        # Default settings - use first available model as default
        default_llm_model = available_llm_models[0] if available_llm_models else "default-llm"
        
        default_settings = {
            "llm_model": default_llm_model,
            "embedding_model": "hf-minilm-l6-v2",
            "temperature": 0.7,
            "max_tokens": 2048,
            "use_rag": True,
            "search_type": "hybrid",
            "top_k_documents": 5,
            "enable_fallback": True,  # Enable text search fallback by default
            "fallback_threshold": 1   # Minimum number of results before fallback kicks in
        }
        
        # Merge with provided settings
        validated_settings = {**default_settings, **settings}
        
        # Validate LLM model
        if validated_settings["llm_model"] not in available_llm_models:
            # Fallback to first available model
            if available_llm_models:
                validated_settings["llm_model"] = available_llm_models[0]
                logger.info(f"LLM model '{settings.get('llm_model', 'None')}' not found, using '{available_llm_models[0]}'")
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
        
        # Validate search type - match Enhanced Search Engine supported types
        if validated_settings["search_type"] not in ["semantic", "keyword", "hybrid", "contextual"]:
            validated_settings["search_type"] = "hybrid"
        
        # Validate boolean settings
        validated_settings["enable_fallback"] = bool(validated_settings["enable_fallback"])
        validated_settings["fallback_threshold"] = max(0, int(validated_settings["fallback_threshold"]))
        
        return validated_settings

    async def process_chat_message(
        self,
        session_id: str,
        message: str,
        user: User,
        db: Session
    ) -> AsyncGenerator[str, None]:
        """Process a chat message and yield streaming responses."""
        logger.info(f"[CHAT] Starting message processing - Session: {session_id}, User: {user.id}, Message length: {len(message)}")
        
        try:
            # Get session (try memory first, then database recovery)
            logger.debug(f"[CHAT] Retrieving session {session_id}")
            session = self.active_sessions.get(session_id)
            if not session:
                logger.info(f"[CHAT] Session {session_id} not found in memory, attempting recovery")
                session = await self._try_recover_session(session_id, user, db)
                
                if not session:
                    logger.error(f"[CHAT] Session {session_id} not found in memory or database")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Chat session not found"
                    )
            
            logger.debug(f"[CHAT] Session found - User: {session.user_id}, Settings: {session.settings}")
            
            # Verify session ownership
            if session.user_id != user.id:
                logger.error(f"[CHAT] Access denied - Session user {session.user_id} != Current user {user.id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to chat session"
                )
            
            # Update session activity
            session.last_activity = datetime.utcnow()
            logger.debug(f"[CHAT] Session activity updated")
            
            # Add user message to session
            user_message = {
                "id": str(uuid.uuid4()),
                "type": "user",
                "content": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            session.messages.append(user_message)
            logger.info(f"[CHAT] User message added - Total messages: {len(session.messages)}")
            
            # Initialize search engine if using RAG
            search_results = []
            if session.settings["use_rag"]:
                logger.info(f"[CHAT] Starting RAG search - Embedding model: {session.settings['embedding_model']}")
                try:
                    search_results = await self._perform_rag_search(
                        message, session.settings, user, db
                    )
                    logger.info(f"[CHAT] RAG search completed - Found {len(search_results)} results")
                except Exception as rag_error:
                    logger.error(f"[CHAT] RAG search failed: {str(rag_error)}", exc_info=True)
                    # Continue without RAG if search fails
                    search_results = []
                
                # Send sources information
                if search_results:
                    logger.debug(f"[CHAT] Sending sources data for {len(search_results)} results")
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
                    logger.debug(f"[CHAT] Sources data sent successfully")
            else:
                logger.info(f"[CHAT] RAG disabled for this session")
            
            # Prepare context for LLM
            logger.info(f"[CHAT] Preparing LLM context - RAG results: {len(search_results)}")
            context = self._prepare_llm_context(message, search_results, session)
            context_length = len(context)
            logger.info(f"[CHAT] Context prepared - Length: {context_length} chars")
            
            # Generate response using selected LLM
            llm_model = session.settings["llm_model"]
            logger.info(f"[CHAT] Starting LLM generation - Model: {llm_model}, Stream: True")
            
            # Ensure model is registered in model manager if it's a database model
            if llm_model.startswith("registered_"):
                logger.info(f"[CHAT] Ensuring database model {llm_model} is registered in model manager")
                model_registered = await self._ensure_model_registered(llm_model, user, db)
                if not model_registered:
                    logger.error(f"[CHAT] Failed to register database model {llm_model}")
                    error_data = {
                        "type": "error",
                        "error": f"Model {llm_model} could not be loaded. Please try a different model.",
                        "error_code": "model_registration_failed"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                else:
                    logger.info(f"[CHAT] Database model {llm_model} successfully registered")
            
            response_content = ""
            chunk_count = 0
            
            try:
                # Get the async generator by awaiting the coroutine
                logger.info(f"[CHAT] Getting streaming generator from model manager")
                generator = None
                
                try:
                    generator = await self.model_manager.generate(
                        prompt=context,
                        model_id=llm_model,
                        stream=True
                    )
                    logger.info(f"[CHAT] Successfully obtained generator from model manager")
                    
                except Exception as model_error:
                    logger.error(f"[CHAT] Failed to get generator from model manager: {str(model_error)}", exc_info=True)
                    
                    # Use the specific error message from the model manager (no fallback)
                    error_message = str(model_error)
                    error_code = "model_specific_error"
                    
                    # Determine error code based on error type
                    error_lower = error_message.lower()
                    if "not registered" in error_lower:
                        error_code = "model_not_registered"
                    elif "authentication" in error_lower or "api key" in error_lower:
                        error_code = "authentication_failed"
                    elif "rate limit" in error_lower or "quota" in error_lower:
                        error_code = "rate_limit_exceeded"
                    elif "not found" in error_lower:
                        error_code = "model_not_found"
                    elif "connection" in error_lower or "timeout" in error_lower:
                        error_code = "connection_failed"
                    elif "unavailable" in error_lower:
                        error_code = "model_unavailable"
                    elif "configuration" in error_lower or "setup" in error_lower:
                        error_code = "configuration_error"
                    
                    # Send detailed error response
                    error_data = {
                        "type": "error",
                        "error": error_message,
                        "error_code": error_code,
                        "model": llm_model,
                        "timestamp": datetime.utcnow().isoformat(),
                        "suggestions": self._get_model_error_suggestions(error_code, llm_model)
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                
                # Verify we got a valid async generator
                if generator is None:
                    logger.error(f"[CHAT] Generator is None from model manager")
                    error_data = {
                        "type": "error",
                        "error": "Failed to initialize LLM response generator",
                        "error_code": "generator_none"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                
                if not hasattr(generator, '__aiter__'):
                    logger.error(f"[CHAT] Expected async generator, got {type(generator)}")
                    error_data = {
                        "type": "error",
                        "error": "Invalid response format from LLM service",
                        "error_code": "invalid_generator_type",
                        "received_type": str(type(generator))
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                
                # Now iterate over the async generator
                logger.info(f"[CHAT] Starting to iterate over streaming generator")
                
                try:
                    async for chunk in generator:
                        chunk_count += 1
                        if isinstance(chunk, str):
                            response_content += chunk
                            chunk_data = {
                                "type": "content",
                                "content": chunk
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                            
                            # Log every 10th chunk to avoid spam
                            if chunk_count % 10 == 0:
                                logger.debug(f"[CHAT] Streamed {chunk_count} chunks, total length: {len(response_content)}")
                        else:
                            logger.warning(f"[CHAT] Received non-string chunk: {type(chunk)} - {chunk}")
                            # Try to convert to string if possible
                            try:
                                chunk_str = str(chunk)
                                if chunk_str and chunk_str != "None":
                                    response_content += chunk_str
                                    chunk_data = {
                                        "type": "content",
                                        "content": chunk_str
                                    }
                                    yield f"data: {json.dumps(chunk_data)}\n\n"
                            except Exception as convert_error:
                                logger.warning(f"[CHAT] Failed to convert chunk to string: {convert_error}")
                    
                    logger.info(f"[CHAT] LLM generation completed - Total chunks: {chunk_count}, Response length: {len(response_content)}")
                    
                except asyncio.CancelledError:
                    logger.info(f"[CHAT] LLM generation was cancelled by client")
                    error_data = {
                        "type": "error",
                        "error": "Response generation was cancelled",
                        "error_code": "generation_cancelled"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                    
                except Exception as stream_error:
                    logger.error(f"[CHAT] Error during streaming: {str(stream_error)}", exc_info=True)
                    
                    # Determine error type for better messages
                    error_message = "Streaming error occurred"
                    if "connection" in str(stream_error).lower():
                        error_message = "Connection lost during response generation"
                    elif "timeout" in str(stream_error).lower():
                        error_message = "Response generation timed out"
                    elif "rate limit" in str(stream_error).lower():
                        error_message = "Rate limit exceeded during generation"  
                    else:
                        error_message = f"Streaming error: {str(stream_error)}"
                    
                    error_data = {
                        "type": "error",
                        "error": error_message,
                        "error_code": "streaming_error",
                        "partial_response": response_content if response_content else None
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                
            except Exception as llm_error:
                logger.error(f"[CHAT] LLM generation failed: {str(llm_error)}", exc_info=True)
                
                # Determine appropriate error message and code based on error type
                error_str = str(llm_error)
                if "geographic location" in error_str or "location is not supported" in error_str:
                    error_message = "LLM service is not available in your geographic location. Please configure an alternative provider."
                    error_code = "geographic_restriction"
                elif "API key" in error_str:
                    error_message = "LLM API authentication failed. Please check your API key configuration."
                    error_code = "auth_error"
                elif "quota" in error_str.lower() or "rate limit" in error_str.lower():
                    error_message = "LLM service quota exceeded. Please try again later."
                    error_code = "quota_exceeded"
                elif "connection" in error_str.lower() or "network" in error_str.lower():
                    error_message = "Unable to connect to LLM service. Please check your internet connection."
                    error_code = "connection_error"
                elif "timeout" in error_str.lower():
                    error_message = "LLM service request timed out. Please try again."
                    error_code = "timeout_error"
                else:
                    error_message = f"LLM generation failed: {error_str}"
                    error_code = "llm_error"
                
                error_data = {
                    "type": "error",
                    "error": error_message,
                    "error_code": error_code,
                    "suggestions": self._get_error_suggestions(error_code)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return
            
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
            logger.info(f"[CHAT] Assistant message added - ID: {assistant_message['id']}, Total messages: {len(session.messages)}")
            
            # Save updated session to database
            try:
                await self._save_session_to_db(session, db)
                logger.debug(f"[CHAT] Session {session_id} saved to database after message")
            except Exception as save_error:
                logger.error(f"[CHAT] Failed to save session to database: {str(save_error)}")
            
            # Send completion signal
            completion_data = {
                "type": "done",
                "message_id": assistant_message["id"]
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            logger.info(f"[CHAT] Completion signal sent successfully")
            
            # Log the interaction
            try:
                await log_user_action(
                    user_id=user.id,
                    action="chat_message_processed",
                    resource_id=session_id,
                    details={
                        "message_length": len(message),
                        "response_length": len(response_content),
                        "rag_used": session.settings["use_rag"],
                        "sources_count": len(search_results),
                        "model_used": session.settings["llm_model"],
                        "chunks_received": chunk_count
                    },
                    db=db
                )
                logger.debug(f"[CHAT] User action logged successfully")
            except Exception as log_error:
                logger.error(f"[CHAT] Failed to log user action: {str(log_error)}")
            
            logger.info(f"[CHAT] Message processing completed successfully - Session: {session_id}")
            
        except Exception as e:
            logger.error(f"[CHAT] Failed to process chat message: {str(e)}", exc_info=True)
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
        """Perform RAG search to find relevant documents with fallback and comprehensive logging."""
        import time
        start_time = time.time()
        
        logger.info(f"[RAG_SEARCH] Starting RAG search - User: {user.id}, Query: '{query[:100]}...', Search type: {settings.get('search_type', 'hybrid')}, Top-k: {settings.get('top_k_documents', 5)}")
        
        # Get search type for enhanced search engine
        search_type = settings.get("search_type", "hybrid")
        logger.info(f"[RAG_SEARCH] Using search type: {search_type}")
        
        # If hybrid search is requested, use the dedicated hybrid method
        if search_type == "hybrid":
            return await self._perform_hybrid_rag_search(query, settings, user, db)
        
        try:
            # Create embedding manager for the selected model
            embedding_model = self.embedding_registry.get_model(settings["embedding_model"])
            if not embedding_model:
                logger.error(f"[RAG_SEARCH] Embedding model {settings['embedding_model']} not found - available models: {[m.model_id for m in self.embedding_registry.list_models()]}")
                return await self._fallback_to_text_search(query, user, db, settings)
            
            logger.info(f"[RAG_SEARCH] Using embedding model: {embedding_model.model_name} (provider: {embedding_model.provider.value})")
            
            # Create appropriate embedding manager
            embedding_manager = None
            try:
                if embedding_model.provider == EmbeddingProvider.HUGGINGFACE:
                    logger.debug(f"[RAG_SEARCH] Creating HuggingFace embedding manager")
                    embedding_manager = EnhancedEmbeddingManager.create_huggingface_manager(
                        embedding_model.model_name
                    )
                elif embedding_model.provider == EmbeddingProvider.OLLAMA:
                    logger.debug(f"[RAG_SEARCH] Creating Ollama embedding manager")
                    embedding_manager = EnhancedEmbeddingManager.create_ollama_manager(
                        model_name=embedding_model.model_name
                    )
                elif embedding_model.provider == EmbeddingProvider.OPENAI:
                    logger.debug(f"[RAG_SEARCH] Creating OpenAI embedding manager")
                    from config import get_settings
                    settings_config = get_settings()
                    if not settings_config.OPENAI_API_KEY:
                        logger.error(f"[RAG_SEARCH] OpenAI API key not configured")
                        return await self._fallback_to_text_search(query, user, db, settings)
                    embedding_manager = EnhancedEmbeddingManager.create_openai_manager(
                        api_key=settings_config.OPENAI_API_KEY,
                        model_name=embedding_model.model_name
                    )
                else:
                    logger.error(f"[RAG_SEARCH] Unsupported embedding provider: {embedding_model.provider}")
                    return await self._fallback_to_text_search(query, user, db, settings)
            except Exception as e:
                logger.error(f"[RAG_SEARCH] Failed to create embedding manager: {str(e)}")
                return await self._fallback_to_text_search(query, user, db, settings)
            
            # Initialize search engine
            try:
                if not self.search_engine:
                    logger.debug(f"[RAG_SEARCH] Initializing search engine")
                    from vector_db.storage_manager import VectorStorageManager
                    storage_manager = VectorStorageManager()
                    self.search_engine = EnhancedSearchEngine(storage_manager, embedding_manager)
                    logger.info(f"[RAG_SEARCH] Search engine initialized successfully")
            except Exception as e:
                logger.error(f"[RAG_SEARCH] Failed to initialize search engine: {str(e)}")
                return await self._fallback_to_text_search(query, user, db, settings)
            
            # Perform search
            logger.info(f"[RAG_SEARCH] Starting vector search with search engine")
            vector_search_start = time.time()
            
            try:
                search_results = await self.search_engine.search_with_context(
                    query=query,
                    search_type=settings["search_type"],
                    user_id=user.id,
                    top_k=settings["top_k_documents"] * 2,  # Get more results for filtering
                    db=db
                )
                
                vector_search_time = (time.time() - vector_search_start) * 1000
                logger.info(f"[RAG_SEARCH] Vector search completed in {vector_search_time:.2f}ms, found {len(search_results)} results")
                
                # Log search result details
                if search_results:
                    avg_score = sum(r.get('similarity_score', 0) for r in search_results) / len(search_results)
                    max_score = max(r.get('similarity_score', 0) for r in search_results)
                    min_score = min(r.get('similarity_score', 0) for r in search_results)
                    logger.debug(f"[RAG_SEARCH] Score stats - Avg: {avg_score:.4f}, Max: {max_score:.4f}, Min: {min_score:.4f}")
                else:
                    logger.warning(f"[RAG_SEARCH] Vector search returned no results for query: '{query}'")
                    
            except Exception as e:
                logger.error(f"[RAG_SEARCH] Vector search failed: {str(e)}", exc_info=True)
                return await self._fallback_to_text_search(query, user, db, settings)
            
            # Format results with detailed logging
            logger.debug(f"[RAG_SEARCH] Formatting {len(search_results)} search results")
            formatted_results = []
            
            for i, result in enumerate(search_results):
                # Get document info
                document_id = result.get("document_id")
                logger.debug(f"[RAG_SEARCH] Processing result {i+1}: document_id={document_id}, score={result.get('similarity_score', 0):.4f}")
                
                if not document_id:
                    logger.warning(f"[RAG_SEARCH] Result {i+1} missing document_id, skipping")
                    continue
                
                document = db.query(Document).filter(
                    and_(
                        Document.id == document_id,
                        Document.user_id == user.id,
                        Document.is_deleted == False
                    )
                ).first()
                
                if document:
                    chunk_text = result.get("text", "")
                    formatted_result = {
                        "document_id": document.id,
                        "filename": document.filename,
                        "chunk_id": result.get("chunk_id", ""),
                        "text": chunk_text,
                        "similarity_score": result.get("similarity_score", 0.0),
                        "metadata": result.get("metadata", {})
                    }
                    formatted_results.append(formatted_result)
                    
                    # Log chunk content for debugging
                    logger.debug(f"[RAG_SEARCH] Added result: {document.filename}, chunk_text_length={len(chunk_text)}, text_preview='{chunk_text[:100]}...'")
                else:
                    logger.warning(f"[RAG_SEARCH] Document {document_id} not found or not accessible for user {user.id}")
            
            final_results = formatted_results[:settings["top_k_documents"]]
            total_time = (time.time() - start_time) * 1000
            
            logger.info(f"[RAG_SEARCH] RAG search completed in {total_time:.2f}ms - Returning {len(final_results)} results from {len(formatted_results)} candidates")
            
            # Check if we should try fallback search
            should_fallback = (
                settings.get("enable_fallback", True) and 
                len(final_results) < settings.get("fallback_threshold", 1)
            )
            
            if should_fallback:
                logger.warning(f"[RAG_SEARCH] Found {len(final_results)} results (below threshold {settings.get('fallback_threshold', 1)}), trying fallback text search")
                return await self._fallback_to_text_search(query, user, db, settings)
            
            return final_results
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error(f"[RAG_SEARCH] RAG search failed after {total_time:.2f}ms: {str(e)}", exc_info=True)
            
            # Try fallback search as last resort if enabled
            if settings.get("enable_fallback", True):
                logger.info(f"[RAG_SEARCH] Attempting fallback text search due to error")
                try:
                    return await self._fallback_to_text_search(query, user, db, settings)
                except Exception as fallback_error:
                    logger.error(f"[RAG_SEARCH] Fallback search also failed: {str(fallback_error)}")
                    return []
            else:
                logger.info(f"[RAG_SEARCH] Fallback search is disabled, returning empty results")
                return []

    async def _fallback_to_text_search(
        self,
        query: str,
        user: User,
        db: Session,
        settings: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Fallback to text-based search when vector search fails or returns no results.
        Uses similar logic to the search controller's text search functionality.
        """
        import time
        start_time = time.time()
        
        logger.info(f"[RAG_FALLBACK] Starting text search fallback for user {user.id}, query: '{query[:100]}...'")
        
        try:
            from sqlalchemy import or_, and_
            
            # Build base query for accessible documents
            base_query = db.query(Document).filter(
                and_(
                    Document.is_deleted == False,
                    Document.status == 'completed',
                    or_(
                        Document.user_id == user.id,  # User's own documents
                        Document.is_public == True     # Public documents
                    )
                )
            )
            
            logger.debug(f"[RAG_FALLBACK] Built base query for accessible documents")
            
            # Apply text search if query provided
            if query and query.strip():
                query_cleaned = query.strip()
                logger.info(f"[RAG_FALLBACK] Applying text search for query: '{query_cleaned}'")
                
                # Create text search conditions
                search_conditions = []
                
                # Search in extracted_text (main content) - case insensitive
                if query_cleaned:
                    search_conditions.append(
                        Document.extracted_text.ilike(f"%{query_cleaned}%")
                    )
                    # Search in title
                    search_conditions.append(
                        Document.title.ilike(f"%{query_cleaned}%")
                    )
                    # Search in filename
                    search_conditions.append(
                        Document.filename.ilike(f"%{query_cleaned}%")
                    )
                    # Search in description
                    search_conditions.append(
                        Document.description.ilike(f"%{query_cleaned}%")
                    )
                
                if search_conditions:
                    base_query = base_query.filter(or_(*search_conditions))
                    logger.debug(f"[RAG_FALLBACK] Added {len(search_conditions)} text search conditions")
            
            # Execute query and get results
            documents = base_query.limit(settings.get("top_k_documents", 5) * 2).all()
            logger.info(f"[RAG_FALLBACK] Found {len(documents)} matching documents")
            
            # Convert to search result format with relevance scoring
            search_results = []
            for doc in documents:
                # Calculate relevance score based on text matches
                score = self._calculate_text_relevance_score(doc, query)
                
                # Extract content snippet with highlighting context
                content_snippet = self._extract_content_snippet(doc, query)
                
                search_result = {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "chunk_id": f"fallback_{doc.id}",  # Use fallback identifier
                    "text": content_snippet,
                    "similarity_score": score,
                    "metadata": {
                        "title": doc.title,
                        "content_type": doc.content_type,
                        "file_size": doc.file_size,
                        "created_at": doc.created_at.isoformat() if doc.created_at else None,
                        "language": doc.language,
                        "description": doc.description,
                        "search_type": "text_fallback"
                    }
                }
                search_results.append(search_result)
                
                logger.debug(f"[RAG_FALLBACK] Added result: {doc.filename}, score={score:.4f}, snippet_length={len(content_snippet)}")
            
            # Sort results by relevance score (highest first)
            search_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Return top results
            final_results = search_results[:settings.get("top_k_documents", 5)]
            total_time = (time.time() - start_time) * 1000
            
            logger.info(f"[RAG_FALLBACK] Text search fallback completed in {total_time:.2f}ms - Returning {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error(f"[RAG_FALLBACK] Text search fallback failed after {total_time:.2f}ms: {str(e)}", exc_info=True)
            return []

    def _calculate_text_relevance_score(self, document: Document, query: str) -> float:
        """Calculate relevance score for a document based on text matches."""
        if not query or not query.strip():
            return 0.5  # Default score for no query
        
        query_lower = query.lower().strip()
        score = 0.0
        
        try:
            # Title matches (highest weight)
            if document.title and query_lower in document.title.lower():
                score += 0.4
                # Exact title match gets bonus
                if query_lower == document.title.lower():
                    score += 0.2
            
            # Filename matches (high weight)
            if document.filename and query_lower in document.filename.lower():
                score += 0.3
            
            # Description matches (medium weight)
            if document.description and query_lower in document.description.lower():
                score += 0.2
            
            # Content matches (lower weight but can accumulate)
            if document.extracted_text:
                content_lower = document.extracted_text.lower()
                # Count occurrences in content
                match_count = content_lower.count(query_lower)
                if match_count > 0:
                    # Diminishing returns for multiple matches
                    content_score = min(0.3, match_count * 0.05)
                    score += content_score
            
            # Ensure score is between 0 and 1
            score = min(1.0, max(0.1, score))
            
        except Exception as e:
            logger.warning(f"[RAG_FALLBACK] Error calculating relevance score for document {document.id}: {e}")
            score = 0.1  # Minimal score for errors
        
        return score

    def _extract_content_snippet(self, document: Document, query: str, max_length: int = 300) -> str:
        """Extract a relevant content snippet with context around query matches."""
        if not document.extracted_text:
            # Fallback to title or description
            if document.title:
                return document.title[:max_length]
            elif document.description:
                return document.description[:max_length]
            else:
                return f"Document: {document.filename}"
        
        content = document.extracted_text
        
        # If no query, return beginning of content
        if not query or not query.strip():
            return content[:max_length] + ("..." if len(content) > max_length else "")
        
        query_lower = query.lower().strip()
        content_lower = content.lower()
        
        # Find the first occurrence of the query
        match_pos = content_lower.find(query_lower)
        
        if match_pos == -1:
            # No match found, return beginning
            return content[:max_length] + ("..." if len(content) > max_length else "")
        
        # Extract context around the match
        context_start = max(0, match_pos - 100)  # 100 chars before
        context_end = min(len(content), match_pos + len(query) + 200)  # 200 chars after
        
        snippet = content[context_start:context_end]
        
        # Add ellipsis if we're not at the beginning/end
        if context_start > 0:
            snippet = "..." + snippet
        if context_end < len(content):
            snippet = snippet + "..."
        
        # Truncate if still too long
        if len(snippet) > max_length:
            snippet = snippet[:max_length-3] + "..."
        
        return snippet

    async def _perform_hybrid_rag_search(
        self,
        query: str,
        settings: Dict[str, Any],
        user: User,
        db: Session
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid RAG search combining vector search and text search.
        This provides the best of both worlds - semantic understanding and exact keyword matching.
        """
        import time
        start_time = time.time()
        
        logger.info(f"[RAG_HYBRID] Starting hybrid RAG search - User: {user.id}, Query: '{query[:100]}...'")
        
        try:
            # Run both vector search and text search concurrently
            vector_task = asyncio.create_task(self._get_vector_search_results(query, settings, user, db))
            text_task = asyncio.create_task(self._fallback_to_text_search(query, user, db, settings))
            
            # Wait for both searches to complete
            vector_results, text_results = await asyncio.gather(vector_task, text_task, return_exceptions=True)
            
            # Handle exceptions from either search
            if isinstance(vector_results, Exception):
                logger.warning(f"[RAG_HYBRID] Vector search failed: {str(vector_results)}")
                vector_results = []
            if isinstance(text_results, Exception):
                logger.warning(f"[RAG_HYBRID] Text search failed: {str(text_results)}")
                text_results = []
            
            logger.info(f"[RAG_HYBRID] Got {len(vector_results)} vector results and {len(text_results)} text results")
            
            # Combine and re-rank results
            combined_results = self._combine_hybrid_results(vector_results, text_results, query)
            
            # Limit to requested number of results
            final_results = combined_results[:settings.get("top_k_documents", 5)]
            total_time = (time.time() - start_time) * 1000
            
            logger.info(f"[RAG_HYBRID] Hybrid search completed in {total_time:.2f}ms - Returning {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error(f"[RAG_HYBRID] Hybrid search failed after {total_time:.2f}ms: {str(e)}", exc_info=True)
            # Fall back to text search only
            return await self._fallback_to_text_search(query, user, db, settings)

    async def _get_vector_search_results(
        self,
        query: str,
        settings: Dict[str, Any],
        user: User,
        db: Session
    ) -> List[Dict[str, Any]]:
        """Get vector search results (extracted from main search method for hybrid use)."""
        try:
            # Create embedding manager for the selected model
            embedding_model = self.embedding_registry.get_model(settings["embedding_model"])
            if not embedding_model:
                logger.warning(f"[RAG_VECTOR] Embedding model {settings['embedding_model']} not found")
                return []
            
            # Create appropriate embedding manager
            embedding_manager = None
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
                    logger.warning("[RAG_VECTOR] OpenAI API key not configured")
                    return []
                embedding_manager = EnhancedEmbeddingManager.create_openai_manager(
                    api_key=settings_config.OPENAI_API_KEY,
                    model_name=embedding_model.model_name
                )
            else:
                logger.warning(f"[RAG_VECTOR] Unsupported embedding provider: {embedding_model.provider}")
                return []
            
            # Initialize search engine
            if not self.search_engine:
                from vector_db.storage_manager import VectorStorageManager
                storage_manager = VectorStorageManager()
                self.search_engine = EnhancedSearchEngine(storage_manager, embedding_manager)
            
            # Perform vector search
            search_results = await self.search_engine.search_with_context(
                query=query,
                search_type=settings.get("search_type", "semantic"),
                user_id=user.id,
                top_k=settings["top_k_documents"] * 2,
                db=db
            )
            
            # Format results
            formatted_results = []
            for result in search_results:
                document_id = result.get("document_id")
                if not document_id:
                    continue
                
                document = db.query(Document).filter(
                    and_(
                        Document.id == document_id,
                        Document.user_id == user.id,
                        Document.is_deleted == False
                    )
                ).first()
                
                if document:
                    formatted_result = {
                        "document_id": document.id,
                        "filename": document.filename,
                        "chunk_id": result.get("chunk_id", ""),
                        "text": result.get("text", ""),
                        "similarity_score": result.get("similarity_score", 0.0),
                        "metadata": result.get("metadata", {}),
                        "search_source": "vector"
                    }
                    formatted_results.append(formatted_result)
            
            logger.debug(f"[RAG_VECTOR] Vector search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"[RAG_VECTOR] Vector search failed: {str(e)}")
            return []

    def _combine_hybrid_results(
        self,
        vector_results: List[Dict[str, Any]],
        text_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and text search results with intelligent scoring.
        Prioritizes results that appear in both searches and handles unique results from each.
        """
        logger.debug(f"[RAG_HYBRID] Combining {len(vector_results)} vector results with {len(text_results)} text results")
        
        # Create a combined results dictionary keyed by document_id
        combined_scores = {}
        
        # Weights for different search types
        vector_weight = 0.6
        text_weight = 0.4
        overlap_bonus = 0.2  # Bonus for results that appear in both searches
        
        # Process vector results
        for result in vector_results:
            doc_id = result["document_id"]
            key = f"doc_{doc_id}"
            
            combined_scores[key] = {
                "result": result,
                "vector_score": result["similarity_score"],
                "text_score": 0.0,
                "final_score": result["similarity_score"] * vector_weight,
                "sources": ["vector"]
            }
        
        # Process text results
        for result in text_results:
            doc_id = result["document_id"]
            key = f"doc_{doc_id}"
            
            if key in combined_scores:
                # Document appears in both searches - update scores and add bonus
                combined_scores[key]["text_score"] = result["similarity_score"]
                combined_scores[key]["final_score"] = (
                    combined_scores[key]["vector_score"] * vector_weight +
                    result["similarity_score"] * text_weight +
                    overlap_bonus
                )
                combined_scores[key]["sources"].append("text")
                
                # Use the text result if it has more/better content
                if len(result.get("text", "")) > len(combined_scores[key]["result"].get("text", "")):
                    # Keep vector metadata but use text content
                    combined_scores[key]["result"]["text"] = result["text"]
                    combined_scores[key]["result"]["metadata"]["search_type"] = "hybrid"
                
                logger.debug(f"[RAG_HYBRID] Document {doc_id} found in both searches, final_score: {combined_scores[key]['final_score']:.4f}")
            else:
                # Document only in text search
                combined_scores[key] = {
                    "result": result,
                    "vector_score": 0.0,
                    "text_score": result["similarity_score"],
                    "final_score": result["similarity_score"] * text_weight,
                    "sources": ["text"]
                }
        
        # Convert back to list and sort by final score
        final_results = []
        for key, data in combined_scores.items():
            result = data["result"]
            result["similarity_score"] = data["final_score"]
            result["metadata"]["hybrid_info"] = {
                "vector_score": data["vector_score"],
                "text_score": data["text_score"],
                "sources": data["sources"]
            }
            final_results.append(result)
        
        # Sort by final score (highest first)
        final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        logger.info(f"[RAG_HYBRID] Combined results: {len(final_results)} total, {len([r for r in final_results if len(r['metadata']['hybrid_info']['sources']) > 1])} overlapping")
        
        return final_results

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
        limit: int = 50,
        db: Session = None
    ) -> Dict[str, Any]:
        """Get chat history for a session."""
        try:
            # Try to get session from memory or recover from database
            session = self.active_sessions.get(session_id)
            if not session and db:
                logger.info(f"Session {session_id} not found in memory, attempting recovery for history")
                session = await self._try_recover_session(session_id, user, db)
                
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
            # Try to get session from memory or recover from database
            session = self.active_sessions.get(session_id)
            if not session:
                logger.info(f"Session {session_id} not found in memory for deletion, attempting recovery")
                session = await self._try_recover_session(session_id, user, db)
                
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
            
            # Remove session from memory
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Deactivate session in database
            try:
                db_session = db.query(ChatSessionModel).filter(
                    ChatSessionModel.session_id == session_id
                ).first()
                if db_session:
                    db_session.deactivate()
                    db.commit()
                    logger.debug(f"Deactivated session {session_id} in database")
            except Exception as db_error:
                logger.error(f"Failed to deactivate session in database: {str(db_error)}")
                db.rollback()
            
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

    async def cleanup_old_sessions(self, max_age_hours: int = 24, db: Session = None):
        """Clean up old inactive sessions from both memory and database."""
        try:
            # Clean up memory sessions
            current_time = datetime.utcnow()
            memory_sessions_to_remove = []
            
            for session_id, session in self.active_sessions.items():
                age = (current_time - session.last_activity).total_seconds() / 3600
                if age > max_age_hours:
                    memory_sessions_to_remove.append(session_id)
            
            for session_id in memory_sessions_to_remove:
                del self.active_sessions[session_id]
            
            logger.info(f"Cleaned up {len(memory_sessions_to_remove)} old chat sessions from memory")
            
            # Clean up database sessions if db session is provided
            if db:
                await self._cleanup_expired_sessions_db(db, max_age_hours)
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {str(e)}")
    
    def _get_model_error_suggestions(self, error_code: str, model_id: str) -> List[str]:
        """Get helpful suggestions based on model error type."""
        suggestions = {
            "model_not_registered": [
                f"Model '{model_id}' is not registered in the system",
                "Check if the model was properly configured in the admin panel",
                "Contact an administrator to register this model"
            ],
            "authentication_failed": [
                f"Check the API key configuration for model '{model_id}'",
                "Verify the API key has the correct permissions",
                "Try regenerating the API key from the provider's dashboard"
            ],
            "rate_limit_exceeded": [
                f"Rate limit exceeded for model '{model_id}'",
                "Wait for the rate limit to reset (usually within an hour)",
                "Try using a different model or upgrade your API plan"
            ],
            "model_not_found": [
                f"Model '{model_id}' was not found on the provider",
                "Check if the model name is spelled correctly",
                "Verify the model is available in your region"
            ],
            "connection_failed": [
                f"Unable to connect to the provider for model '{model_id}'",
                "Check your internet connection",
                "The provider service may be temporarily unavailable"
            ],
            "model_unavailable": [
                f"Model '{model_id}' is currently marked as unavailable",
                "Try using a different model",
                "Contact an administrator to check the model status"
            ],
            "configuration_error": [
                f"Configuration issue with model '{model_id}'",
                "Check the model settings in the admin panel",
                "Verify all required parameters are correctly set"
            ]
        }
        
        return suggestions.get(error_code, [
            f"Error occurred with model '{model_id}'",
            "Try using a different model",
            "Contact support if the issue persists"
        ])

    def _get_error_suggestions(self, error_code: str) -> List[str]:
        """Get helpful suggestions based on error type."""
        suggestions = {
            "geographic_restriction": [
                "Configure an alternative LLM provider (OpenAI, Ollama)",
                "Use a VPN to access from a supported region",
                "Check the provider's supported regions documentation"
            ],
            "auth_error": [
                "Verify your API key is correct and active",
                "Check if your API key has the required permissions",
                "Try regenerating your API key from the provider's dashboard"
            ],
            "quota_exceeded": [
                "Wait for your quota to reset (usually at start of next billing period)",
                "Upgrade your API plan for higher limits",
                "Try using a different LLM provider as backup"
            ],
            "connection_error": [
                "Check your internet connection",
                "Verify the API endpoint is accessible",
                "Try again in a few moments"
            ],
            "timeout_error": [
                "Try with a shorter prompt",
                "Reduce the maximum response length",
                "Try again with the same request"
            ]
        }
        
        return suggestions.get(error_code, [
            "Try again in a few moments",
            "Check your configuration settings",
            "Contact support if the issue persists"
        ])

# Factory function for creating chat controllers with user context
def get_chat_controller(user_id: Optional[int] = None) -> ChatController:
    """Get a ChatController instance with models loaded for the specified user."""
    return ChatController(user_id)

# Backward compatibility - default instance with no user context
chat_controller = ChatController()