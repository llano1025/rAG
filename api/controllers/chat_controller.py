# api/controllers/chat_controller.py

from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import logging
import json
import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from sqlalchemy import and_, or_

from database.models import User, Document, ChatSessionModel, RegisteredModel
from llm.factory import create_model_manager_with_registered_models_sync
from vector_db.embedding_model_registry import get_embedding_model_registry, EmbeddingProvider
from vector_db.search_engine import EnhancedSearchEngine
from utils.security.audit_logger import log_user_action

logger = logging.getLogger(__name__)

class ChatSession:
    """Chat session management."""
    def __init__(self, session_id: str, user_id: int, settings: Dict[str, Any]):
        self.session_id = session_id
        self.user_id = user_id
        self.settings = settings
        self.messages: List[Dict[str, Any]] = []
        self.created_at = datetime.now(timezone.utc)
        self.last_activity = datetime.now(timezone.utc)

class ChatController:
    """Controller for chat functionality with RAG support."""
    
    def __init__(self, user_id: Optional[int] = None):
        self.model_manager = create_model_manager_with_registered_models_sync(user_id)
        self.embedding_registry = get_embedding_model_registry()
        self.search_engine = None  # Will be initialized as needed using shared infrastructure
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
        logger.info(f"Attempting to recover session {session_id} from database")
        
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
            
            logger.info(f"Successfully recovered session {session_id} for user {user.id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to recover session {session_id}: {str(e)}")
            return None
    
    async def _cleanup_expired_sessions_db(self, db: Session, max_age_hours: int = 24):
        """Clean up expired sessions from database."""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            
            # Find expired sessions
            expired_sessions = db.query(ChatSessionModel).filter(
                or_(
                    ChatSessionModel.last_activity < cutoff_time,
                    ChatSessionModel.expires_at < datetime.now(timezone.utc),
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
        logger.info(f"Validating settings for user {user.id}")
        logger.debug(f"Received settings: {settings}")

        # Get available LLM models first
        available_llm_models = await self._get_available_llm_model_ids(user, db)
        logger.debug(f"Available LLM models: {available_llm_models}")

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
            "fallback_threshold": 1,  # Minimum number of results before fallback kicks in
            # Reranker settings
            "enable_reranking": False,
            "reranker_model": None,  # Use default from config
            "rerank_score_weight": 0.5,
            "min_rerank_score": None
        }
        
        # Merge with provided settings
        logger.debug(f"Merging provided settings with defaults")
        logger.debug(f"Default embedding_model: '{default_settings['embedding_model']}'")
        logger.debug(f"Provided embedding_model: '{settings.get('embedding_model', 'NOT_PROVIDED')}'")

        validated_settings = {**default_settings, **settings}

        logger.info(f"After merge - embedding_model: '{validated_settings['embedding_model']}'")
        
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
        requested_embedding_model = validated_settings["embedding_model"]
        logger.info(f"Validating embedding model: '{requested_embedding_model}'")

        embedding_model = self.embedding_registry.get_model(requested_embedding_model)
        if not embedding_model:
            logger.warning(f"Embedding model '{requested_embedding_model}' not found, falling back")
            # Fallback to first available model
            available_models = self.embedding_registry.list_models()
            if available_models:
                fallback_model = available_models[0].model_id
                validated_settings["embedding_model"] = fallback_model
                logger.info(f"Using fallback embedding model: '{fallback_model}'")
            else:
                logger.error(f"No embedding models available")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No embedding models available"
                )
        else:
            logger.info(f"Embedding model '{requested_embedding_model}' validated successfully")
        
        # Validate numeric parameters
        validated_settings["temperature"] = max(0.0, min(2.0, float(validated_settings["temperature"])))
        validated_settings["max_tokens"] = max(1, min(8192, int(validated_settings["max_tokens"])))
        validated_settings["top_k_documents"] = max(1, min(20, int(validated_settings["top_k_documents"])))
        
        # Validate search type - match Enhanced Search Engine supported types
        if validated_settings["search_type"] not in ["semantic", "keyword", "hybrid", "contextual"]:
            validated_settings["search_type"] = "hybrid"
        
        # Validate reranker settings
        validated_settings["enable_reranking"] = bool(validated_settings.get("enable_reranking", False))
        if validated_settings.get("rerank_score_weight") is not None:
            validated_settings["rerank_score_weight"] = max(0.0, min(1.0, float(validated_settings["rerank_score_weight"])))
        if validated_settings.get("min_rerank_score") is not None:
            validated_settings["min_rerank_score"] = max(0.0, min(1.0, float(validated_settings["min_rerank_score"])))
        
        # Validate boolean settings
        validated_settings["enable_fallback"] = bool(validated_settings["enable_fallback"])
        validated_settings["fallback_threshold"] = max(0, int(validated_settings["fallback_threshold"]))

        logger.info(f"Settings validation completed successfully")
        logger.debug(f"Final validated settings: {validated_settings}")

        return validated_settings

    async def process_chat_message(
        self,
        session_id: str,
        message: str,
        user: User,
        db: Session,
        settings_override: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Process a chat message and yield streaming responses."""
        logger.info(f"Starting message processing - Session: {session_id}, User: {user.id}, Message length: {len(message)}")
        
        try:
            # Get session (try memory first, then database recovery)
            logger.debug(f"Retrieving session {session_id}")
            session = self.active_sessions.get(session_id)
            if not session:
                logger.info(f"Session {session_id} not found in memory, attempting recovery")
                session = await self._try_recover_session(session_id, user, db)
                
                if not session:
                    logger.error(f"Session {session_id} not found in memory or database")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Chat session not found"
                    )
            
            logger.debug(f"Session found - User: {session.user_id}, Settings: {session.settings}")
            
            # Verify session ownership
            if session.user_id != user.id:
                logger.error(f"Access denied - Session user {session.user_id} != Current user {user.id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to chat session"
                )

            # Update session settings if override provided
            if settings_override:
                logger.info(f"Applying settings override - Original embedding model: {session.settings.get('embedding_model')}")
                logger.debug(f"Settings override: {settings_override}")

                # Validate and merge new settings
                validated_override = await self._validate_chat_settings(settings_override, user, db)
                original_settings = session.settings.copy()
                session.settings.update(validated_override)

                logger.info(f"Settings updated - New embedding model: {session.settings.get('embedding_model')}")
                logger.debug(f"Original settings: {original_settings}")
                logger.debug(f"Updated settings: {session.settings}")

            # Update session activity
            session.last_activity = datetime.now(timezone.utc)
            logger.debug(f"Session activity updated")
            
            # Add user message to session
            user_message = {
                "id": str(uuid.uuid4()),
                "type": "user",
                "content": message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            session.messages.append(user_message)
            logger.info(f"User message added - Total messages: {len(session.messages)}")
            
            # Initialize search engine if using RAG
            search_results = []
            if session.settings["use_rag"]:
                logger.info(f"Starting RAG search - Embedding model: {session.settings['embedding_model']}")
                try:
                    search_results = await self._perform_rag_search(
                        message, session.settings, user, db
                    )
                    logger.info(f"RAG search completed - Found {len(search_results)} results")
                except Exception as rag_error:
                    logger.error(f"RAG search failed: {str(rag_error)}", exc_info=True)
                    # Continue without RAG if search fails
                    search_results = []
                
                # Send sources information
                if search_results:
                    logger.debug(f"Sending sources data for {len(search_results)} results")
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
                            for result in search_results[:session.settings.get("max_results", 20)]
                        ]
                    }
                    yield f"data: {json.dumps(sources_data)}\n\n"
                    logger.debug(f"Sources data sent successfully")
            else:
                logger.info(f"RAG disabled for this session")
            
            # Prepare context for LLM
            logger.info(f"Preparing LLM context - RAG results: {len(search_results)}")
            context = self._prepare_llm_context(message, search_results, session)
            context_length = len(context)
            logger.info(f"Context prepared - Length: {context_length} chars")
            
            # Generate response using selected LLM
            llm_model = session.settings["llm_model"]
            logger.info(f"Starting LLM generation - Model: {llm_model}, Stream: True")
            
            # Ensure model is registered in model manager if it's a database model
            if llm_model.startswith("registered_"):
                logger.info(f"Ensuring database model {llm_model} is registered in model manager")
                model_registered = await self._ensure_model_registered(llm_model, user, db)
                if not model_registered:
                    logger.error(f"Failed to register database model {llm_model}")
                    error_data = {
                        "type": "error",
                        "error": f"Model {llm_model} could not be loaded. Please try a different model.",
                        "error_code": "model_registration_failed"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                else:
                    logger.info(f"Database model {llm_model} successfully registered")
            
            response_content = ""
            chunk_count = 0
            
            try:
                # Get the async generator by awaiting the coroutine
                logger.info(f"Getting streaming generator from model manager")
                generator = None
                
                try:
                    generator = await self.model_manager.generate(
                        prompt=context,
                        model_id=llm_model,
                        stream=True
                    )
                    logger.info(f"Successfully obtained generator from model manager")
                    
                except Exception as model_error:
                    logger.error(f"Failed to get generator from model manager: {str(model_error)}", exc_info=True)
                    
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
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "suggestions": self._get_model_error_suggestions(error_code, llm_model)
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                
                # Verify we got a valid async generator
                if generator is None:
                    logger.error(f"Generator is None from model manager")
                    error_data = {
                        "type": "error",
                        "error": "Failed to initialize LLM response generator",
                        "error_code": "generator_none"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                
                if not hasattr(generator, '__aiter__'):
                    logger.error(f"Expected async generator, got {type(generator)}")
                    error_data = {
                        "type": "error",
                        "error": "Invalid response format from LLM service",
                        "error_code": "invalid_generator_type",
                        "received_type": str(type(generator))
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                
                # Now iterate over the async generator
                logger.info(f"Starting to iterate over streaming generator")
                
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
                                logger.debug(f"Streamed {chunk_count} chunks, total length: {len(response_content)}")
                        else:
                            logger.warning(f"Received non-string chunk: {type(chunk)} - {chunk}")
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
                                logger.warning(f"Failed to convert chunk to string: {convert_error}")
                    
                    logger.info(f"LLM generation completed - Total chunks: {chunk_count}, Response length: {len(response_content)}")
                    
                except asyncio.CancelledError:
                    logger.info(f"LLM generation was cancelled by client")
                    error_data = {
                        "type": "error",
                        "error": "Response generation was cancelled",
                        "error_code": "generation_cancelled"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                    
                except Exception as stream_error:
                    logger.error(f"Error during streaming: {str(stream_error)}", exc_info=True)
                    
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
                logger.error(f"LLM generation failed: {str(llm_error)}", exc_info=True)
                
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": session.settings["llm_model"],
                "sources": search_results[:session.settings["top_k_documents"]] if search_results else []
            }
            session.messages.append(assistant_message)
            logger.info(f"Assistant message added - ID: {assistant_message['id']}, Total messages: {len(session.messages)}")
            
            # Save updated session to database
            try:
                await self._save_session_to_db(session, db)
                logger.debug(f"Session {session_id} saved to database after message")
            except Exception as save_error:
                logger.error(f"Failed to save session to database: {str(save_error)}")
            
            # Send completion signal
            completion_data = {
                "type": "done",
                "message_id": assistant_message["id"]
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            logger.info(f"Completion signal sent successfully")
            
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
                logger.debug(f"User action logged successfully")
            except Exception as log_error:
                logger.error(f"Failed to log user action: {str(log_error)}")
            
            logger.info(f"Message processing completed successfully - Session: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to process chat message: {str(e)}", exc_info=True)
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
        """Perform RAG search using search_with_context with direct method selection."""
        import time
        start_time = time.time()

        search_type = settings.get("search_type", "semantic")
        logger.info(f"Starting RAG search - User: {user.id}, Query: '{query[:100]}...', Search type: {search_type}, Top-k: {settings.get('max_results', 20)}")

        try:
            # Initialize search engine if needed
            if not self.search_engine:
                logger.debug(f"Initializing search engine")
                from vector_db.storage_manager import VectorStorageManager
                from vector_db.embedding_manager import EnhancedEmbeddingManager

                storage_manager = VectorStorageManager()

                # Create embedding manager based on settings
                requested_model_id = settings["embedding_model"]
                logger.info(f"Requested embedding model ID: '{requested_model_id}'")

                embedding_model = self.embedding_registry.get_model(requested_model_id)
                if not embedding_model:
                    logger.error(f"Embedding model '{requested_model_id}' not found in registry")
                    available_models = self.embedding_registry.list_models()
                    logger.error(f"Available models: {[m.model_id for m in available_models]}")
                    return []

                logger.info(f"Resolved embedding model - ID: '{embedding_model.model_id}', Name: '{embedding_model.model_name}', Provider: {embedding_model.provider}")

                # Create appropriate embedding manager
                logger.info(f"Creating HuggingFace manager with model: '{embedding_model.model_name}'")
                embedding_manager = EnhancedEmbeddingManager.create_huggingface_manager(
                    embedding_model.model_name
                )
                self.search_engine = EnhancedSearchEngine(storage_manager, embedding_manager)
                logger.info(f"Search engine initialized successfully")

            # Use search_with_context with the selected search type
            logger.info(f"Using search_with_context with search type: {search_type}")
            search_start = time.time()

            search_results = await self.search_engine.search_with_context(
                query=query,
                search_type=search_type,
                user_id=user.id,
                top_k=settings.get("max_results", 20),
                db=db,
                enable_reranking=settings.get("enable_reranking", False),
                reranker_model=settings.get("reranker_model"),
                rerank_score_weight=settings.get("rerank_score_weight", 0.5),
                min_rerank_score=settings.get("min_rerank_score"),
                # Filter parameters for search
                tags=settings.get("tags"),
                tag_match_mode=settings.get("tag_match_mode"),
                exclude_tags=settings.get("exclude_tags"),
                file_type=settings.get("file_type"),
                language=settings.get("language"),
                is_public=settings.get("is_public"),
                min_score=settings.get("min_score"),
                file_size_range=settings.get("file_size_range")
            )

            search_time = (time.time() - search_start) * 1000
            logger.info(f"Search with context completed in {search_time:.2f}ms, found {len(search_results)} results")

            # Format results for chat controller compatibility
            formatted_results = []
            for result in search_results:
                document_id = result.get("document_id")
                if not document_id:
                    continue

                # Get document info for filename
                document = db.query(Document).filter(
                    and_(
                        Document.id == document_id,
                        or_(
                            Document.user_id == user.id,  # User's own documents
                            Document.is_public == True     # Public documents
                        ),
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
                        "metadata": result.get("metadata", {})
                    }
                    formatted_results.append(formatted_result)

            total_time = (time.time() - start_time) * 1000
            logger.info(f"RAG search completed in {total_time:.2f}ms - Returning {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error(f"RAG search failed after {total_time:.2f}ms: {str(e)}", exc_info=True)
            return []

    def _prepare_llm_context(
        self,
        current_message: str,
        search_results: List[Dict[str, Any]],
        session: ChatSession
    ) -> str:
        """Prepare context for LLM including RAG results and conversation history."""
        logger.info(f"Preparing context - Search results: {len(search_results)}, Message: '{current_message[:100]}...'")

        # Start with system message
        context_parts = [
            "You are a helpful AI assistant with access to document knowledge.",
            "Use the provided document context to answer questions accurately.",
            "If you cannot find relevant information in the documents, say so clearly.",
            ""
        ]

        # Add document context if available
        if search_results:
            logger.info(f"Adding {len(search_results)} search results to context")

            # Validate and filter search results
            valid_results = []
            for result in search_results:
                if result.get('text') and len(result['text'].strip()) > 20:  # At least 20 characters
                    valid_results.append(result)
                else:
                    logger.warning(f"Skipping invalid result from '{result.get('filename', 'unknown')}' - text too short or empty")

            if valid_results:
                context_parts.append("Relevant document excerpts:")
                for i, result in enumerate(valid_results, 1):
                    # Log each search result for debugging
                    score = result.get('similarity_score', 'N/A')
                    text_length = len(result['text'])
                    logger.debug(f"Result {i}: File='{result['filename']}', Score={score}, Text_length={text_length}")
                    logger.debug(f"Result {i} text preview: '{result['text'][:200]}...'")

                    # Improved formatting with score and metadata
                    context_parts.append(f"{i}. From '{result['filename']}' (relevance: {score}):")
                    context_parts.append(f"   {result['text'].strip()}")
                    context_parts.append("")

                logger.info(f"Added {len(valid_results)} valid results out of {len(search_results)} total")
            else:
                logger.warning(f"No valid search results found - all results were too short or empty")
                context_parts.append("No relevant document excerpts found.")
                context_parts.append("")
        else:
            logger.warning(f"No search results provided to context preparation")
        
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

        final_context = "\n".join(context_parts)
        logger.info(f"Context preparation completed - Total length: {len(final_context)} chars")
        logger.debug(f"Final context preview: '{final_context[:500]}...'")

        return final_context

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
            current_time = datetime.now(timezone.utc)
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