"""
Model Management API Routes

Provides endpoints for:
- Model discovery from different providers
- Model registration and management
- Model testing and validation
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import logging

from database.connection import get_db
from database.models import RegisteredModel, ModelTest, ModelProviderEnum
from api.middleware.auth import get_current_active_user
from llm.services.model_discovery_service import get_discovery_service
from llm.base.models import ModelConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["models"])

# Pydantic schemas for request/response
class ModelDiscoveryRequest(BaseModel):
    provider: str = Field(..., description="Provider name (openai, gemini, anthropic, ollama, lmstudio)")
    api_key: Optional[str] = Field(None, description="API key for providers that require it")
    base_url: Optional[str] = Field(None, description="Custom base URL for local providers")

class ModelRegistrationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="User-defined model name")
    display_name: Optional[str] = Field(None, max_length=200, description="Display name for the model")
    description: Optional[str] = Field(None, description="Model description")
    llm_model_name: str = Field(..., min_length=1, max_length=200, description="Actual model name for API calls")
    provider: ModelProviderEnum = Field(..., description="Model provider")
    config: Dict[str, Any] = Field(..., description="Model configuration parameters")
    provider_config: Optional[Dict[str, Any]] = Field(None, description="Provider-specific configuration")
    is_public: bool = Field(False, description="Whether to share model with other users")
    fallback_priority: Optional[int] = Field(None, description="Priority in fallback chain")

class ModelUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    display_name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None)
    config: Optional[Dict[str, Any]] = Field(None)
    provider_config: Optional[Dict[str, Any]] = Field(None)
    is_public: Optional[bool] = Field(None)
    is_active: Optional[bool] = Field(None)
    fallback_priority: Optional[int] = Field(None)

class ModelTestRequest(BaseModel):
    test_type: str = Field(..., description="Type of test (connectivity, generation, embedding)")
    test_prompt: Optional[str] = Field("Hello, how are you?", description="Test prompt for generation tests")
    timeout_seconds: int = Field(30, ge=5, le=300, description="Test timeout in seconds")

class RegisteredModelResponse(BaseModel):
    id: int
    name: str
    display_name: Optional[str]
    description: Optional[str]
    llm_model_name: str
    provider: str
    is_active: bool
    is_public: bool
    usage_count: int
    success_rate: float
    average_response_time: Optional[float]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class ModelTestResponse(BaseModel):
    id: int
    test_type: str
    status: str
    response_time_ms: Optional[float]
    error_message: Optional[str]
    created_at: str
    completed_at: Optional[str]

    class Config:
        from_attributes = True

# ==============================================================================
# Provider and Discovery Endpoints
# ==============================================================================

@router.get("/providers")
async def get_providers():
    """Get list of all available LLM providers with their information."""
    try:
        discovery_service = get_discovery_service()
        providers = await discovery_service.get_available_providers()
        return {
            "providers": providers,
            "total": len(providers)
        }
    except Exception as e:
        logger.error(f"Error getting providers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {str(e)}")

@router.post("/discover/{provider}")
async def discover_models(
    provider: str,
    request: Optional[ModelDiscoveryRequest] = None,
    _current_user = Depends(get_current_active_user)
):
    """Discover available models from a specific provider."""
    try:
        discovery_service = get_discovery_service()
        
        api_key = request.api_key if request else None
        base_url = request.base_url if request else None
        
        models = await discovery_service.discover_models(provider, api_key, base_url)
        
        return {
            "provider": provider,
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description,
                    "context_window": model.context_window,
                    "max_tokens": model.max_tokens,
                    "supports_streaming": model.supports_streaming,
                    "supports_embeddings": model.supports_embeddings,
                    "is_available": model.is_available,
                    "metadata": model.metadata
                }
                for model in models
            ],
            "total": len(models)
        }
    except Exception as e:
        logger.error(f"Error discovering models for {provider}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to discover models: {str(e)}")

@router.get("/templates")
async def get_model_templates(
    provider: Optional[str] = Query(None, description="Filter by specific provider"),
    _current_user = Depends(get_current_active_user)
):
    """Get model configuration templates."""
    try:
        discovery_service = get_discovery_service()
        templates = await discovery_service.get_model_templates(provider)
        
        return {
            "templates": templates,
            "providers": list(templates.keys()) if provider else None
        }
    except Exception as e:
        logger.error(f"Error getting model templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

@router.post("/test-connectivity/{provider}")
async def test_provider_connectivity(
    provider: str,
    request: Optional[ModelDiscoveryRequest] = None,
    _current_user = Depends(get_current_active_user)
):
    """Test connectivity to a specific provider."""
    try:
        discovery_service = get_discovery_service()
        
        api_key = request.api_key if request else None
        base_url = request.base_url if request else None
        
        result = await discovery_service.test_provider_connectivity(provider, api_key, base_url)
        
        return result
    except Exception as e:
        logger.error(f"Error testing connectivity for {provider}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test connectivity: {str(e)}")

# ==============================================================================
# Model Registration Endpoints
# ==============================================================================

@router.post("/register", response_model=RegisteredModelResponse)
async def register_model(
    request: ModelRegistrationRequest,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Register a new LLM model."""
    try:
        # Validate model configuration
        try:
            model_config = ModelConfig.from_dict(request.config)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid model configuration: {str(e)}")
        
        # Check if model name already exists for this user
        existing = db.query(RegisteredModel).filter(
            RegisteredModel.user_id == current_user.id,
            RegisteredModel.name == request.name,
            RegisteredModel.deleted_at.is_(None)
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail=f"Model with name '{request.name}' already exists")
        
        # Create new registered model
        registered_model = RegisteredModel(
            user_id=current_user.id,
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            model_name=request.llm_model_name,
            provider=request.provider,
            is_public=request.is_public,
            fallback_priority=request.fallback_priority,
            context_window=model_config.context_window,
            max_tokens=model_config.max_tokens
        )
        
        # Set configurations
        registered_model.set_config(request.config)
        if request.provider_config:
            registered_model.set_provider_config(request.provider_config)
        
        db.add(registered_model)
        db.commit()
        db.refresh(registered_model)
        
        logger.info(f"User {current_user.id} registered model: {request.name} ({request.provider})")
        
        return RegisteredModelResponse(
            id=registered_model.id,
            name=registered_model.name,
            display_name=registered_model.display_name,
            description=registered_model.description,
            llm_model_name=registered_model.model_name,
            provider=registered_model.provider.value,
            is_active=registered_model.is_active,
            is_public=registered_model.is_public,
            usage_count=registered_model.usage_count,
            success_rate=registered_model.success_rate,
            average_response_time=registered_model.average_response_time,
            created_at=registered_model.created_at.isoformat(),
            updated_at=registered_model.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")

@router.get("/registered", response_model=List[RegisteredModelResponse])
async def get_registered_models(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    active_only: bool = Query(True, description="Show only active models"),
    include_public: bool = Query(False, description="Include public models from other users"),
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get list of registered models for the current user."""
    try:
        query = db.query(RegisteredModel).filter(RegisteredModel.deleted_at.is_(None))
        
        if include_public:
            # Include user's own models and public models from others
            query = query.filter(
                (RegisteredModel.user_id == current_user.id) | 
                (RegisteredModel.is_public == True)
            )
        else:
            # Only user's own models
            query = query.filter(RegisteredModel.user_id == current_user.id)
        
        if provider:
            try:
                provider_enum = ModelProviderEnum(provider.lower())
                query = query.filter(RegisteredModel.provider == provider_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
        
        if active_only:
            query = query.filter(RegisteredModel.is_active == True)
        
        models = query.order_by(RegisteredModel.created_at.desc()).all()
        
        return [
            RegisteredModelResponse(
                id=model.id,
                name=model.name,
                display_name=model.display_name,
                description=model.description,
                llm_model_name=model.model_name,
                provider=model.provider.value,
                is_active=model.is_active,
                is_public=model.is_public,
                usage_count=model.usage_count,
                success_rate=model.success_rate,
                average_response_time=model.average_response_time,
                created_at=model.created_at.isoformat(),
                updated_at=model.updated_at.isoformat()
            )
            for model in models
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting registered models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@router.get("/registered/{model_id}")
async def get_registered_model(
    model_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get details of a specific registered model."""
    try:
        model = db.query(RegisteredModel).filter(
            RegisteredModel.id == model_id,
            RegisteredModel.deleted_at.is_(None)
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Check if user has access (owns model or model is public)
        if model.user_id != current_user.id and not model.is_public:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "id": model.id,
            "name": model.name,
            "display_name": model.display_name,
            "description": model.description,
            "llm_model_name": model.model_name,
            "provider": model.provider.value,
            "config": model.get_config(),
            "provider_config": model.get_provider_config(),
            "is_active": model.is_active,
            "is_public": model.is_public,
            "fallback_priority": model.fallback_priority,
            "usage_count": model.usage_count,
            "success_rate": model.success_rate,
            "average_response_time": model.average_response_time,
            "total_tokens_used": model.total_tokens_used,
            "estimated_cost": model.estimated_cost,
            "last_used": model.last_used.isoformat() if model.last_used else None,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat(),
            "owner": {
                "id": model.user.id,
                "username": model.user.username
            } if model.user_id != current_user.id else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")

@router.put("/registered/{model_id}", response_model=RegisteredModelResponse)
async def update_registered_model(
    model_id: int,
    request: ModelUpdateRequest,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update a registered model."""
    try:
        model = db.query(RegisteredModel).filter(
            RegisteredModel.id == model_id,
            RegisteredModel.user_id == current_user.id,
            RegisteredModel.deleted_at.is_(None)
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Update fields if provided
        if request.name is not None:
            # Check for name conflicts
            existing = db.query(RegisteredModel).filter(
                RegisteredModel.user_id == current_user.id,
                RegisteredModel.name == request.name,
                RegisteredModel.id != model_id,
                RegisteredModel.deleted_at.is_(None)
            ).first()
            
            if existing:
                raise HTTPException(status_code=400, detail=f"Model with name '{request.name}' already exists")
            
            model.name = request.name
        
        if request.display_name is not None:
            model.display_name = request.display_name
        
        if request.description is not None:
            model.description = request.description
        
        if request.config is not None:
            # Validate configuration
            try:
                model_config = ModelConfig.from_dict(request.config)
                model.set_config(request.config)
                model.context_window = model_config.context_window
                model.max_tokens = model_config.max_tokens
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid model configuration: {str(e)}")
        
        if request.provider_config is not None:
            model.set_provider_config(request.provider_config)
        
        if request.is_public is not None:
            model.is_public = request.is_public
        
        if request.is_active is not None:
            model.is_active = request.is_active
        
        if request.fallback_priority is not None:
            model.fallback_priority = request.fallback_priority
        
        db.commit()
        db.refresh(model)
        
        logger.info(f"User {current_user.id} updated model: {model.name}")
        
        return RegisteredModelResponse(
            id=model.id,
            name=model.name,
            display_name=model.display_name,
            description=model.description,
            llm_model_name=model.model_name,
            provider=model.provider.value,
            is_active=model.is_active,
            is_public=model.is_public,
            usage_count=model.usage_count,
            success_rate=model.success_rate,
            average_response_time=model.average_response_time,
            created_at=model.created_at.isoformat(),
            updated_at=model.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model {model_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")

@router.delete("/registered/{model_id}")
async def delete_registered_model(
    model_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a registered model (soft delete)."""
    try:
        model = db.query(RegisteredModel).filter(
            RegisteredModel.id == model_id,
            RegisteredModel.user_id == current_user.id,
            RegisteredModel.deleted_at.is_(None)
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Soft delete
        from datetime import datetime, timezone
        model.deleted_at = datetime.now(timezone.utc)
        model.is_active = False
        
        db.commit()
        
        logger.info(f"User {current_user.id} deleted model: {model.name}")
        
        return {"message": f"Model '{model.name}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

# ==============================================================================
# Model Loading and Management Endpoints
# ==============================================================================

@router.post("/load-registered")
async def load_registered_models(
    user_id: Optional[int] = Query(None, description="Load models for specific user (admin only)"),
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Load all registered models into the ModelManager."""
    try:
        from llm.services.model_registration_service import get_registration_service
        
        # Check if user_id filtering is requested
        if user_id is not None and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required to load models for specific user")
        
        # Use current user if no user_id specified and not admin
        load_user_id = user_id if current_user.is_admin else current_user.id
        
        registration_service = get_registration_service()
        loaded_count = await registration_service.load_all_registered_models(load_user_id)
        
        return {
            "message": f"Loaded {loaded_count} registered models",
            "loaded_count": loaded_count,
            "user_id": load_user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading registered models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@router.post("/sync")
async def sync_registered_models(
    current_user = Depends(get_current_active_user)
):
    """Synchronize loaded models with database state."""
    try:
        from llm.services.model_registration_service import get_registration_service
        
        registration_service = get_registration_service()
        stats = await registration_service.sync_with_database(current_user.id)
        
        return {
            "message": "Model synchronization completed",
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error syncing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to sync models: {str(e)}")

@router.get("/loaded")
async def get_loaded_models(
    current_user = Depends(get_current_active_user)
):
    """Get list of currently loaded models."""
    try:
        from llm.services.model_registration_service import get_registration_service
        
        registration_service = get_registration_service()
        model_infos = await registration_service.get_model_info_list(current_user.id)
        
        return {
            "models": [
                {
                    "model_id": info.model_id,
                    "model_name": info.model_name,
                    "provider": info.provider,
                    "display_name": info.display_name,
                    "description": info.description,
                    "context_window": info.context_window,
                    "max_tokens": info.max_tokens,
                    "supports_streaming": info.supports_streaming,
                    "supports_embeddings": info.supports_embeddings,
                    "capabilities": info.capabilities
                }
                for info in model_infos
            ],
            "total": len(model_infos)
        }
        
    except Exception as e:
        logger.error(f"Error getting loaded models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get loaded models: {str(e)}")

@router.post("/registered/{model_id}/reload")
async def reload_registered_model(
    model_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Reload a specific registered model."""
    try:
        # Check if user owns the model
        model = db.query(RegisteredModel).filter(
            RegisteredModel.id == model_id,
            RegisteredModel.user_id == current_user.id,
            RegisteredModel.deleted_at.is_(None)
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        from llm.services.model_registration_service import get_registration_service
        
        registration_service = get_registration_service()
        success = await registration_service.reload_model(model_id)
        
        if success:
            return {"message": f"Model '{model.name}' reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

# ==============================================================================
# Model Testing Endpoints
# ==============================================================================

async def _run_model_test(model_id: int, test_request: ModelTestRequest, db: Session):
    """Background task to run model test."""
    from llm.model_manager import ModelManager
    from llm.base.models import ModelConfig
    
    test = None
    try:
        # Get the test record
        test = db.query(ModelTest).filter(ModelTest.id == model_id).first()
        if not test:
            return
        
        test.start_test()
        db.commit()
        
        # Get the model
        model = test.model
        if not model:
            test.complete_test(False, error_message="Model not found", error_code="MODEL_NOT_FOUND")
            db.commit()
            return
        
        # Create model manager and register the model
        manager = ModelManager()
        config = ModelConfig.from_dict(model.get_config())
        provider_config = model.get_provider_config()
        
        manager.register_model(
            model_id=str(model.id),
            provider_name=model.provider.value,
            config=config,
            provider_kwargs=provider_config
        )
        
        # Run the test
        if test_request.test_type == "connectivity":
            # Simple health check
            health_results = await manager.health_check_all_models()
            model_health = health_results.get(str(model.id), {})
            
            if model_health.get('available', False):
                test.complete_test(True, response_text="Health check passed")
            else:
                test.complete_test(False, error_message="Health check failed", error_code="HEALTH_CHECK_FAILED")
        
        elif test_request.test_type == "generation":
            # Test text generation
            import time
            start_time = time.time()
            
            response = await manager.generate_with_fallback(
                prompt=test_request.test_prompt,
                primary_model_id=str(model.id),
                stream=False
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            if response and response.text:
                test.complete_test(
                    True, 
                    response_text=response.text[:1000],  # Limit response text
                    response_time_ms=response_time_ms,
                    tokens_used=response.usage.total_tokens if response.usage else None
                )
            else:
                test.complete_test(False, error_message="No response generated", error_code="NO_RESPONSE")
        
        elif test_request.test_type == "embedding":
            # Test embedding generation
            import time
            start_time = time.time()
            
            embeddings = await manager.get_embedding_with_fallback(
                text="Test embedding text",
                primary_model_id=str(model.id)
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            if embeddings and len(embeddings) > 0:
                test.complete_test(
                    True,
                    response_text=f"Generated {len(embeddings)} dimensional embedding",
                    response_time_ms=response_time_ms
                )
            else:
                test.complete_test(False, error_message="No embeddings generated", error_code="NO_EMBEDDINGS")
        
        db.commit()
        
    except Exception as e:
        if test:
            test.complete_test(False, error_message=str(e), error_code="TEST_ERROR")
            db.commit()

@router.post("/registered/{model_id}/test", response_model=ModelTestResponse)
async def test_registered_model(
    model_id: int,
    request: ModelTestRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Test a registered model."""
    try:
        model = db.query(RegisteredModel).filter(
            RegisteredModel.id == model_id,
            RegisteredModel.deleted_at.is_(None)
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Check if user has access
        if model.user_id != current_user.id and not model.is_public:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create test record
        test = ModelTest(
            model_id=model_id,
            test_type=request.test_type,
            test_prompt=request.test_prompt,
            timeout_seconds=request.timeout_seconds
        )
        
        test.set_test_parameters({
            "test_type": request.test_type,
            "timeout_seconds": request.timeout_seconds
        })
        
        db.add(test)
        db.commit()
        db.refresh(test)
        
        # Run test in background
        background_tasks.add_task(_run_model_test, test.id, request, db)
        
        return ModelTestResponse(
            id=test.id,
            test_type=test.test_type,
            status=test.status.value,
            response_time_ms=test.response_time_ms,
            error_message=test.error_message,
            created_at=test.created_at.isoformat(),
            completed_at=test.completed_at.isoformat() if test.completed_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test model: {str(e)}")

@router.get("/registered/{model_id}/tests", response_model=List[ModelTestResponse])
async def get_model_tests(
    model_id: int,
    limit: int = Query(10, ge=1, le=100),
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get test history for a registered model."""
    try:
        model = db.query(RegisteredModel).filter(
            RegisteredModel.id == model_id,
            RegisteredModel.deleted_at.is_(None)
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Check if user has access
        if model.user_id != current_user.id and not model.is_public:
            raise HTTPException(status_code=403, detail="Access denied")
        
        tests = db.query(ModelTest).filter(
            ModelTest.model_id == model_id
        ).order_by(ModelTest.created_at.desc()).limit(limit).all()
        
        return [
            ModelTestResponse(
                id=test.id,
                test_type=test.test_type,
                status=test.status.value,
                response_time_ms=test.response_time_ms,
                error_message=test.error_message,
                created_at=test.created_at.isoformat(),
                completed_at=test.completed_at.isoformat() if test.completed_at else None
            )
            for test in tests
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tests for model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tests: {str(e)}")