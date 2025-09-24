from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import logging
# datetime removed - not used
from pydantic import BaseModel, Field
from ..middleware.auth import get_current_active_user
from ..controllers import search_controller
from ..controllers.search_controller import SearchController
from ..schemas.search_schemas import (
    SearchQuery,
    SearchResponse,
    convert_search_response_to_api_format,
    convert_api_filters_to_search_filter
)
from vector_db.search_manager import get_initialized_search_engine
from vector_db.search_types import SearchType
from vector_db.embedding_manager import EnhancedEmbeddingManager
from database.connection import get_db
from utils.security.audit_logger import AuditLogger, AuditLoggerConfig
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

class SearchMetadata(BaseModel):
    total_results: int
    execution_time: float
    filters_applied: List[str]

async def get_search_controller(
    current_user = Depends(get_current_active_user),
    db = Depends(get_db)
) -> SearchController:
    """Dependency to get initialized SearchController."""
    search_engine = await get_initialized_search_engine()

    # Create audit logger configuration
    audit_config = AuditLoggerConfig(
        log_path=Path("/tmp/audit_logs"),  # Default path, should be configurable
        rotation_size_mb=10,
        retention_days=90,
        batch_size=100,
        flush_interval_seconds=5,
        enable_async=True
    )
    audit_logger = AuditLogger(audit_config)

    return SearchController(
        search_engine=search_engine,
        db=db,
        audit_logger=audit_logger
    )

@router.post("/", response_model=SearchResponse)
async def search(
    request: SearchQuery,
    current_user = Depends(get_current_active_user),
    search_controller = Depends(get_search_controller)
):
    """
    Unified intelligent search that automatically selects the best search strategy.
    Uses SearchController with EnhancedSearchEngine and supports MMR and reranking.
    """
    try:
        # Execute search using SearchController
        results = await search_controller.search(
            query=request.query,
            user=current_user,
            search_type=request.search_type,
            filters=request.filters,
            top_k=request.top_k or 20,
            similarity_threshold=request.similarity_threshold,
            embedding_model=getattr(request, 'embedding_model', None)
        )

        # Convert to API response format using schema converter
        return convert_search_response_to_api_format(
            results=results,
            query=request.query,
            execution_time=0.0,  # Controller handles timing internally
            search_type=request.search_type,
            filters=request.filters,
            reranking_applied=getattr(request, 'enable_reranking', False)
        )
    except Exception as e:
        logger.error(f"Unified search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

class FilterOption(BaseModel):
    value: str
    label: str
    count: Optional[int] = None
    icon: Optional[str] = None

class DateRange(BaseModel):
    min_date: Optional[str] = None
    max_date: Optional[str] = None

class FileSizeRange(BaseModel):
    min_size: int
    max_size: int
    avg_size: int

class SearchTypeOption(BaseModel):
    value: str
    label: str
    description: str

class AvailableFilters(BaseModel):
    file_types: List[FilterOption]
    tags: List[FilterOption]
    languages: List[FilterOption]
    folders: List[FilterOption]
    date_range: DateRange
    file_size_range: FileSizeRange
    search_types: List[SearchTypeOption]

@router.get("/filters", response_model=AvailableFilters)
async def get_available_filters(
    current_user = Depends(get_current_active_user)
):
    """
    Get available search filters including folders, tags, and document types via EnhancedSearchEngine.
    """
    try:
        # Get search engine components
        # Use persistent search engine instance
        search_engine = await get_initialized_search_engine()
        embedding_manager = EnhancedEmbeddingManager.create_default_manager()
        
        db = next(get_db())
        
        # Get filters using unified search engine
        filters = await search_engine.get_available_filters(
            user=current_user,
            db=db
        )
        return filters
        
    except Exception:
        # Fallback to legacy controller with deprecation warning
        warnings.warn("Falling back to legacy search controller for filters", DeprecationWarning)
        filters = await search_controller.get_available_filters(
            user_id=current_user.id
        )
        return filters

class SaveSearchRequest(BaseModel):
    search_query: SearchQuery
    name: str = Field(..., description="Name for the saved search")

@router.post("/save", response_model=dict)
async def save_search(
    request: SaveSearchRequest,
    current_user = Depends(get_current_active_user)
):
    """
    Save a search query for later use.
    """
    # Use legacy controller for now - this functionality doesn't need SearchEngine
    saved_search = await search_controller.save_search(
        name=request.name,
        search_request=request.search_query.model_dump(),
        user_id=current_user.id
    )
    return {"message": "Search saved successfully", "id": saved_search.id, "name": request.name}

@router.get("/saved", response_model=List[dict])
async def get_saved_searches(
    current_user = Depends(get_current_active_user)
):
    """
    Retrieve user's saved searches.
    """
    # Use legacy controller for now - this functionality doesn't need SearchEngine
    saved_searches = await search_controller.get_saved_searches(
        user_id=current_user.id
    )
    return saved_searches

@router.get("/recent", response_model=List[dict])
async def get_recent_searches(
    limit: int = Query(10, ge=1, le=50),
    current_user = Depends(get_current_active_user)
):
    """
    Get user's recent search queries.
    """
    # Use legacy controller for now - this functionality doesn't need SearchEngine
    recent_searches = await search_controller.get_recent_searches(
        user_id=current_user.id,
        limit=limit
    )
    return recent_searches

@router.get("/history", response_model=List[dict])
async def get_search_history(
    limit: int = Query(20, ge=1, le=100),
    current_user = Depends(get_current_active_user)
):
    """
    Get user's search history.
    """
    # Use legacy controller for now - this functionality doesn't need SearchEngine
    history = await search_controller.get_recent_searches(
        user_id=current_user.id,
        limit=limit
    )
    return history

class SearchSuggestion(BaseModel):
    type: str = Field(..., description="Type of suggestion (history, tag, document_title, etc.)")
    text: str = Field(..., description="Suggestion text")
    icon: str = Field(..., description="Icon name for the suggestion")

@router.get("/suggestions", response_model=List[SearchSuggestion])
async def get_search_suggestions(
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
    current_user = Depends(get_current_active_user)
):
    """
    Get search suggestions based on partial query via EnhancedSearchEngine.
    """
    try:
        # Get search engine components
        # Use persistent search engine instance
        search_engine = await get_initialized_search_engine()
        embedding_manager = EnhancedEmbeddingManager.create_default_manager()
        
        db = next(get_db())
        
        # Get suggestions using unified search engine
        suggestions_data = await search_engine.get_search_suggestions(
            query=query,
            user=current_user,
            limit=limit,
            db=db
        )
        
        # Convert to API format
        suggestions = [SearchSuggestion(**suggestion) for suggestion in suggestions_data]
        return suggestions
        
    except Exception:
        # Fallback to legacy controller with deprecation warning
        warnings.warn("Falling back to legacy search controller for suggestions", DeprecationWarning)
        suggestions = await search_controller.get_search_suggestions(
            query=query,
            limit=limit,
            user_id=current_user.id
        )
        return suggestions

class RerankerModel(BaseModel):
    alias: str = Field(..., description="Model alias/name")
    full_name: str = Field(..., description="Full model path")
    description: str = Field(..., description="Model description")
    performance_tier: str = Field(..., description="Performance tier (fast/balanced/accurate)")
    provider: str = Field(..., description="Model provider")

@router.get("/reranker/models", response_model=List[RerankerModel])
async def get_available_reranker_models(
    current_user = Depends(get_current_active_user)  # Authentication required
):
    """
    Get list of available reranker models.
    """
    try:
        from vector_db.reranker import get_reranker_manager
        
        reranker_manager = get_reranker_manager()
        models = await reranker_manager.get_available_models()
        
        return [RerankerModel(**model) for model in models]
        
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get reranker models")

@router.get("/reranker/health")
async def check_reranker_health(
    model_name: Optional[str] = Query(None, description="Specific model to check"),
    current_user = Depends(get_current_active_user)  # Authentication required
):
    """
    Check health status of reranker models.
    """
    try:
        from vector_db.reranker import get_reranker_manager
        
        reranker_manager = get_reranker_manager()
        health_status = await reranker_manager.health_check(model_name)
        
        return health_status
        
    except Exception:
        raise HTTPException(status_code=500, detail="Health check failed")