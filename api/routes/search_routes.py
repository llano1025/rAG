from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import logging
# datetime removed - not used
from pydantic import BaseModel, Field
from ..middleware.auth import get_current_active_user
from ..controllers import search_controller
from ..schemas.search_schemas import (
    SearchQuery,
    SearchResponse,
    convert_search_response_to_api_format,
    convert_api_filters_to_search_filter
)
from vector_db.search_engine import EnhancedSearchEngine, get_initialized_search_engine
from vector_db.search_types import SearchType
from vector_db.embedding_manager import EnhancedEmbeddingManager
from database.connection import get_db
import warnings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

class SearchMetadata(BaseModel):
    total_results: int
    execution_time: float
    filters_applied: List[str]

@router.post("/", response_model=SearchResponse)
async def unified_search(
    request: SearchQuery,
    current_user = Depends(get_current_active_user)
):
    """
    Intelligent unified search that automatically selects the best search strategy.
    Uses EnhancedSearchEngine with automatic mode selection based on query characteristics.
    """
    try:
        # Get search engine components
        # Use persistent search engine instance
        search_engine = await get_initialized_search_engine()
        embedding_manager = EnhancedEmbeddingManager.create_default_manager()
        
        db = next(get_db())
        
        # Intelligent search type detection based on query characteristics
        def detect_optimal_search_type(query: str) -> str:
            # Short queries → keyword search
            if len(query.split()) <= 2:
                return SearchType.KEYWORD
            
            # Question-like queries → semantic search  
            if query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')):
                return SearchType.SEMANTIC
            
            # Complex queries → contextual search
            if len(query.split()) > 5:
                return SearchType.CONTEXTUAL

            # Default to contextual for best results
            return SearchType.CONTEXTUAL
        
        search_type = detect_optimal_search_type(request.query)
        
        # Convert API filters to search engine format
        search_filters = convert_api_filters_to_search_filter(request.filters)
        if request.similarity_threshold is not None:
            search_filters.min_score = request.similarity_threshold
        
        # Apply reranking settings
        if hasattr(request, 'enable_reranking'):
            search_filters.enable_reranking = request.enable_reranking
        if hasattr(request, 'reranker_model'):
            search_filters.reranker_model = request.reranker_model
        if hasattr(request, 'rerank_score_weight'):
            search_filters.rerank_score_weight = request.rerank_score_weight
        if hasattr(request, 'min_rerank_score'):
            search_filters.min_rerank_score = request.min_rerank_score
        if hasattr(request, 'embedding_model'):
            search_filters.embedding_model = request.embedding_model

        # Execute unified search
        results = await search_engine.search(
            query=request.query,
            user=current_user,
            search_type=search_type,
            filters=search_filters,
            limit=request.top_k,
            db=db
        )
        
        # Convert to API response format
        return convert_search_response_to_api_format(
            results=results,
            query=request.query,
            execution_time=0.0,  # TODO: Add timing
            search_type=search_type.value,
            filters=request.filters,
            reranking_applied=getattr(search_filters, 'enable_reranking', False)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unified search failed: {str(e)}")

@router.post("/text", response_model=SearchResponse)
async def text_search(
    request: SearchQuery,
    current_user = Depends(get_current_active_user)
):
    """
    Pure text-based search using keyword matching algorithm via EnhancedSearchEngine.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Text search request: query='{request.query[:50]}...', user_id={current_user.id}, top_k={request.top_k}")
        
        # Get search engine components
        # Use persistent search engine instance
        search_engine = await get_initialized_search_engine()
        embedding_manager = EnhancedEmbeddingManager.create_default_manager()
        logger.debug("Search engine components initialized successfully")
        
        # Get database session with proper error handling
        try:
            db = next(get_db())
            logger.debug("Database session obtained")
        except Exception as db_error:
            logger.error(f"Failed to get database session: {db_error}")
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        # Convert API filters to search engine format with validation
        try:
            search_filters = convert_api_filters_to_search_filter(request.filters)
            if request.similarity_threshold is not None:
                search_filters.min_score = request.similarity_threshold
            
            logger.info(f"Search filters: min_score={getattr(search_filters, 'min_score', None)}, "
                       f"content_types={getattr(search_filters, 'content_types', None)}, "
                       f"tags={getattr(search_filters, 'tags', None)}")
        except Exception as filter_error:
            logger.error(f"Failed to convert search filters: {filter_error}")
            raise HTTPException(status_code=400, detail=f"Invalid search filters: {str(filter_error)}")
        
        # Execute keyword search using SearchEngine
        logger.info(f"Calling search() with SearchType.KEYWORD")
        results = await search_engine.search(
            query=request.query,
            user=current_user,
            search_type=SearchType.KEYWORD,
            filters=search_filters,
            limit=request.top_k,
            db=db
        )
        
        logger.info(f"Search completed successfully, returned {len(results)} results")
        
        # Convert to API response format
        response = convert_search_response_to_api_format(
            results=results,
            query=request.query,
            execution_time=0.0,  # TODO: Add timing
            search_type="keyword",
            filters=request.filters
        )
        
        logger.debug("Response formatted successfully")
        return response
        
    except Exception as e:
        error_msg = f"Text search failed: {type(e).__name__}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=error_msg)

@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
    request: SearchQuery,
    current_user = Depends(get_current_active_user)
):
    """
    Semantic search using document vectors via EnhancedSearchEngine.
    """
    try:
        # Get search engine components
        # Use persistent search engine instance
        search_engine = await get_initialized_search_engine()
        embedding_manager = EnhancedEmbeddingManager.create_default_manager()
        
        db = next(get_db())
        
        # Convert API filters to search engine format
        search_filters = convert_api_filters_to_search_filter(request.filters)
        if request.similarity_threshold is not None:
            search_filters.min_score = request.similarity_threshold
        
        # Apply reranking settings
        if hasattr(request, 'enable_reranking'):
            search_filters.enable_reranking = request.enable_reranking
        if hasattr(request, 'reranker_model'):
            search_filters.reranker_model = request.reranker_model
        if hasattr(request, 'rerank_score_weight'):
            search_filters.rerank_score_weight = request.rerank_score_weight
        if hasattr(request, 'min_rerank_score'):
            search_filters.min_rerank_score = request.min_rerank_score
        
        # Execute semantic search using SearchEngine
        results = await search_engine.search(
            query=request.query,
            user=current_user,
            search_type=SearchType.SEMANTIC,
            filters=search_filters,
            limit=request.top_k,
            db=db
        )
        
        # Convert to API response format
        return convert_search_response_to_api_format(
            results=results,
            query=request.query,
            execution_time=0.0,  # TODO: Add timing
            search_type="semantic",
            filters=request.filters,
            reranking_applied=getattr(search_filters, 'enable_reranking', False)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Semantic search failed: {str(e)}")


@router.post("/contextual", response_model=SearchResponse)
async def contextual_search(
    request: SearchQuery,
    current_user = Depends(get_current_active_user)
):
    """
    Contextual search combining content and context vectors for enhanced relevance via EnhancedSearchEngine.
    Uses weighted scoring (70% content, 30% context) for superior document understanding.
    """
    try:
        # Get search engine components
        # Use persistent search engine instance
        search_engine = await get_initialized_search_engine()
        embedding_manager = EnhancedEmbeddingManager.create_default_manager()
        
        db = next(get_db())
        
        # Convert API filters to search engine format
        search_filters = convert_api_filters_to_search_filter(request.filters)
        if request.similarity_threshold is not None:
            search_filters.min_score = request.similarity_threshold
        
        # Apply reranking settings
        if hasattr(request, 'enable_reranking'):
            search_filters.enable_reranking = request.enable_reranking
        if hasattr(request, 'reranker_model'):
            search_filters.reranker_model = request.reranker_model
        if hasattr(request, 'rerank_score_weight'):
            search_filters.rerank_score_weight = request.rerank_score_weight
        if hasattr(request, 'min_rerank_score'):
            search_filters.min_rerank_score = request.min_rerank_score
        
        # Execute contextual search using SearchEngine
        results = await search_engine.search(
            query=request.query,
            user=current_user,
            search_type=SearchType.CONTEXTUAL,
            filters=search_filters,
            limit=request.top_k,
            db=db
        )
        
        # Convert to API response format
        return convert_search_response_to_api_format(
            results=results,
            query=request.query,
            execution_time=0.0,  # TODO: Add timing
            search_type="contextual",
            filters=request.filters,
            fusion_method="contextual_content_context_vectors",
            reranking_applied=getattr(search_filters, 'enable_reranking', False)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Contextual search failed: {str(e)}")

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