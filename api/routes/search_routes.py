from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from ..middleware.auth import get_current_active_user
from ..controllers import search_controller
from ..schemas.search_schemas import (
    SearchQuery,
    SearchResponse,
    SearchFilters,
    SearchResult
)

router = APIRouter(prefix="/search", tags=["search"])

class SearchMetadata(BaseModel):
    total_results: int
    execution_time: float
    filters_applied: List[str]

@router.post("/", response_model=SearchResponse)
async def search_documents(
    request: SearchQuery,
    current_user = Depends(get_current_active_user)
):
    """
    Hybrid search across documents (recommended) with intelligent fallback to text search.
    Combines semantic similarity and keyword matching for best results.
    """
    try:
        # Use hybrid search as default, fallback to text search if needed
        try:
            # Try hybrid search first
            results = await hybrid_search(request, current_user)
            return results
        except Exception as hybrid_error:
            # Fallback to text search if hybrid fails
            results = await search_controller.search_documents(
                query=request.query,
                filters=request.filters,
                sort=request.sort,
                page=request.page,
                page_size=request.page_size,
                user_id=current_user.id,
                min_score=request.similarity_threshold if request.similarity_threshold is not None else 0.0
            )
            return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/text", response_model=SearchResponse)
async def text_search(
    request: SearchQuery,
    current_user = Depends(get_current_active_user)
):
    """
    Pure text-based search using keyword matching and intelligent query processing.
    """
    try:
        results = await search_controller.search_documents(
            query=request.query,
            filters=request.filters,
            sort=request.sort,
            page=request.page,
            page_size=request.page_size,
            user_id=current_user.id,
            min_score=request.similarity_threshold if request.similarity_threshold is not None else 0.0
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
    request: SearchQuery,
    current_user = Depends(get_current_active_user)
):
    """
    Semantic search using document vectors for intelligent similarity-based retrieval.
    """
    try:
        results = await search_controller.similarity_search(
            query_text=request.query,
            filters=request.filters,
            top_k=request.top_k,
            threshold=request.similarity_threshold if request.similarity_threshold is not None else 0.0,
            user_id=current_user.id,
            enable_reranking=request.enable_reranking,
            reranker_model=request.reranker_model,
            rerank_score_weight=request.rerank_score_weight,
            min_rerank_score=request.min_rerank_score
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    request: SearchQuery,
    current_user = Depends(get_current_active_user)
):
    """
    Professional hybrid search using Reciprocal Rank Fusion (RRF).
    """
    try:
        # Import RRF fusion module
        from utils.search.result_fusion import fuse_search_results
        
        # Use Enhanced Search Engine's built-in hybrid search
        from vector_db.search_engine import EnhancedSearchEngine, SearchType, SearchFilter
        from vector_db.storage_manager import get_storage_manager
        from vector_db.embedding_manager import EnhancedEmbeddingManager
        from database.connection import get_db
        
        db = next(get_db())
        user = current_user
        
        # Initialize search engine components
        storage_manager = get_storage_manager()
        embedding_manager = EnhancedEmbeddingManager.create_default_manager()
        search_engine = EnhancedSearchEngine(storage_manager, embedding_manager)
        
        # Create SearchFilter with min_score from request
        search_filters = SearchFilter()
        search_filters.min_score = request.similarity_threshold if request.similarity_threshold is not None else 0.0
        
        # Apply additional filters if provided in request
        if hasattr(request, 'filters') and request.filters:
            # Handle request.filters if they exist - convert from API format
            if hasattr(request.filters, 'file_types') and request.filters.file_types:
                search_filters.content_types = request.filters.file_types
            if hasattr(request.filters, 'tag_ids') and request.filters.tag_ids:
                search_filters.tags = request.filters.tag_ids
            if hasattr(request.filters, 'date_range') and request.filters.date_range:
                try:
                    start_date, end_date = request.filters.date_range
                    search_filters.date_range = (start_date, end_date)
                except (ValueError, TypeError):
                    pass
        
        # Apply reranker settings from request
        if hasattr(request, 'enable_reranking'):
            search_filters.enable_reranking = request.enable_reranking or True
        if hasattr(request, 'reranker_model'):
            search_filters.reranker_model = request.reranker_model
        if hasattr(request, 'rerank_score_weight'):
            search_filters.rerank_score_weight = request.rerank_score_weight or 0.5
        if hasattr(request, 'min_rerank_score'):
            search_filters.min_rerank_score = request.min_rerank_score
        
        # Perform hybrid search using Enhanced Search Engine with proper filtering
        search_results = await search_engine.search(
            query=request.query,
            user=user,
            search_type=SearchType.HYBRID,
            filters=search_filters,
            limit=request.top_k,
            db=db
        )
        
        # Format results to match API schema
        fused_results = []
        for result in search_results:
            formatted_result = {
                "document_id": str(result.document_id),
                "filename": result.document_metadata.get("filename", "Unknown"),
                "content_snippet": result.text[:300] + "..." if len(result.text) > 300 else result.text,
                "score": result.score,
                "metadata": {
                    **result.metadata,
                    **result.document_metadata,
                    "search_type": "hybrid"
                }
            }
            fused_results.append(formatted_result)
        
        # Build final response
        hybrid_response = {
            "results": fused_results,
            "total_hits": len(fused_results),
            "execution_time_ms": 0,  # Enhanced search engine doesn't return timing yet
            "filters_applied": request.filters,
            "query_vector_id": None,  # Could be enhanced to return this
            "query": request.query,
            "processing_time": 0.0,  # Could be enhanced to measure this
            "fusion_method": "enhanced_search_engine_hybrid"
        }
        
        return hybrid_response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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

class SearchType(BaseModel):
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
    search_types: List[SearchType]

@router.get("/filters", response_model=AvailableFilters)
async def get_available_filters(
    current_user = Depends(get_current_active_user)
):
    """
    Get available search filters including folders, tags, and document types.
    """
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
    Get search suggestions based on partial query.
    """
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
    current_user = Depends(get_current_active_user)
):
    """
    Get list of available reranker models.
    """
    try:
        from vector_db.reranker import get_reranker_manager
        
        reranker_manager = get_reranker_manager()
        models = await reranker_manager.get_available_models()
        
        return [RerankerModel(**model) for model in models]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get reranker models: {str(e)}")

@router.get("/reranker/health")
async def check_reranker_health(
    model_name: Optional[str] = Query(None, description="Specific model to check"),
    current_user = Depends(get_current_active_user)
):
    """
    Check health status of reranker models.
    """
    try:
        from vector_db.reranker import get_reranker_manager
        
        reranker_manager = get_reranker_manager()
        health_status = await reranker_manager.health_check(model_name)
        
        return health_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")