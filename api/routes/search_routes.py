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
    Full-text search across documents with filtering and ranking.
    """
    try:
        results = await search_controller.search_documents(
            query=request.query,
            filters=request.filters,
            sort=request.sort,
            page=request.page,
            page_size=request.page_size,
            user_id=current_user.id
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/similarity", response_model=SearchResponse)
async def similarity_search(
    request: SearchQuery,
    current_user = Depends(get_current_active_user)
):
    """
    Semantic similarity search using document vectors.
    """
    try:
        results = await search_controller.similarity_search(
            query_text=request.query,
            filters=request.filters,
            top_k=request.top_k,
            threshold=request.similarity_threshold or 0.0,
            user_id=current_user.id
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
    Semantic search using document vectors (alias for similarity search).
    """
    try:
        results = await search_controller.similarity_search(
            query_text=request.query,
            filters=request.filters,
            top_k=request.top_k,
            threshold=request.similarity_threshold or 0.0,
            user_id=current_user.id
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
    Hybrid search combining text search and semantic similarity.
    """
    try:
        # Perform both text search and semantic search
        text_results = await search_controller.search_documents(
            query=request.query,
            filters=request.filters,
            sort=request.sort,
            page=request.page,
            page_size=request.page_size,
            user_id=current_user.id
        )
        
        semantic_results = await search_controller.similarity_search(
            query_text=request.query,
            filters=request.filters,
            top_k=request.top_k,
            threshold=request.similarity_threshold or 0.0,
            user_id=current_user.id
        )
        
        # Combine and deduplicate results
        combined_results = text_results.copy()
        
        # Add semantic results that aren't already in text results
        text_doc_ids = {result.get('document_id') for result in text_results.get('results', [])}
        for semantic_result in semantic_results.get('results', []):
            if semantic_result.get('document_id') not in text_doc_ids:
                combined_results['results'].append(semantic_result)
        
        # Update total count
        combined_results['total_hits'] = len(combined_results.get('results', []))
        
        return combined_results
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