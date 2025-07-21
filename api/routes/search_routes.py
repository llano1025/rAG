from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from ..middleware.auth import get_current_user
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
    current_user = Depends(get_current_user)
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
    current_user = Depends(get_current_user)
):
    """
    Semantic similarity search using document vectors.
    """
    try:
        results = await search_controller.similarity_search(
            query_text=request.query_text,
            filters=request.filters,
            top_k=request.top_k,
            threshold=request.threshold,
            user_id=current_user.id
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/filters", response_model=List[SearchFilters])
async def get_available_filters(
    current_user = Depends(get_current_user)
):
    """
    Get available search filters including folders, tags, and document types.
    """
    filters = await search_controller.get_available_filters(
        user_id=current_user.id
    )
    return filters

@router.post("/save", response_model=dict)
async def save_search(
    request: SearchQuery,
    name: str = Query(..., description="Name for the saved search"),
    current_user = Depends(get_current_user)
):
    """
    Save a search query for later use.
    """
    saved_search = await search_controller.save_search(
        name=name,
        search_request=request,
        user_id=current_user.id
    )
    return {"message": "Search saved successfully", "id": saved_search.id}

@router.get("/saved", response_model=List[dict])
async def get_saved_searches(
    current_user = Depends(get_current_user)
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
    current_user = Depends(get_current_user)
):
    """
    Get user's recent search queries.
    """
    recent_searches = await search_controller.get_recent_searches(
        user_id=current_user.id,
        limit=limit
    )
    return recent_searches

@router.post("/suggest", response_model=List[str])
async def get_search_suggestions(
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
    current_user = Depends(get_current_user)
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