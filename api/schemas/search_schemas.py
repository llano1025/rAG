from typing import List, Optional
from pydantic import BaseModel, Field, confloat

class SearchFilters(BaseModel):
    folder_ids: Optional[List[str]] = Field(None, description="Filter by folder IDs")
    tags: Optional[List[str]] = Field(None, description="Filter by tags (case-insensitive)")
    tag_match_mode: Optional[str] = Field(
        "any", 
        description="Tag matching mode: 'any' (OR logic), 'all' (AND logic), or 'exact' (exact match)",
        pattern="^(any|all|exact)$"
    )
    exclude_tags: Optional[List[str]] = Field(None, description="Tags to exclude from results")
    file_types: Optional[List[str]] = Field(None, description="Filter by file types (MIME types)")
    date_range: Optional[tuple[str, str]] = Field(None, description="Filter by date range (ISO format)")
    file_size_range: Optional[tuple[int, int]] = Field(None, description="Filter by file size range (bytes)")
    language: Optional[str] = Field(None, description="Filter by document language")
    is_public: Optional[bool] = Field(None, description="Filter by public/private status")
    metadata_filters: Optional[dict] = Field(None, description="Filter by custom metadata")
    embedding_model: Optional[str] = Field(None, description="Filter by embedding model used to index documents")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tags": ["python", "machine-learning"],
                "tag_match_mode": "any",
                "exclude_tags": ["deprecated"],
                "file_types": ["application/pdf", "text/plain"],
                "date_range": ["2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z"],
                "file_size_range": [1024, 10485760],
                "language": "en",
                "is_public": False,
                "embedding_model": "all-MiniLM-L6-v2"
            }
        }

class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query text")
    filters: Optional[SearchFilters] = None
    semantic_search: bool = Field(True, description="Whether to use semantic search")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    similarity_threshold: Optional[float] = Field(
        None, 
        description="Minimum similarity score threshold",
        ge=0.0,
        le=1.0
    )
    # Reranker settings
    enable_reranking: bool = Field(False, description="Whether to enable reranking of search results")
    reranker_model: Optional[str] = Field(None, description="Reranker model to use (e.g., 'ms-marco-MiniLM-L-6-v2')")
    rerank_score_weight: float = Field(
        0.5, 
        description="Weight for rerank score vs original score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    min_rerank_score: Optional[float] = Field(
        None,
        description="Minimum rerank score threshold",
        ge=0.0,
        le=1.0
    )
    # Embedding model selection
    embedding_model: Optional[str] = Field(None, description="Embedding model to use for query encoding (e.g., 'all-MiniLM-L6-v2')")
    # Pagination fields
    page: int = Field(1, description="Page number for pagination", ge=1)
    page_size: int = Field(10, description="Number of results per page", ge=1, le=100)
    # Sorting field
    sort: Optional[str] = Field(None, description="Sort order (relevance, date, title)")

from pydantic import BaseModel, Field
from typing import List, Tuple, Dict
from decimal import Decimal

class SearchResult(BaseModel):
    """
    Represents a single search result from the RAG system.
    
    Attributes:
        document_id: Unique identifier of the document
        filename: Name of the document file
        content_snippet: Relevant text excerpt from the document
        score: Relevance score between 0 and 1
        metadata: Additional document metadata
    """
    document_id: str = Field(..., description="Unique identifier of the document")
    filename: str = Field(..., description="Name of the document file")
    content_snippet: str = Field(..., description="Relevant text excerpt from the document")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score between 0 and 1"
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional document metadata"
    )

    class Config:
        """Pydantic model configuration"""
        json_schema_extra = {
            "example": {
                "document_id": "doc123",
                "filename": "example.pdf",
                "content_snippet": "This is a relevant excerpt from the document...",
                "score": 0.95,
                "metadata": {
                    "title": "Example Document",
                    "created_at": "2024-02-08T12:00:00Z",
                    "content_type": "application/pdf",
                    "tags": ["python", "machine-learning", "tutorial"],
                    "version": 1
                }
            }
        }

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_hits: int
    execution_time_ms: float
    filters_applied: Optional[SearchFilters] = None
    query_vector_id: Optional[str] = Field(None, description="ID of generated query vector")
    query: Optional[str] = Field(None, description="Original search query")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    # Additional compatibility fields
    search_type: Optional[str] = Field(None, description="Type of search performed (semantic/keyword/hybrid)")
    fusion_method: Optional[str] = Field(None, description="Method used for result fusion")
    reranking_applied: bool = Field(False, description="Whether reranking was applied to results")
    cache_hit: bool = Field(False, description="Whether results came from cache")
    
    class Config:
        """Pydantic model configuration"""
        json_schema_extra = {
            "example": {
                "results": [],
                "total_hits": 0,
                "execution_time_ms": 150.5,
                "query": "machine learning",
                "search_type": "semantic",
                "reranking_applied": True,
                "cache_hit": False
            }
        }


class AvailableFilters(BaseModel):
    """Available filter options for the user interface."""
    file_types: List[dict] = Field(default_factory=list)
    tags: List[dict] = Field(default_factory=list)
    languages: List[dict] = Field(default_factory=list)
    folders: List[dict] = Field(default_factory=list)
    date_range: dict = Field(default_factory=dict)
    file_size_range: dict = Field(default_factory=dict)
    search_types: List[dict] = Field(default_factory=list)


class SearchSuggestion(BaseModel):
    """Search suggestion for autocomplete."""
    type: str = Field(..., description="Type of suggestion (history/document_title/content)")
    text: str = Field(..., description="Suggested text")
    icon: str = Field(..., description="Icon for the suggestion")


class SavedSearchResponse(BaseModel):
    """Response for saved search operations."""
    id: int
    name: str
    search_query: dict
    created_at: str


class RecentSearchResponse(BaseModel):
    """Response for recent search queries."""
    id: int
    query: str
    created_at: str
    results_count: int = 0


# Helper function to convert SearchResult from EnhancedSearchEngine to API format
def convert_search_result_to_api_format(result, search_type: str = "semantic") -> SearchResult:
    """Convert SearchEngine result to API SearchResult format."""
    return SearchResult(
        document_id=str(result.document_id),
        filename=result.document_metadata.get("filename", "Unknown"),
        content_snippet=result.text[:300] + "..." if len(result.text) > 300 else result.text,
        score=result.score,
        metadata={
            **result.metadata,
            **result.document_metadata,
            "search_type": search_type,
            "chunk_id": result.chunk_id,
            "chunk_index": getattr(result, 'chunk_index', 0)
        }
    )


def convert_search_response_to_api_format(results: List, query: str, 
                                        execution_time: float = 0.0,
                                        search_type: str = "semantic",
                                        filters: Optional[SearchFilters] = None,
                                        **kwargs) -> SearchResponse:
    """Convert SearchEngine results to API SearchResponse format."""
    return SearchResponse(
        results=[convert_search_result_to_api_format(r, search_type) for r in results],
        total_hits=len(results),
        execution_time_ms=execution_time * 1000,  # Convert seconds to milliseconds
        filters_applied=filters,
        query=query,
        processing_time=execution_time,
        search_type=search_type,
        fusion_method=kwargs.get('fusion_method', 'enhanced_search_engine'),
        reranking_applied=kwargs.get('reranking_applied', False),
        cache_hit=kwargs.get('cache_hit', False)
    )


def convert_api_filters_to_search_filter(api_filters: Optional[SearchFilters]) -> 'SearchFilter':
    """Convert API SearchFilters to SearchEngine SearchFilter with enhanced tag support."""
    from vector_db.search_types import SearchFilter, TagMatchMode
    from datetime import datetime
    
    search_filter = SearchFilter()
    
    if api_filters:
        # Convert file types
        if api_filters.file_types:
            search_filter.content_types = api_filters.file_types
        
        # Convert and normalize tags with proper mode handling
        if api_filters.tags:
            search_filter.set_tags(api_filters.tags)
            
            # Set tag matching mode
            if api_filters.tag_match_mode:
                mode_map = {
                    "any": TagMatchMode.ANY,
                    "all": TagMatchMode.ALL,
                    "exact": TagMatchMode.EXACT
                }
                search_filter.tag_match_mode = mode_map.get(api_filters.tag_match_mode.lower(), TagMatchMode.ANY)
        
        # Convert exclude tags
        if api_filters.exclude_tags:
            search_filter.set_exclude_tags(api_filters.exclude_tags)
        
        # Convert date range
        if api_filters.date_range and len(api_filters.date_range) == 2:
            try:
                start_date = datetime.fromisoformat(api_filters.date_range[0])
                end_date = datetime.fromisoformat(api_filters.date_range[1])
                search_filter.date_range = (start_date, end_date)
            except (ValueError, TypeError) as e:
                # Log but don't fail - just ignore invalid dates
                import logging
                logging.getLogger(__name__).warning(f"Invalid date range format: {e}")
        
        # Convert file size range
        if api_filters.file_size_range and len(api_filters.file_size_range) == 2:
            try:
                min_size, max_size = api_filters.file_size_range
                if min_size >= 0 and max_size >= min_size:
                    search_filter.file_size_range = (min_size, max_size)
            except (ValueError, TypeError):
                pass  # Invalid size range, ignore
        
        # Convert language filter
        if api_filters.language:
            search_filter.language = api_filters.language.strip().lower()
        
        # Convert public/private filter
        if api_filters.is_public is not None:
            search_filter.is_public = api_filters.is_public
        
        # Handle custom metadata filters if needed
        if api_filters.metadata_filters:
            # Custom metadata filtering can be extended here
            pass
    
    return search_filter