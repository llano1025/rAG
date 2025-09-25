from typing import List, Optional
from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    # Content settings
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
    # MMR (Maximal Marginal Relevance) diversification settings
    enable_mmr: bool = Field(False, description="Whether to enable MMR diversification of search results")
    mmr_lambda: float = Field(
        0.6,
        description="MMR lambda parameter: balance between relevance (1.0) and diversity (0.0)",
        ge=0.0,
        le=1.0
    )
    mmr_similarity_threshold: float = Field(
        0.8,
        description="Minimum similarity threshold for MMR diversity penalty",
        ge=0.0,
        le=1.0
    )
    mmr_max_results: Optional[int] = Field(
        None,
        description="Maximum number of results after MMR diversification (None = no limit)",
        ge=1,
        le=100
    )
    mmr_similarity_metric: str = Field(
        "cosine",
        description="Similarity metric for MMR diversification",
        pattern="^(cosine|euclidean|dot_product)$"
    )

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
    search_type: Optional[str] = Field(
        "contextual",
        description="Type of search to perform: 'semantic', 'contextual', or 'text'",
        pattern="^(semantic|contextual|text)$"
    )
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    similarity_threshold: Optional[float] = Field(
        None, 
        description="Minimum similarity score threshold",
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
from typing import List, Dict


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


def convert_api_filters_to_search_filter(api_filters: Optional[SearchFilters]):
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

        # Convert reranker settings
        if api_filters.enable_reranking is not None:
            search_filter.enable_reranking = api_filters.enable_reranking

        if api_filters.reranker_model:
            search_filter.reranker_model = api_filters.reranker_model

        if api_filters.rerank_score_weight is not None:
            search_filter.rerank_score_weight = api_filters.rerank_score_weight

        if api_filters.min_rerank_score is not None:
            search_filter.min_rerank_score = api_filters.min_rerank_score

        # Convert MMR settings
        if api_filters.enable_mmr is not None:
            search_filter.enable_mmr = api_filters.enable_mmr

        if api_filters.mmr_lambda is not None:
            search_filter.mmr_lambda = api_filters.mmr_lambda

        if api_filters.mmr_similarity_threshold is not None:
            search_filter.mmr_similarity_threshold = api_filters.mmr_similarity_threshold

        if api_filters.mmr_max_results is not None:
            search_filter.mmr_max_results = api_filters.mmr_max_results

        if api_filters.mmr_similarity_metric:
            search_filter.mmr_similarity_metric = api_filters.mmr_similarity_metric

    return search_filter


def convert_dict_to_search_filter(settings: dict):
    """Convert chat settings dict to SearchFilter object."""
    from vector_db.search_types import SearchFilter, TagMatchMode

    search_filter = SearchFilter()

    # Basic search settings
    if settings.get("tags"):
        search_filter.set_tags(settings["tags"])

        # Set tag matching mode
        if settings.get("tag_match_mode"):
            mode_map = {
                "any": TagMatchMode.ANY,
                "all": TagMatchMode.ALL,
                "exact": TagMatchMode.EXACT
            }
            search_filter.tag_match_mode = mode_map.get(settings["tag_match_mode"].lower(), TagMatchMode.ANY)

    if settings.get("exclude_tags"):
        search_filter.set_exclude_tags(settings["exclude_tags"])

    # Handle both file_type (chat) and file_types (search) fields
    file_types = settings.get("file_type") or settings.get("file_types")
    if file_types:
        search_filter.content_types = file_types

    # Handle both language (single) and languages (array) fields
    language = settings.get("language")
    languages = settings.get("languages")
    if language:
        search_filter.language = language
    elif languages and len(languages) > 0:
        search_filter.language = languages[0]  # Use first language for now

    if settings.get("is_public") is not None:
        search_filter.is_public = settings["is_public"]

    if settings.get("file_size_range"):
        search_filter.file_size_range = tuple(settings["file_size_range"])

    if settings.get("date_range"):
        dr = settings["date_range"]
        if isinstance(dr, dict) and "start" in dr and "end" in dr:
            search_filter.date_range = (dr["start"], dr["end"])

    # Handle min_score and similarity_threshold
    min_score = settings.get("min_score") or settings.get("similarity_threshold")
    if min_score is not None:
        search_filter.min_score = min_score

    # Reranker settings
    if settings.get("enable_reranking") is not None:
        search_filter.enable_reranking = settings["enable_reranking"]

    if settings.get("reranker_model"):
        search_filter.reranker_model = settings["reranker_model"]

    if settings.get("rerank_score_weight") is not None:
        search_filter.rerank_score_weight = settings["rerank_score_weight"]

    if settings.get("min_rerank_score") is not None:
        search_filter.min_rerank_score = settings["min_rerank_score"]

    # MMR settings
    if settings.get("enable_mmr") is not None:
        search_filter.enable_mmr = settings["enable_mmr"]

    if settings.get("mmr_lambda") is not None:
        search_filter.mmr_lambda = settings["mmr_lambda"]

    if settings.get("mmr_similarity_threshold") is not None:
        search_filter.mmr_similarity_threshold = settings["mmr_similarity_threshold"]

    if settings.get("mmr_max_results") is not None:
        search_filter.mmr_max_results = settings["mmr_max_results"]

    if settings.get("mmr_similarity_metric"):
        search_filter.mmr_similarity_metric = settings["mmr_similarity_metric"]

    return search_filter