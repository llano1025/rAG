from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class SearchType(str, Enum):
    """Unified search type definitions for all search operations."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"
    TEXT = "text"  # Alias for keyword search
    # Table-specific search types
    TABLE_CONTENT = "table_content"
    TABLE_HEADERS = "table_headers"
    TABLE_CONTEXT = "table_context"
    TABLE_HYBRID = "table_hybrid"


class TagMatchMode(str, Enum):
    """Tag matching modes for search filtering."""
    ANY = "any"      # Document must have ANY of the specified tags (OR logic)
    ALL = "all"      # Document must have ALL of the specified tags (AND logic)
    EXACT = "exact"  # Document must have EXACTLY the specified tags (no more, no less)


class TableSearchType(str, Enum):
    """Extended search types for table-aware search."""
    TABLE_CONTENT = "table_content"  # Search within table data
    TABLE_HEADERS = "table_headers"  # Search table headers/columns
    TABLE_CONTEXT = "table_context"  # Search text around tables
    TABLE_HYBRID = "table_hybrid"    # Combined table and text search


class SearchFilters(BaseModel):
    """Unified search filters for both API and internal use."""
    # Content settings
    folder_ids: Optional[List[str]] = Field(None, description="Filter by folder IDs")
    tags: Optional[List[str]] = Field(None, description="Filter by tags (case-insensitive)")
    tag_match_mode: TagMatchMode = Field(
        TagMatchMode.ANY,
        description="Tag matching mode: 'any' (OR logic), 'all' (AND logic), or 'exact' (exact match)"
    )
    exclude_tags: Optional[List[str]] = Field(None, description="Tags to exclude from results")
    file_types: Optional[List[str]] = Field(None, description="Filter by file types (MIME types)")
    date_range: Optional[tuple[str, str]] = Field(None, description="Filter by date range (ISO format)")
    file_size_range: Optional[tuple[int, int]] = Field(None, description="Filter by file size range (bytes)")
    language: Optional[str] = Field(None, description="Filter by document language")
    is_public: Optional[bool] = Field(None, description="Filter by public/private status")
    metadata_filters: Optional[dict] = Field(None, description="Filter by custom metadata")

    # Internal engine fields (optional for API use)
    user_id: Optional[int] = Field(None, description="User ID for access control (internal use)")
    document_ids: Optional[List[int]] = Field(None, description="Specific document IDs to search (internal use)")
    content_types: Optional[List[str]] = Field(None, description="Content types filter (alias for file_types)")
    min_score: Optional[float] = Field(None, description="Minimum similarity score threshold")
    embedding_model: Optional[str] = Field(None, description="Embedding model to use for vector operations")
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

    def set_tags(self, tags: List[str]) -> None:
        """Set tags with normalization."""
        if tags:
            self.tags = [tag.strip().lower() for tag in tags if tag.strip()]

    def set_exclude_tags(self, exclude_tags: List[str]) -> None:
        """Set exclude tags with normalization."""
        if exclude_tags:
            self.exclude_tags = [tag.strip().lower() for tag in exclude_tags if tag.strip()]

    def to_dict(self) -> Dict:
        """Convert filters to dictionary for internal compatibility."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                if isinstance(field_value, Enum):
                    result[field_name] = field_value.value
                elif hasattr(field_value, 'isoformat'):
                    result[field_name] = field_value.isoformat()
                else:
                    result[field_name] = field_value
        return result

    @property
    def max_results_to_rerank(self) -> int:
        """Compatibility property for internal use."""
        return 100  # Default value from original SearchFilter

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


class TableSearchFilter(SearchFilters):
    """Extended search filter with table-specific options."""
    # Table-specific attributes
    table_only: bool = Field(False, description="Search only in table content")
    table_types: List[str] = Field(default_factory=list, description="Table types filter (e.g., ['table', 'mixed'])")
    has_cross_page_tables: Optional[bool] = Field(None, description="Filter for cross-page tables")
    min_table_rows: Optional[int] = Field(None, description="Minimum number of table rows")
    min_table_columns: Optional[int] = Field(None, description="Minimum number of table columns")
    table_confidence_threshold: float = Field(0.0, description="Table detection confidence threshold", ge=0.0, le=1.0)


class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query text")
    filters: Optional[SearchFilters] = None
    search_type: SearchType = Field(
        SearchType.CONTEXTUAL,
        description="Type of search to perform"
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
    Unified search result model for both API and internal use.

    Attributes:
        document_id: Unique identifier of the document (flexible str/int)
        chunk_id: Unique identifier of the text chunk
        text: Full text content of the chunk
        snippet: Short preview text for quick scanning
        content_snippet: Alias for snippet (for API compatibility)
        filename: Name of the document file (optional, for API use)
        score: Relevance score between 0 and 1
        metadata: Chunk-level metadata
        document_metadata: Document-level metadata
        highlight: Highlighted text snippet (optional)
        timestamp: When this result was created
    """
    document_id: Union[str, int] = Field(..., description="Unique identifier of the document")
    chunk_id: str = Field(..., description="Unique identifier of the text chunk")
    text: str = Field(..., description="Full text content of the chunk")
    snippet: Optional[str] = Field(None, description="Short preview text for quick result scanning")
    score: float = Field(..., description="Relevance score")
    metadata: Dict = Field(
        default_factory=dict,
        description="Chunk-level metadata"
    )
    document_metadata: Dict = Field(
        default_factory=dict,
        description="Document-level metadata"
    )

    # Optional fields for specific use cases
    filename: Optional[str] = Field(None, description="Name of the document file")
    highlight: Optional[str] = Field(None, description="Highlighted text snippet")
    timestamp: Optional[datetime] = Field(None, description="When this result was created")

    # API compatibility - content_snippet as computed field
    @property
    def content_snippet(self) -> str:
        """Alias for snippet field for API compatibility."""
        if self.snippet is not None:
            return self.snippet
        # Generate snippet from text if not provided
        return generate_intelligent_snippet(self.text, self.highlight)

    def to_dict(self) -> Dict:
        """Convert result to dictionary for internal compatibility."""
        result = {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'text': self.text,
            'score': self.score,
            'metadata': self.metadata,
            'document_metadata': self.document_metadata,
        }
        if self.snippet is not None:
            result['snippet'] = self.snippet
        if self.highlight is not None:
            result['highlight'] = self.highlight
        if self.timestamp is not None:
            result['timestamp'] = self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'SearchResult':
        """Create SearchResult from dictionary for internal compatibility."""
        # Handle timestamp conversion
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                timestamp = None

        return cls(
            document_id=data['document_id'],
            chunk_id=data['chunk_id'],
            text=data['text'],
            snippet=data.get('snippet'),  # Can be None, will auto-generate in property
            score=data['score'],
            metadata=data.get('metadata', {}),
            document_metadata=data.get('document_metadata', {}),
            highlight=data.get('highlight'),
            timestamp=timestamp
        )

    class Config:
        """Pydantic model configuration"""
        json_schema_extra = {
            "example": {
                "document_id": "doc123",
                "chunk_id": "chunk_456",
                "text": "This is the full text content of the document chunk that contains all the detailed information...",
                "snippet": "This is a relevant excerpt from the document...",
                "filename": "example.pdf",
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
    search_type: Optional[SearchType] = Field(None, description="Type of search performed")
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


# Helper functions for search result processing
def generate_intelligent_snippet(text: str, highlight: Optional[str] = None, max_length: int = 250) -> str:
    """
    Generate an intelligent snippet from full text.

    Args:
        text: Full text content
        highlight: Optional highlighted text snippet
        max_length: Maximum length of snippet

    Returns:
        Intelligent snippet with word boundaries
    """
    if not text:
        return ""

    # If we have highlight text and it's not too long, use it
    if highlight and len(highlight.strip()) <= max_length:
        return highlight.strip()

    # If text is already short enough, return as-is
    if len(text) <= max_length:
        return text.strip()

    # Find a good cutoff point at word boundary
    snippet = text[:max_length].strip()

    # If the cutoff splits a word, back up to the last complete word
    if len(text) > max_length and text[max_length] not in [' ', '\n', '\t', '.', ',', ';', '!', '?']:
        last_space = snippet.rfind(' ')
        if last_space > max_length // 2:  # Only if we don't lose too much content
            snippet = snippet[:last_space]

    # Add ellipsis if we truncated
    if len(snippet) < len(text.strip()):
        snippet = snippet.strip() + "..."

    return snippet


# Helper function to convert SearchResult from EnhancedSearchEngine to API format
def convert_search_result_to_api_format(result, search_type: SearchType = SearchType.SEMANTIC) -> SearchResult:
    """Convert SearchEngine result to API SearchResult format.

    Args:
        result: Search result object from search engine
        search_type: Type of search performed (unused but kept for API compatibility)
    """
    # Safe attribute access with fallbacks
    document_metadata = getattr(result, 'document_metadata', {}) or {}
    metadata = getattr(result, 'metadata', {}) or {}
    highlight = getattr(result, 'highlight', None)
    text = getattr(result, 'text', '')

    # Generate intelligent snippet
    snippet = generate_intelligent_snippet(text, highlight)

    return SearchResult(
        document_id=str(getattr(result, 'document_id', '')),
        chunk_id=str(getattr(result, 'chunk_id', '')),
        text=str(text),
        snippet=snippet,
        filename=document_metadata.get("filename", "Unknown") if isinstance(document_metadata, dict) else "Unknown",
        score=getattr(result, 'score', 0.0),
        metadata=metadata,
        document_metadata=document_metadata,
        highlight=highlight,
    )


def convert_search_response_to_api_format(results: List, query: str,
                                        execution_time: float = 0.0,
                                        search_type: SearchType = SearchType.SEMANTIC,
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
    
    search_filter = SearchFilters()
    
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

    search_filter = SearchFilters()

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

    # Handle embedding model
    if settings.get("embedding_model"):
        search_filter.embedding_model = settings["embedding_model"]

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