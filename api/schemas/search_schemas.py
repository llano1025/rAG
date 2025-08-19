from typing import List, Optional
from pydantic import BaseModel, Field, confloat

class SearchFilters(BaseModel):
    folder_ids: Optional[List[str]] = Field(None, description="Filter by folder IDs")
    tag_ids: Optional[List[str]] = Field(None, description="Filter by tag IDs")
    file_types: Optional[List[str]] = Field(None, description="Filter by file types")
    date_range: Optional[tuple[str, str]] = Field(None, description="Filter by date range")
    metadata_filters: Optional[dict] = Field(None, description="Filter by custom metadata")

class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query text")
    filters: Optional[SearchFilters] = None
    semantic_search: bool = Field(True, description="Whether to use semantic search")
    hybrid_search: bool = Field(False, description="Whether to combine semantic and keyword search")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    similarity_threshold: Optional[float] = Field(
        None, 
        description="Minimum similarity score threshold",
        ge=0.0,
        le=1.0
    )
    # Reranker settings
    enable_reranking: bool = Field(True, description="Whether to enable reranking of search results")
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
                "score": 0.95,
                "snippet": "This is a relevant excerpt from the document...",
                "metadata": {
                    "title": "Example Document",
                    "created_at": "2024-02-08T12:00:00Z",
                    "file_type": "pdf"
                },
                "highlights": [(4, 10), (15, 22)]
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