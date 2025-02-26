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

from pydantic import BaseModel, Field
from typing import List, Tuple, Dict
from decimal import Decimal

class SearchResult(BaseModel):
    """
    Represents a single search result from the RAG system.
    
    Attributes:
        document_id: Unique identifier of the document
        score: Relevance score between 0 and 1
        snippet: Relevant text excerpt from the document
        metadata: Additional document metadata
        highlights: List of tuples containing start and end positions of query matches
    """
    document_id: str = Field(..., description="Unique identifier of the document")
    score: Decimal = Field(
        ...,
        ge=Decimal('0.0'),
        le=Decimal('1.0'),
        description="Relevance score between 0 and 1"
    )
    snippet: str = Field(
        ...,
        min_length=1,
        description="Relevant text excerpt from the document"
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional document metadata"
    )
    highlights: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="List of (start, end) positions of query matches"
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
    filters_applied: SearchFilters
    query_vector_id: Optional[str] = Field(None, description="ID of generated query vector")