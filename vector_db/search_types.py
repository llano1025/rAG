"""
Shared search types and classes to avoid circular imports.
Contains base search types, filters, and result classes used across the search system.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone

class SearchType:
    """Search type constants."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    CONTEXTUAL = "contextual"
    TABLE_CONTENT = "table_content"
    TABLE_HEADERS = "table_headers"
    TABLE_CONTEXT = "table_context"
    TABLE_HYBRID = "table_hybrid"

class TableSearchType:
    """Extended search types for table-aware search."""
    TABLE_CONTENT = "table_content"  # Search within table data
    TABLE_HEADERS = "table_headers"  # Search table headers/columns
    TABLE_CONTEXT = "table_context"  # Search text around tables
    TABLE_HYBRID = "table_hybrid"    # Combined table and text search

class TagMatchMode:
    """Tag matching modes for search filtering."""
    ANY = "any"  # Document must have ANY of the specified tags (OR logic)
    ALL = "all"  # Document must have ALL of the specified tags (AND logic)
    EXACT = "exact"  # Document must have EXACTLY the specified tags (no more, no less)

class SearchFilter:
    """Search filter class for structured filtering."""

    def __init__(self):
        self.user_id: Optional[int] = None
        self.document_ids: Optional[List[int]] = None
        self.content_types: Optional[List[str]] = None
        self.date_range: Optional[Tuple[datetime, datetime]] = None
        self.file_size_range: Optional[Tuple[int, int]] = None
        self.tags: Optional[List[str]] = None
        self.tag_match_mode: str = TagMatchMode.ANY
        self.exclude_tags: Optional[List[str]] = None  # Tags to exclude (NOT logic)
        self.language: Optional[str] = None
        self.is_public: Optional[bool] = None
        self.min_score: Optional[float] = None
        self.embedding_model: Optional[str] = None  # Embedding model to use for vector operations

        # Reranker settings
        self.enable_reranking: bool = False
        self.reranker_model: Optional[str] = None
        self.rerank_score_weight: float = 0.5
        self.min_rerank_score: Optional[float] = None
        self.max_results_to_rerank: int = 100

        # MMR (Maximal Marginal Relevance) diversification settings
        self.enable_mmr: bool = False
        self.mmr_lambda: float = 0.6  # Balance between relevance (1.0) and diversity (0.0)
        self.mmr_similarity_threshold: float = 0.8  # Minimum similarity for diversity penalty
        self.mmr_max_results: Optional[int] = None  # Maximum diversified results (None = no limit)
        self.mmr_similarity_metric: str = "cosine"  # "cosine", "euclidean", "dot_product"

    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'document_ids': self.document_ids,
            'content_types': self.content_types,
            'date_range': [dt.isoformat() for dt in self.date_range] if self.date_range else None,
            'file_size_range': self.file_size_range,
            'tags': self.tags,
            'tag_match_mode': self.tag_match_mode,
            'exclude_tags': self.exclude_tags,
            'language': self.language,
            'is_public': self.is_public,
            'min_score': self.min_score,
            'embedding_model': self.embedding_model,
            'enable_reranking': self.enable_reranking,
            'reranker_model': self.reranker_model,
            'rerank_score_weight': self.rerank_score_weight,
            'min_rerank_score': self.min_rerank_score,
            'max_results_to_rerank': self.max_results_to_rerank,
            'enable_mmr': self.enable_mmr,
            'mmr_lambda': self.mmr_lambda,
            'mmr_similarity_threshold': self.mmr_similarity_threshold,
            'mmr_max_results': self.mmr_max_results,
            'mmr_similarity_metric': self.mmr_similarity_metric
        }

    def normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize tags for consistent matching (lowercase, strip whitespace)."""
        if not tags:
            return []

        normalized = []
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                normalized_tag = tag.strip().lower()
                if normalized_tag not in normalized:  # Remove duplicates
                    normalized.append(normalized_tag)

        return normalized

    def set_tags(self, tags: Optional[List[str]]):
        """Set tags with automatic normalization."""
        if tags:
            self.tags = self.normalize_tags(tags)
        else:
            self.tags = None

    def set_exclude_tags(self, exclude_tags: Optional[List[str]]):
        """Set exclude tags with automatic normalization."""
        if exclude_tags:
            self.exclude_tags = self.normalize_tags(exclude_tags)
        else:
            self.exclude_tags = None

class TableSearchFilter(SearchFilter):
    """Extended search filter with table-specific options."""

    def __init__(self, **kwargs):
        super().__init__()

        # Apply any provided kwargs to base class attributes
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Table-specific attributes
        self.table_only: bool = False
        self.table_types: List[str] = []  # e.g., ["table", "mixed"]
        self.has_cross_page_tables: Optional[bool] = None
        self.min_table_rows: Optional[int] = None
        self.min_table_columns: Optional[int] = None
        self.table_confidence_threshold: float = 0.0

class SearchResult:
    """Enhanced search result with metadata and access control."""

    def __init__(
        self,
        chunk_id: str,
        document_id: int,
        text: str,
        score: float,
        metadata: Dict[str, Any],
        document_metadata: Dict[str, Any] = None,
        highlight: str = None
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.text = text
        self.score = score
        self.metadata = metadata
        self.document_metadata = document_metadata or {}
        self.highlight = highlight
        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'text': self.text,
            'score': self.score,
            'metadata': self.metadata,
            'document_metadata': self.document_metadata,
            'highlight': self.highlight,
            'timestamp': self.timestamp.isoformat()
        }