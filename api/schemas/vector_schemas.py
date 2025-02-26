from typing import List, Optional, Union
from pydantic import BaseModel, Field
import numpy as np

class VectorMetadata(BaseModel):
    document_id: str
    chunk_index: int
    chunk_text: str
    embedding_model: str
    custom_metadata: Optional[dict] = None

class VectorEntry(BaseModel):
    id: str = Field(..., description="Unique identifier for the vector")
    vector: List[float] = Field(..., description="Vector embedding values")
    metadata: VectorMetadata

    class Config:
        arbitrary_types_allowed = True

class VectorUpsertRequest(BaseModel):
    vectors: List[VectorEntry]
    index_name: str = Field(..., description="Name of the vector index")

class VectorSearchRequest(BaseModel):
    query_vector: List[float] = Field(..., description="Query vector for similarity search")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    index_name: str = Field(..., description="Name of the vector index to search")
    metadata_filter: Optional[dict] = None
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

class VectorSearchResult(BaseModel):
    id: str
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: VectorMetadata

class VectorSearchResponse(BaseModel):
    results: List[VectorSearchResult]
    total_searched: int
    execution_time_ms: float

class IndexStats(BaseModel):
    index_name: str
    vector_count: int
    dimension: int
    metadata_schema: dict
    disk_usage_bytes: int
    index_type: str
    created_at: str
    last_modified: str

class VectorDeleteRequest(BaseModel):
    vector_ids: List[str]
    index_name: str