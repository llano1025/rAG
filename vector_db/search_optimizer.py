"""
Qdrant-based vector search optimizer with advanced search capabilities.
Replaces the previous dual FAISS/Qdrant architecture with Qdrant-only implementation.
"""

from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum
import time
import json

# Qdrant client imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, SearchParams, Range, MatchValue
    QDRANT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Qdrant client not available: {e}. Install required dependency: qdrant-client")
    QdrantClient = None
    Distance = None
    VectorParams = None
    PointStruct = None
    Filter = None
    FieldCondition = None
    SearchParams = None
    Range = None
    MatchValue = None
    QDRANT_AVAILABLE = False

from .chunking import Chunk

class MetricType(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"

class IndexType(Enum):
    """Index types for backward compatibility (Qdrant handles optimization internally)"""
    HNSW = "hnsw"  # Qdrant's default
    FLAT = "flat"  # For exact search if needed

@dataclass
class SearchConfig:
    """Configuration for Qdrant-based search operations."""
    dimension: int
    metric: MetricType = MetricType.COSINE
    collection_name: str = "default"
    # Qdrant-specific parameters
    hnsw_config: Dict[str, Any] = None
    quantization_config: Dict[str, Any] = None
    
    def __post_init__(self):
        # Set default HNSW config if not provided
        if self.hnsw_config is None:
            self.hnsw_config = {
                "m": 16,  # Number of edges per node
                "ef_construct": 100,  # Size of the dynamic candidate list
                "full_scan_threshold": 10000,  # Switch to exact search for small collections
                "max_indexing_threads": 0,  # Use all available cores
                "on_disk": False  # Keep in memory for better performance
            }

class SearchError(Exception):
    """Raised when search operation fails."""
    pass

class QdrantSearchOptimizer:
    """
    Qdrant-based vector search optimizer with advanced filtering and scoring capabilities.
    
    This replaces the previous dual FAISS/Qdrant architecture with a single, robust
    Qdrant-only implementation that eliminates metadata synchronization issues.
    """
    
    def __init__(self, config: SearchConfig, qdrant_client: QdrantClient = None):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required for search optimization functionality. "
                "Install it with: pip install qdrant-client"
            )
        
        self.config = config
        self.client = qdrant_client
        self.collection_name = config.collection_name
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for search operations."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _get_qdrant_distance(self) -> Distance:
        """Convert MetricType to Qdrant Distance."""
        if self.config.metric == MetricType.COSINE:
            return Distance.COSINE
        elif self.config.metric == MetricType.EUCLIDEAN:
            return Distance.EUCLID
        elif self.config.metric == MetricType.DOT_PRODUCT:
            return Distance.DOT
        else:
            return Distance.COSINE  # Default fallback

    async def create_collection(self) -> bool:
        """Create Qdrant collection with optimized settings."""
        try:
            if not self.client:
                raise ValueError("Qdrant client not initialized")
            
            # Check if collection already exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if self.collection_name in existing_names:
                self.logger.info(f"Collection {self.collection_name} already exists")
                return True
            
            # Create collection with vector configuration
            vector_config = VectorParams(
                size=self.config.dimension,
                distance=self._get_qdrant_distance(),
                hnsw_config=self.config.hnsw_config,
                quantization_config=self.config.quantization_config
            )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_config
            )
            
            self.logger.info(f"Created Qdrant collection: {self.collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create collection {self.collection_name}: {e}")
            raise SearchError(f"Collection creation failed: {e}")


    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        min_score: float = 0.0,
        filters: Dict[str, Any] = None,
        search_params: Dict[str, Any] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar vectors in Qdrant collection.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            min_score: Minimum similarity score threshold
            filters: Metadata filters to apply
            search_params: Additional search parameters for Qdrant
            
        Returns:
            List of (Chunk, score) tuples sorted by similarity
        """
        try:
            start_time = time.time()
            
            if not self.client:
                raise ValueError("Qdrant client not initialized")
            
            # Build filter conditions
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Handle list values (IN operator)
                        for v in value:
                            conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
                    elif isinstance(value, dict) and 'range' in value:
                        # Handle range filters
                        range_config = value['range']
                        conditions.append(
                            FieldCondition(
                                key=key,
                                range=Range(
                                    gte=range_config.get('gte'),
                                    lte=range_config.get('lte'),
                                    gt=range_config.get('gt'),
                                    lt=range_config.get('lt')
                                )
                            )
                        )
                    else:
                        # Handle exact match
                        conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Configure search parameters
            if search_params is None:
                search_params = {}
            
            qdrant_search_params = SearchParams(
                hnsw_ef=search_params.get('hnsw_ef', 128),  # Size of the dynamic candidate list
                exact=search_params.get('exact', False)     # Use exact search if needed
            )
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=min_score,
                search_params=qdrant_search_params,
                with_payload=True,
                with_vectors=False  # We don't need vectors in results
            )
            
            # Convert results to Chunk objects
            results = []
            for result in search_results:
                try:
                    payload = result.payload or {}
                    
                    # Debug payload contents to understand what's stored
                    self.logger.debug(f"DEBUG: Qdrant result point_id={result.id}")
                    self.logger.debug(f"DEBUG: Payload keys: {list(payload.keys())}")
                    
                    # Get original chunk_id from payload metadata (prioritize over UUID)
                    original_chunk_id = payload.get('chunk_id')
                    if not original_chunk_id:
                        self.logger.warning(
                            f"Missing original chunk_id in Qdrant payload for point {result.id}. "
                            f"This indicates metadata corruption during upload. Using UUID as fallback."
                        )
                        original_chunk_id = str(result.id)  # UUID fallback
                    elif original_chunk_id == str(result.id):
                        self.logger.warning(f"chunk_id equals UUID for point {result.id} - this will cause database lookup issues")
                    else:
                        self.logger.debug(f"Using original chunk_id: {original_chunk_id} for point {result.id}")
                    
                    # Create Chunk object from payload
                    chunk = Chunk(
                        document_id=payload.get('document_id', 0),
                        chunk_id=original_chunk_id,
                        start_idx=payload.get('start_char', 0),
                        end_idx=payload.get('end_char', 0),
                        text=payload.get('text', ''),
                        metadata=payload,
                        embedding=None  # Don't include embedding vector in memory
                    )
                    
                    results.append((chunk, float(result.score)))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process search result {result.id}: {e}")
                    continue
            
            search_time = time.time() - start_time
            self.logger.debug(
                f"Qdrant search completed in {search_time:.4f} seconds, "
                f"found {len(results)} results"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search operation failed: {str(e)}")
            raise SearchError(f"Search failed: {str(e)}")

    async def batch_search(
        self,
        query_vectors: List[List[float]],
        limit: int = 10,
        min_score: float = 0.0,
        filters: Dict[str, Any] = None,
        search_params: Dict[str, Any] = None
    ) -> List[List[Tuple[Chunk, float]]]:
        """
        Perform batch search for multiple query vectors.
        
        Args:
            query_vectors: List of query embedding vectors
            limit: Maximum number of results per query
            min_score: Minimum similarity score threshold
            filters: Metadata filters to apply
            search_params: Additional search parameters
            
        Returns:
            List of search results for each query
        """
        try:
            start_time = time.time()
            
            # For now, perform individual searches
            # Qdrant supports batch search, but this approach gives us more control
            batch_results = []
            for query_vector in query_vectors:
                results = await self.search(
                    query_vector=query_vector,
                    limit=limit,
                    min_score=min_score,
                    filters=filters,
                    search_params=search_params
                )
                batch_results.append(results)
            
            batch_time = time.time() - start_time
            self.logger.debug(
                f"Batch search completed in {batch_time:.4f} seconds, "
                f"processed {len(query_vectors)} queries"
            )
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch search failed: {str(e)}")
            raise SearchError(f"Batch search failed: {str(e)}")

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection."""
        try:
            if not self.client:
                raise ValueError("Qdrant client not initialized")
            
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance),
                    "hnsw_config": collection_info.config.params.vectors.hnsw_config,
                    "optimizer_config": collection_info.config.optimizer_config,
                },
                "status": str(collection_info.status)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    async def delete_collection(self) -> bool:
        """Delete the Qdrant collection."""
        try:
            if not self.client:
                raise ValueError("Qdrant client not initialized")
            
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return False

    async def get_vector_count(self) -> int:
        """Get the number of vectors in the collection."""
        try:
            info = await self.get_collection_info()
            return info.get("vectors_count", 0)
        except Exception:
            return 0

    def validate_health(self) -> Dict[str, Any]:
        """Validate search optimizer health and return diagnostic information."""
        try:
            if not self.client:
                return {
                    "status": "error",
                    "message": "Qdrant client not initialized",
                    "collection_name": self.collection_name
                }
            
            # Try to get collection info as health check
            try:
                collection_info = self.client.get_collection(self.collection_name)
                return {
                    "status": "healthy",
                    "collection_name": self.collection_name,
                    "vectors_count": collection_info.vectors_count,
                    "indexed_vectors_count": collection_info.indexed_vectors_count,
                    "collection_status": str(collection_info.status)
                }
            except Exception as e:
                return {
                    "status": "degraded",
                    "message": f"Collection access failed: {e}",
                    "collection_name": self.collection_name
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {e}",
                "collection_name": self.collection_name
            }

# Backward compatibility - alias to new class name
SearchOptimizer = QdrantSearchOptimizer