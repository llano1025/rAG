"""
Qdrant client for vector storage and retrieval.
Provides high-level interface for managing vector collections and operations.
"""

import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, CollectionConfig, PointStruct, 
    Filter, FieldCondition, MatchValue, SearchRequest, UpdateStatus
)
from qdrant_client.http.exceptions import UnexpectedResponse

from config import get_settings
from utils.service_fallbacks import get_qdrant_client

logger = logging.getLogger(__name__)
settings = get_settings()

class QdrantManager:
    """
    High-level Qdrant client manager for vector operations.
    Handles connection management, collection operations, and vector CRUD.
    """
    
    def __init__(self, host: str = None, port: int = None, api_key: str = None):
        """Initialize Qdrant client with connection parameters."""
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.api_key = api_key or getattr(settings, 'QDRANT_API_KEY', None)
        
        self.client = None
        self.is_connected = False
        
        # Default collection configurations
        self.default_vector_config = {
            "size": 384,  # Default embedding dimension
            "distance": Distance.COSINE
        }
    
    async def connect(self) -> bool:
        """Establish connection to Qdrant server with fallback."""
        try:
            # Use fallback-aware client
            self.client = get_qdrant_client(
                host=self.host,
                port=self.port
            )
            
            # Test connection for real Qdrant
            if hasattr(self.client, 'get_collections'):
                self.client.get_collections()
            
            self.is_connected = True
            logger.info(f"Connected to Qdrant successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            # Fallback should already be handled by get_qdrant_client
            if self.client:
                self.is_connected = True
                return True
            return False
    
    async def disconnect(self):
        """Close connection to Qdrant server."""
        if self.client:
            try:
                self.client.close()
                self.is_connected = False
                logger.info("Disconnected from Qdrant")
            except Exception as e:
                logger.error(f"Error disconnecting from Qdrant: {e}")
    
    def ensure_connected(self):
        """Ensure client is connected, raise exception if not."""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to Qdrant. Call connect() first.")
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance_metric: Distance = Distance.COSINE,
        overwrite: bool = False
    ) -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors to store
            distance_metric: Distance metric for similarity search
            overwrite: Whether to overwrite existing collection
            
        Returns:
            True if collection was created successfully
        """
        self.ensure_connected()
        
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name in existing_names:
                if overwrite:
                    logger.info(f"Deleting existing collection: {collection_name}")
                    self.client.delete_collection(collection_name)
                else:
                    logger.info(f"Collection {collection_name} already exists")
                    return True
            
            # Create collection
            vector_config = VectorParams(
                size=vector_size,
                distance=distance_metric
            )
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_config
            )
            
            logger.info(f"Created collection: {collection_name} (size={vector_size}, metric={distance_metric})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from Qdrant."""
        self.ensure_connected()
        
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection."""
        self.ensure_connected()
        
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return None
    
    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]] = None,
        ids: List[str] = None
    ) -> List[str]:
        """
        Insert or update vectors in a collection.
        
        Args:
            collection_name: Target collection name
            vectors: List of vector embeddings
            payloads: List of metadata dictionaries for each vector
            ids: List of point IDs (auto-generated if not provided)
            
        Returns:
            List of point IDs that were upserted
        """
        self.ensure_connected()
        
        if not vectors:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        
        # Ensure payloads list has same length as vectors
        if payloads is None:
            payloads = [{}] * len(vectors)
        elif len(payloads) != len(vectors):
            raise ValueError("Payloads list must have same length as vectors list")
        
        try:
            # Create points
            points = []
            for i, (vector, payload, point_id) in enumerate(zip(vectors, payloads, ids)):
                # Add timestamp to payload
                payload_with_timestamp = payload.copy()
                payload_with_timestamp["indexed_at"] = datetime.utcnow().isoformat()
                payload_with_timestamp["point_index"] = i
                
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload_with_timestamp
                )
                points.append(point)
            
            # Upsert points
            operation_info = self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                logger.info(f"Successfully upserted {len(points)} vectors to {collection_name}")
                return ids
            else:
                logger.error(f"Upsert operation failed with status: {operation_info.status}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to upsert vectors to {collection_name}: {e}")
            return []
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = None,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a collection.
        
        Args:
            collection_name: Collection to search in
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Metadata filters to apply
            
        Returns:
            List of search results with scores and payloads
        """
        self.ensure_connected()
        
        try:
            # Build filter if provided
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        # Handle multiple values (OR condition)
                        for v in value:
                            conditions.append(
                                FieldCondition(key=key, match=MatchValue(value=v))
                            )
                    else:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                
                if conditions:
                    qdrant_filter = Filter(should=conditions)
            
            # Perform search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload or {}
                }
                results.append(result)
            
            logger.debug(f"Found {len(results)} results in {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in {collection_name}: {e}")
            return []
    
    async def get_vector(self, collection_name: str, point_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific vector by ID."""
        self.ensure_connected()
        
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )
            
            if points:
                point = points[0]
                return {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload or {}
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get vector {point_id} from {collection_name}: {e}")
            return None
    
    async def delete_vectors(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete specific vectors by their IDs."""
        self.ensure_connected()
        
        try:
            operation_info = self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids,
                wait=True
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                logger.info(f"Successfully deleted {len(point_ids)} vectors from {collection_name}")
                return True
            else:
                logger.error(f"Delete operation failed with status: {operation_info.status}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete vectors from {collection_name}: {e}")
            return False
    
    async def count_vectors(self, collection_name: str) -> int:
        """Get the number of vectors in a collection."""
        self.ensure_connected()
        
        try:
            info = self.client.get_collection(collection_name)
            return info.points_count
        except Exception as e:
            logger.error(f"Failed to count vectors in {collection_name}: {e}")
            return 0
    
    async def is_connected(self) -> bool:
        """Check if Qdrant client is connected and responsive."""
        try:
            if not self.client:
                return False
            
            # Try a simple operation to verify connectivity
            collections = self.client.get_collections()
            return True
            
        except Exception as e:
            logger.debug(f"Qdrant connectivity check failed: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant server health and return status information."""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Get basic info
            collections = self.client.get_collections()
            
            health_info = {
                "status": "healthy" if self.is_connected else "unhealthy",
                "connected": self.is_connected,
                "host": self.host,
                "port": self.port,
                "collections_count": len(collections.collections),
                "collections": [col.name for col in collections.collections],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Get detailed info for each collection
            collection_details = {}
            for collection in collections.collections:
                try:
                    info = await self.get_collection_info(collection.name)
                    collection_details[collection.name] = info
                except Exception as e:
                    collection_details[collection.name] = {"error": str(e)}
            
            health_info["collection_details"] = collection_details
            
            return health_info
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def backup_collection(self, collection_name: str, backup_path: str) -> bool:
        """Create a backup snapshot of a collection."""
        # Note: This would require Qdrant snapshot API
        # For now, this is a placeholder for future implementation
        logger.warning("Collection backup not yet implemented")
        return False
    
    async def restore_collection(self, collection_name: str, backup_path: str) -> bool:
        """Restore a collection from backup."""
        # Note: This would require Qdrant snapshot API
        # For now, this is a placeholder for future implementation
        logger.warning("Collection restore not yet implemented")
        return False


# Global Qdrant manager instance
_qdrant_manager: Optional[QdrantManager] = None

def get_qdrant_manager() -> QdrantManager:
    """Get the global Qdrant manager instance."""
    global _qdrant_manager
    if _qdrant_manager is None:
        _qdrant_manager = QdrantManager()
    return _qdrant_manager

async def init_qdrant() -> QdrantManager:
    """Initialize Qdrant connection and return manager."""
    manager = get_qdrant_manager()
    await manager.connect()
    return manager

async def cleanup_qdrant():
    """Cleanup Qdrant connections."""
    global _qdrant_manager
    if _qdrant_manager:
        await _qdrant_manager.disconnect()
        _qdrant_manager = None