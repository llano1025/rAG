"""
Qdrant client for vector storage and retrieval.
Provides high-level interface for managing vector collections and operations.
"""

import asyncio
import logging
import uuid
import json
import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path

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
        
        # Store original chunk IDs for payload reference
        original_chunk_ids = ids[:] if ids else None
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        else:
            # Convert string chunk IDs to UUIDs for Qdrant compatibility
            converted_ids = []
            for chunk_id in ids:
                if isinstance(chunk_id, str) and not chunk_id.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                    # Convert string chunk ID to UUID5 (deterministic)
                    # This ensures the same chunk_id always gets the same UUID
                    namespace = uuid.UUID('12345678-1234-5678-1234-567812345678')  # Fixed namespace
                    converted_id = str(uuid.uuid5(namespace, chunk_id))
                    converted_ids.append(converted_id)
                else:
                    # Already a valid ID format
                    converted_ids.append(str(chunk_id))
            ids = converted_ids
        
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
                payload_with_timestamp["indexed_at"] = datetime.now(timezone.utc).isoformat()
                payload_with_timestamp["point_index"] = i
                
                # Preserve original chunk_id if it was converted
                if original_chunk_ids and i < len(original_chunk_ids):
                    if "chunk_id" not in payload_with_timestamp:
                        payload_with_timestamp["chunk_id"] = original_chunk_ids[i]
                
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

            logger.info(f"{search_result}")
            
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
                "timestamp": datetime.now(timezone.utc).isoformat()
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
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def backup_collection(self, collection_name: str, backup_path: str, include_vectors: bool = True) -> Dict[str, Any]:
        """
        Create a comprehensive backup of a collection.
        
        Args:
            collection_name: Name of the collection to backup
            backup_path: Path where backup will be stored
            include_vectors: Whether to include vector data (set False for metadata-only backup)
            
        Returns:
            Backup metadata dictionary with success status and details
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            if not self.client:
                await self.connect()
            
            # Ensure backup directory exists
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate backup filename with timestamp
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{collection_name}_backup_{timestamp}"
            backup_file_path = backup_dir / f"{backup_filename}.json"
            
            logger.info(f"Starting backup of collection '{collection_name}' to {backup_file_path}")
            
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            if not collection_info:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            # Initialize backup data structure
            backup_data = {
                "metadata": {
                    "collection_name": collection_name,
                    "backup_timestamp": start_time.isoformat(),
                    "backup_version": "1.0",
                    "qdrant_client_version": "1.6.0+",  # Current version
                    "include_vectors": include_vectors,
                    "total_points": collection_info.points_count,
                    "vectors_count": collection_info.vectors_count
                },
                "collection_config": {
                    "vectors": {
                        "size": collection_info.config.params.vectors.size,
                        "distance": str(collection_info.config.params.vectors.distance),
                        "hnsw_config": collection_info.config.params.vectors.hnsw_config,
                        "quantization_config": collection_info.config.params.vectors.quantization_config
                    },
                    "optimizer_config": collection_info.config.optimizer_config,
                    "wal_config": collection_info.config.wal_config
                },
                "points": []
            }
            
            # Retrieve all points with pagination
            points_exported = 0
            offset = None
            batch_size = 1000
            
            while True:
                try:
                    # Scroll through points
                    points_batch, next_offset = self.client.scroll(
                        collection_name=collection_name,
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=include_vectors
                    )
                    
                    if not points_batch:
                        break
                    
                    # Process batch
                    for point in points_batch:
                        point_data = {
                            "id": str(point.id),
                            "payload": point.payload if point.payload else {}
                        }
                        
                        # Include vector data if requested
                        if include_vectors and point.vector:
                            if isinstance(point.vector, dict):
                                # Named vectors
                                point_data["vector"] = {name: vec for name, vec in point.vector.items()}
                            else:
                                # Single vector
                                point_data["vector"] = point.vector
                        
                        backup_data["points"].append(point_data)
                    
                    points_exported += len(points_batch)
                    offset = next_offset
                    
                    # Log progress
                    if points_exported % 10000 == 0:
                        logger.info(f"Exported {points_exported} points...")
                    
                    # Break if no more points
                    if next_offset is None:
                        break
                        
                except Exception as e:
                    logger.error(f"Error during point export batch: {e}")
                    break
            
            # Update final metadata
            backup_data["metadata"]["exported_points"] = points_exported
            backup_data["metadata"]["backup_completed_at"] = datetime.now(timezone.utc).isoformat()
            backup_data["metadata"]["backup_duration_seconds"] = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Write backup to file
            with open(backup_file_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # Verify backup file
            backup_size_mb = backup_file_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"Backup completed: {points_exported} points exported to {backup_file_path}")
            logger.info(f"Backup file size: {backup_size_mb:.2f} MB")
            
            return {
                "success": True,
                "backup_file": str(backup_file_path),
                "collection_name": collection_name,
                "points_exported": points_exported,
                "backup_size_mb": backup_size_mb,
                "duration_seconds": backup_data["metadata"]["backup_duration_seconds"],
                "include_vectors": include_vectors,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            error_msg = f"Backup failed for collection '{collection_name}': {e}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "error": str(e),
                "collection_name": collection_name,
                "timestamp": start_time.isoformat(),
                "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds()
            }
    
    async def restore_collection(self, backup_file_path: str, target_collection_name: Optional[str] = None, 
                                recreate_collection: bool = False) -> Dict[str, Any]:
        """
        Restore a collection from backup.
        
        Args:
            backup_file_path: Path to the backup file
            target_collection_name: Target collection name (defaults to original name)
            recreate_collection: Whether to recreate the collection if it exists
            
        Returns:
            Restore operation results
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            if not self.client:
                await self.connect()
            
            # Load backup data
            backup_path = Path(backup_file_path)
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file_path}")
            
            logger.info(f"Starting restore from backup: {backup_file_path}")
            
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Validate backup format
            if "metadata" not in backup_data or "collection_config" not in backup_data:
                raise ValueError("Invalid backup format: missing required sections")
            
            # Determine collection name
            original_name = backup_data["metadata"]["collection_name"]
            collection_name = target_collection_name or original_name
            
            logger.info(f"Restoring to collection '{collection_name}' from backup of '{original_name}'")
            
            # Check if collection exists
            try:
                existing_info = self.client.get_collection(collection_name)
                if existing_info and not recreate_collection:
                    raise ValueError(f"Collection '{collection_name}' already exists. Use recreate_collection=True to overwrite.")
                
                if recreate_collection:
                    logger.info(f"Deleting existing collection '{collection_name}'")
                    self.client.delete_collection(collection_name)
                    
            except Exception as e:
                # Collection doesn't exist or couldn't check - proceed with creation
                logger.debug(f"Collection check result: {e}")
            
            # Create collection with backed-up configuration
            vector_config = backup_data["collection_config"]["vectors"]
            
            # Map distance string back to enum
            distance_map = {
                "Distance.COSINE": Distance.COSINE,
                "Distance.EUCLID": Distance.EUCLID,
                "Distance.DOT": Distance.DOT
            }
            distance = distance_map.get(vector_config["distance"], Distance.COSINE)
            
            vectors_config = VectorParams(
                size=vector_config["size"],
                distance=distance,
                hnsw_config=vector_config.get("hnsw_config"),
                quantization_config=vector_config.get("quantization_config")
            )
            
            logger.info(f"Creating collection '{collection_name}' with vector dimension {vector_config['size']}")
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config
            )
            
            # Restore points in batches
            points_data = backup_data.get("points", [])
            total_points = len(points_data)
            batch_size = 1000
            restored_points = 0
            
            logger.info(f"Restoring {total_points} points in batches of {batch_size}")
            
            for i in range(0, total_points, batch_size):
                batch = points_data[i:i + batch_size]
                
                # Convert to PointStruct objects
                points_to_upsert = []
                for point_data in batch:
                    point = PointStruct(
                        id=point_data["id"],
                        vector=point_data.get("vector"),
                        payload=point_data.get("payload", {})
                    )
                    points_to_upsert.append(point)
                
                # Upsert batch
                self.client.upsert(
                    collection_name=collection_name,
                    points=points_to_upsert
                )
                
                restored_points += len(batch)
                
                # Log progress
                if restored_points % 10000 == 0 or restored_points == total_points:
                    logger.info(f"Restored {restored_points}/{total_points} points...")
            
            # Verify restoration
            final_info = self.client.get_collection(collection_name)
            
            duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            logger.info(f"Restore completed: {restored_points} points restored to '{collection_name}'")
            logger.info(f"Final collection stats: {final_info.points_count} points, {final_info.vectors_count} vectors")
            
            return {
                "success": True,
                "collection_name": collection_name,
                "original_collection": original_name,
                "points_restored": restored_points,
                "backup_timestamp": backup_data["metadata"]["backup_timestamp"],
                "restore_timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": duration_seconds,
                "final_points_count": final_info.points_count,
                "final_vectors_count": final_info.vectors_count
            }
            
        except Exception as e:
            error_msg = f"Restore failed from '{backup_file_path}': {e}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "error": str(e),
                "backup_file": backup_file_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds()
            }
    
    async def list_backups(self, backup_directory: str) -> List[Dict[str, Any]]:
        """
        List available backup files in a directory.
        
        Args:
            backup_directory: Directory to scan for backups
            
        Returns:
            List of backup file information
        """
        try:
            backup_dir = Path(backup_directory)
            if not backup_dir.exists():
                return []
            
            backups = []
            for backup_file in backup_dir.glob("*_backup_*.json"):
                try:
                    # Get file stats
                    file_stats = backup_file.stat()
                    
                    # Try to read backup metadata
                    with open(backup_file, 'r') as f:
                        backup_data = json.load(f)
                    
                    metadata = backup_data.get("metadata", {})
                    
                    backups.append({
                        "filename": backup_file.name,
                        "full_path": str(backup_file),
                        "collection_name": metadata.get("collection_name", "unknown"),
                        "backup_timestamp": metadata.get("backup_timestamp", "unknown"),
                        "total_points": metadata.get("total_points", 0),
                        "exported_points": metadata.get("exported_points", 0),
                        "include_vectors": metadata.get("include_vectors", True),
                        "file_size_mb": file_stats.st_size / (1024 * 1024),
                        "backup_version": metadata.get("backup_version", "unknown")
                    })
                    
                except Exception as e:
                    logger.warning(f"Could not read backup metadata from {backup_file}: {e}")
                    # Add basic file info even if metadata can't be read
                    backups.append({
                        "filename": backup_file.name,
                        "full_path": str(backup_file),
                        "collection_name": "unknown",
                        "backup_timestamp": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        "file_size_mb": file_stats.st_size / (1024 * 1024),
                        "error": f"Metadata read failed: {e}"
                    })
            
            # Sort by backup timestamp (newest first)
            backups.sort(key=lambda x: x.get("backup_timestamp", ""), reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups in {backup_directory}: {e}")
            return []

    async def list_collections(self) -> List[str]:
        """List all collections in Qdrant."""
        try:
            if not self.client:
                logger.error("Qdrant client not connected")
                return []
            
            collections_response = self.client.get_collections()
            collections = [col.name for col in collections_response.collections]
            logger.debug(f"Found {len(collections)} collections in Qdrant: {collections}")
            return collections
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []


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