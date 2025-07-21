"""
Unified storage manager for coordinating FAISS and Qdrant vector storage.
Provides high-level interface for vector operations with automatic synchronization.
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from .search_optimizer import SearchOptimizer
from .qdrant_client import QdrantManager, get_qdrant_manager
from database.models import VectorIndex, Document, DocumentChunk
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class VectorStorageManager:
    """
    Unified storage manager that coordinates FAISS and Qdrant operations.
    
    Architecture:
    - FAISS: High-performance similarity search with local indices
    - Qdrant: Persistent vector storage with metadata and filtering
    - PostgreSQL: Metadata and relationships
    """
    
    def __init__(self, storage_path: str = None):
        """Initialize storage manager."""
        self.storage_path = storage_path or "./vector_storage"
        self.faiss_indices: Dict[str, SearchOptimizer] = {}
        self.qdrant_manager: Optional[QdrantManager] = None
        
        # Ensure storage directory exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Collection naming convention
        self.content_collection_suffix = "_content"
        self.context_collection_suffix = "_context"
    
    async def initialize(self) -> bool:
        """Initialize all storage backends."""
        try:
            # Initialize Qdrant connection
            self.qdrant_manager = get_qdrant_manager()
            await self.qdrant_manager.connect()
            
            logger.info("Vector storage manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector storage manager: {e}")
            return False
    
    def _get_collection_names(self, index_name: str) -> Tuple[str, str]:
        """Get Qdrant collection names for content and context vectors."""
        content_collection = f"{index_name}{self.content_collection_suffix}"
        context_collection = f"{index_name}{self.context_collection_suffix}"
        return content_collection, context_collection
    
    def _get_faiss_paths(self, index_name: str) -> Tuple[str, str]:
        """Get FAISS index file paths for content and context vectors."""
        content_path = os.path.join(self.storage_path, f"{index_name}_content.faiss")
        context_path = os.path.join(self.storage_path, f"{index_name}_context.faiss")
        return content_path, context_path
    
    async def create_index(
        self,
        index_name: str,
        embedding_dimension: int = 384,
        faiss_index_type: str = "HNSW",
        user_id: int = None,
        document_id: int = None,
        db: Session = None
    ) -> bool:
        """
        Create a new vector index with both FAISS and Qdrant backends.
        
        Args:
            index_name: Unique name for the index
            embedding_dimension: Dimension of the embedding vectors
            faiss_index_type: Type of FAISS index (FLAT, HNSW, IVF, etc.)
            user_id: ID of the user creating the index
            document_id: Optional document ID for document-specific indices
            db: Database session for metadata storage
            
        Returns:
            True if index was created successfully
        """
        try:
            # Create FAISS search optimizers
            content_optimizer = SearchOptimizer(
                embedding_dim=embedding_dimension,
                index_type=faiss_index_type,
                similarity_metric="cosine"
            )
            
            context_optimizer = SearchOptimizer(
                embedding_dim=embedding_dimension,
                index_type=faiss_index_type,
                similarity_metric="cosine"
            )
            
            # Store FAISS optimizers
            self.faiss_indices[f"{index_name}_content"] = content_optimizer
            self.faiss_indices[f"{index_name}_context"] = context_optimizer
            
            # Create Qdrant collections
            content_collection, context_collection = self._get_collection_names(index_name)
            
            await self.qdrant_manager.create_collection(
                collection_name=content_collection,
                vector_size=embedding_dimension
            )
            
            await self.qdrant_manager.create_collection(
                collection_name=context_collection,
                vector_size=embedding_dimension
            )
            
            # Save metadata to database if provided
            if db and user_id:
                vector_index = VectorIndex(
                    index_name=index_name,
                    index_type="combined",
                    user_id=user_id,
                    document_id=document_id,
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Default
                    embedding_dimension=embedding_dimension,
                    faiss_index_type=faiss_index_type,
                    qdrant_collection_name=content_collection,
                    build_status="ready"
                )
                
                db.add(vector_index)
                db.commit()
                
                logger.info(f"Created vector index metadata in database: {index_name}")
            
            logger.info(f"Created vector index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector index {index_name}: {e}")
            return False
    
    async def add_vectors(
        self,
        index_name: str,
        content_vectors: List[List[float]],
        context_vectors: List[List[float]],
        metadata_list: List[Dict[str, Any]],
        chunk_ids: List[str] = None
    ) -> List[str]:
        """
        Add vectors to both FAISS and Qdrant storage.
        
        Args:
            index_name: Name of the target index
            content_vectors: List of content embedding vectors
            context_vectors: List of context embedding vectors  
            metadata_list: List of metadata dictionaries for each vector
            chunk_ids: Optional list of chunk IDs
            
        Returns:
            List of point IDs that were added
        """
        try:
            if len(content_vectors) != len(context_vectors) or len(content_vectors) != len(metadata_list):
                raise ValueError("All input lists must have the same length")
            
            # Generate chunk IDs if not provided
            if chunk_ids is None:
                chunk_ids = [f"{index_name}_{i}_{datetime.utcnow().timestamp()}" 
                           for i in range(len(content_vectors))]
            
            # Add to FAISS indices
            content_optimizer = self.faiss_indices.get(f"{index_name}_content")
            context_optimizer = self.faiss_indices.get(f"{index_name}_context")
            
            if not content_optimizer or not context_optimizer:
                raise ValueError(f"Index {index_name} not found. Create index first.")
            
            # Convert to numpy arrays
            content_vectors_np = np.array(content_vectors, dtype=np.float32)
            context_vectors_np = np.array(context_vectors, dtype=np.float32)
            
            # Add to FAISS
            content_optimizer.add_vectors(content_vectors_np, metadata_list)
            context_optimizer.add_vectors(context_vectors_np, metadata_list)
            
            # Add to Qdrant collections
            content_collection, context_collection = self._get_collection_names(index_name)
            
            # Add enhanced metadata for Qdrant
            enhanced_metadata = []
            for i, metadata in enumerate(metadata_list):
                enhanced = metadata.copy()
                enhanced.update({
                    "chunk_id": chunk_ids[i],
                    "index_name": index_name,
                    "vector_type": "content",
                    "added_at": datetime.utcnow().isoformat()
                })
                enhanced_metadata.append(enhanced)
            
            # Upsert to Qdrant
            content_ids = await self.qdrant_manager.upsert_vectors(
                collection_name=content_collection,
                vectors=content_vectors,
                payloads=enhanced_metadata,
                ids=chunk_ids
            )
            
            # Update metadata for context vectors
            for metadata in enhanced_metadata:
                metadata["vector_type"] = "context"
            
            context_ids = await self.qdrant_manager.upsert_vectors(
                collection_name=context_collection,
                vectors=context_vectors,
                payloads=enhanced_metadata,
                ids=chunk_ids
            )
            
            logger.info(f"Added {len(content_vectors)} vectors to index {index_name}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to add vectors to index {index_name}: {e}")
            return []
    
    async def search_vectors(
        self,
        index_name: str,
        query_vector: List[float],
        vector_type: str = "content",
        limit: int = 10,
        score_threshold: float = None,
        metadata_filters: Dict[str, Any] = None,
        use_faiss: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using FAISS or Qdrant.
        
        Args:
            index_name: Name of the index to search
            query_vector: Query vector for similarity search
            vector_type: Type of vectors to search ("content" or "context")
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            metadata_filters: Filters to apply to metadata
            use_faiss: Whether to use FAISS (faster) or Qdrant (with filters)
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            if use_faiss and not metadata_filters:
                # Use FAISS for fast search without filters
                optimizer_key = f"{index_name}_{vector_type}"
                optimizer = self.faiss_indices.get(optimizer_key)
                
                if not optimizer:
                    raise ValueError(f"FAISS index {optimizer_key} not found")
                
                query_vector_np = np.array([query_vector], dtype=np.float32)
                results = optimizer.search(
                    query_vectors=query_vector_np,
                    k=limit,
                    score_threshold=score_threshold
                )
                
                # Format results
                formatted_results = []
                if results and len(results) > 0:
                    for i, (distance, metadata) in enumerate(zip(results[0]["distances"], results[0]["metadata"])):
                        formatted_results.append({
                            "score": 1.0 - distance,  # Convert distance to similarity
                            "metadata": metadata,
                            "source": "faiss"
                        })
                
                return formatted_results
                
            else:
                # Use Qdrant for search with filters
                collection_name = (f"{index_name}{self.content_collection_suffix}" 
                                 if vector_type == "content" 
                                 else f"{index_name}{self.context_collection_suffix}")
                
                results = await self.qdrant_manager.search_vectors(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    filters=metadata_filters
                )
                
                # Format results
                for result in results:
                    result["source"] = "qdrant"
                    result["metadata"] = result.get("payload", {})
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search vectors in index {index_name}: {e}")
            return []
    
    async def contextual_search(
        self,
        index_name: str,
        query_vector: List[float],
        limit: int = 10,
        content_weight: float = 0.7,
        context_weight: float = 0.3,
        score_threshold: float = None,
        metadata_filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform contextual search combining content and context vectors.
        
        Args:
            index_name: Name of the index to search
            query_vector: Query vector for similarity search
            limit: Maximum number of results
            content_weight: Weight for content similarity scores
            context_weight: Weight for context similarity scores
            score_threshold: Minimum combined score threshold
            metadata_filters: Filters to apply to metadata
            
        Returns:
            List of search results with combined scores
        """
        try:
            # Search both content and context vectors
            content_results = await self.search_vectors(
                index_name=index_name,
                query_vector=query_vector,
                vector_type="content",
                limit=limit * 2,  # Get more results for better combination
                metadata_filters=metadata_filters,
                use_faiss=not metadata_filters  # Use FAISS if no filters
            )
            
            context_results = await self.search_vectors(
                index_name=index_name,
                query_vector=query_vector,
                vector_type="context",
                limit=limit * 2,
                metadata_filters=metadata_filters,
                use_faiss=not metadata_filters
            )
            
            # Combine results by chunk_id
            combined_scores = {}
            
            # Process content results
            for result in content_results:
                chunk_id = result["metadata"].get("chunk_id")
                if chunk_id:
                    combined_scores[chunk_id] = {
                        "content_score": result["score"],
                        "context_score": 0.0,
                        "metadata": result["metadata"],
                        "source": result.get("source", "unknown")
                    }
            
            # Process context results
            for result in context_results:
                chunk_id = result["metadata"].get("chunk_id")
                if chunk_id:
                    if chunk_id in combined_scores:
                        combined_scores[chunk_id]["context_score"] = result["score"]
                    else:
                        combined_scores[chunk_id] = {
                            "content_score": 0.0,
                            "context_score": result["score"],
                            "metadata": result["metadata"],
                            "source": result.get("source", "unknown")
                        }
            
            # Calculate combined scores
            final_results = []
            for chunk_id, scores in combined_scores.items():
                combined_score = (
                    scores["content_score"] * content_weight +
                    scores["context_score"] * context_weight
                )
                
                if score_threshold is None or combined_score >= score_threshold:
                    final_results.append({
                        "chunk_id": chunk_id,
                        "score": combined_score,
                        "content_score": scores["content_score"],
                        "context_score": scores["context_score"],
                        "metadata": scores["metadata"],
                        "source": scores["source"]
                    })
            
            # Sort by combined score and return top results
            final_results.sort(key=lambda x: x["score"], reverse=True)
            return final_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to perform contextual search in index {index_name}: {e}")
            return []
    
    async def save_index(self, index_name: str) -> bool:
        """Save FAISS indices to disk."""
        try:
            content_path, context_path = self._get_faiss_paths(index_name)
            
            content_optimizer = self.faiss_indices.get(f"{index_name}_content")
            context_optimizer = self.faiss_indices.get(f"{index_name}_context")
            
            if content_optimizer:
                content_optimizer.save_index(content_path)
                logger.info(f"Saved content index to {content_path}")
            
            if context_optimizer:
                context_optimizer.save_index(context_path)
                logger.info(f"Saved context index to {context_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index {index_name}: {e}")
            return False
    
    async def load_index(self, index_name: str, embedding_dimension: int = 384) -> bool:
        """Load FAISS indices from disk."""
        try:
            content_path, context_path = self._get_faiss_paths(index_name)
            
            if os.path.exists(content_path):
                content_optimizer = SearchOptimizer(embedding_dim=embedding_dimension)
                content_optimizer.load_index(content_path)
                self.faiss_indices[f"{index_name}_content"] = content_optimizer
                logger.info(f"Loaded content index from {content_path}")
            
            if os.path.exists(context_path):
                context_optimizer = SearchOptimizer(embedding_dim=embedding_dimension)
                context_optimizer.load_index(context_path)
                self.faiss_indices[f"{index_name}_context"] = context_optimizer
                logger.info(f"Loaded context index from {context_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index {index_name}: {e}")
            return False
    
    async def delete_index(self, index_name: str, db: Session = None) -> bool:
        """Delete an index from all storage backends."""
        try:
            # Remove from FAISS
            content_key = f"{index_name}_content"
            context_key = f"{index_name}_context"
            
            if content_key in self.faiss_indices:
                del self.faiss_indices[content_key]
            
            if context_key in self.faiss_indices:
                del self.faiss_indices[context_key]
            
            # Delete FAISS files
            content_path, context_path = self._get_faiss_paths(index_name)
            for path in [content_path, context_path]:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Deleted FAISS index file: {path}")
            
            # Delete Qdrant collections
            content_collection, context_collection = self._get_collection_names(index_name)
            await self.qdrant_manager.delete_collection(content_collection)
            await self.qdrant_manager.delete_collection(context_collection)
            
            # Delete from database
            if db:
                vector_index = db.query(VectorIndex).filter(
                    VectorIndex.index_name == index_name
                ).first()
                if vector_index:
                    db.delete(vector_index)
                    db.commit()
            
            logger.info(f"Deleted vector index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            return False
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for a vector index."""
        try:
            stats = {
                "index_name": index_name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # FAISS stats
            content_optimizer = self.faiss_indices.get(f"{index_name}_content")
            context_optimizer = self.faiss_indices.get(f"{index_name}_context")
            
            stats["faiss"] = {
                "content_vectors": content_optimizer.get_index_size() if content_optimizer else 0,
                "context_vectors": context_optimizer.get_index_size() if context_optimizer else 0
            }
            
            # Qdrant stats
            content_collection, context_collection = self._get_collection_names(index_name)
            
            content_info = await self.qdrant_manager.get_collection_info(content_collection)
            context_info = await self.qdrant_manager.get_collection_info(context_collection)
            
            stats["qdrant"] = {
                "content_collection": content_info,
                "context_collection": context_info
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for index {index_name}: {e}")
            return {"error": str(e)}
    
    async def get_faiss_stats(self) -> Dict[str, Any]:
        """Get FAISS indices statistics for health monitoring."""
        try:
            total_indices = len(self.faiss_indices)
            active_indices = 0
            index_details = {}
            
            for index_name, optimizer in self.faiss_indices.items():
                try:
                    if hasattr(optimizer, 'index') and optimizer.index is not None:
                        active_indices += 1
                        index_details[index_name] = {
                            'vectors_count': optimizer.index.ntotal if hasattr(optimizer.index, 'ntotal') else 0,
                            'dimension': optimizer.config.dimension if hasattr(optimizer, 'config') else 'unknown',
                            'is_trained': optimizer.index.is_trained if hasattr(optimizer.index, 'is_trained') else True
                        }
                    else:
                        index_details[index_name] = {'status': 'inactive'}
                except Exception as e:
                    index_details[index_name] = {'error': str(e)}
            
            return {
                'total_indices': total_indices,
                'active_indices': active_indices,
                'health_ratio': active_indices / total_indices if total_indices > 0 else 1.0,
                'indices': index_details
            }
            
        except Exception as e:
            logger.error(f"Failed to get FAISS stats: {e}")
            return {
                'total_indices': 0,
                'active_indices': 0,
                'health_ratio': 0.0,
                'error': str(e)
            }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for health monitoring."""
        try:
            stats = {
                'faiss_indices_count': len(self.faiss_indices),
                'storage_path': self.storage_path,
                'storage_path_exists': Path(self.storage_path).exists()
            }
            
            # Add Qdrant connection status
            if self.qdrant_manager:
                stats['qdrant_connected'] = await self.qdrant_manager.is_connected()
                stats['qdrant_collections'] = len(await self.qdrant_manager.list_collections())
            else:
                stats['qdrant_connected'] = False
                stats['qdrant_collections'] = 0
            
            # Check storage directory size
            try:
                storage_size = sum(
                    f.stat().st_size for f in Path(self.storage_path).rglob('*') if f.is_file()
                )
                stats['storage_size_bytes'] = storage_size
                stats['storage_size_mb'] = round(storage_size / (1024 * 1024), 2)
            except Exception:
                stats['storage_size_bytes'] = 0
                stats['storage_size_mb'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {'error': str(e)}

    async def cleanup(self):
        """Cleanup storage manager and close connections."""
        if self.qdrant_manager:
            await self.qdrant_manager.disconnect()
        
        self.faiss_indices.clear()
        logger.info("Vector storage manager cleaned up")


# Global storage manager instance
_storage_manager: Optional[VectorStorageManager] = None

def get_storage_manager() -> VectorStorageManager:
    """Get the global vector storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = VectorStorageManager()
    return _storage_manager

async def init_storage_manager() -> VectorStorageManager:
    """Initialize storage manager and return instance."""
    manager = get_storage_manager()
    await manager.initialize()
    return manager