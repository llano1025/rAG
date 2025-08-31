"""
Simplified Qdrant-only vector storage manager.
Replaces the previous dual FAISS/Qdrant architecture with a single, robust Qdrant implementation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from .qdrant_client import QdrantManager, get_qdrant_manager
from .search_optimizer import QdrantSearchOptimizer, SearchConfig, MetricType
from database.models import VectorIndex, DocumentChunk
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class VectorStorageManager:
    """
    Simplified Qdrant-only vector storage manager.
    
    Architecture:
    - Qdrant: Primary vector storage with integrated metadata and filtering
    - PostgreSQL: Index metadata and relationships
    
    This eliminates the complexity of the dual FAISS/Qdrant system while
    maintaining all functionality through Qdrant's comprehensive feature set.
    """
    
    def __init__(self):
        """Initialize Qdrant-only storage manager."""
        self.qdrant_manager: Optional[QdrantManager] = None
        self.search_optimizers: Dict[str, QdrantSearchOptimizer] = {}
        
        # Collection naming convention
        self.content_collection_suffix = "_content"
        self.context_collection_suffix = "_context"
        
        # Optimization state tracking
        self._is_initialized = False
        self._collections_state = {}  # Track collection state for smart refresh
    
    async def initialize(self) -> bool:
        """Initialize Qdrant connection and discover existing collections."""
        if self._is_initialized:
            logger.debug("Storage manager already initialized, skipping")
            return True
            
        try:
            # Initialize Qdrant connection
            self.qdrant_manager = get_qdrant_manager()
            await self.qdrant_manager.connect()
            logger.info("Qdrant connection established successfully")
            
            # Initialize search optimizers for existing collections
            await self._discover_and_init_collections()
            
            self._is_initialized = True
            logger.info("Vector storage manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector storage manager: {e}")
            return False
    
    async def _discover_and_init_collections(self):
        """Discover existing active Qdrant collections and initialize search optimizers."""
        try:
            if not self.qdrant_manager:
                return
            
            # Only initialize search optimizers for active collections
            collections = await self.qdrant_manager.list_active_collections()
            logger.info(f"Discovered {len(collections)} active Qdrant collections")
            logger.info(f"Looking for collections with suffixes: '{self.content_collection_suffix}' and '{self.context_collection_suffix}'")
            
            # Group collections by index name
            index_collections = {}
            for collection_name in collections:
                logger.debug(f"Processing collection: {collection_name}")
                if collection_name.endswith(self.content_collection_suffix):
                    index_name = collection_name[:-len(self.content_collection_suffix)]
                    logger.debug(f"Found content collection for index '{index_name}': {collection_name}")
                    if index_name not in index_collections:
                        index_collections[index_name] = {}
                    index_collections[index_name]['content'] = collection_name
                elif collection_name.endswith(self.context_collection_suffix):
                    index_name = collection_name[:-len(self.context_collection_suffix)]
                    logger.debug(f"Found context collection for index '{index_name}': {collection_name}")
                    if index_name not in index_collections:
                        index_collections[index_name] = {}
                    index_collections[index_name]['context'] = collection_name
                else:
                    logger.debug(f"Collection '{collection_name}' doesn't match expected suffixes, skipping")
            
            logger.info(f"Grouped active collections into {len(index_collections)} indices: {list(index_collections.keys())}")
            
            # Initialize search optimizers for discovered active indices
            for index_name, collections_dict in index_collections.items():
                try:
                    logger.info(f"Initializing search optimizer for active index '{index_name}' with collections: {collections_dict}")
                    await self._init_search_optimizer_for_index(index_name, collections_dict)
                    logger.info(f"Successfully initialized search optimizer for active index: {index_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize search optimizer for {index_name}: {e}")
            
            logger.info(f"Search optimizer initialization complete. Active optimizers: {len(self.search_optimizers)}")
            logger.info(f"Available optimizer keys: {list(self.search_optimizers.keys())}")
                    
        except Exception as e:
            logger.error(f"Failed to discover collections: {e}")
    
    async def _init_search_optimizer_for_index(self, index_name: str, collections_dict: Dict[str, str]):
        """Initialize search optimizers for a specific index."""
        try:
            # Get collection info to determine vector dimension
            content_collection = collections_dict.get('content')
            if not content_collection:
                return
            
            collection_info = await self.qdrant_manager.get_collection_info(content_collection)
            if not collection_info:
                return
            
            vector_size = collection_info.get('config', {}).get('vector_size', 384)
            
            # Create search optimizers for content and context
            if 'content' in collections_dict:
                content_config = SearchConfig(
                    dimension=vector_size,
                    metric=MetricType.COSINE,
                    collection_name=collections_dict['content']
                )
                content_optimizer = QdrantSearchOptimizer(
                    config=content_config,
                    qdrant_client=self.qdrant_manager.client
                )
                self.search_optimizers[f"{index_name}_content"] = content_optimizer
            
            if 'context' in collections_dict:
                context_config = SearchConfig(
                    dimension=vector_size,
                    metric=MetricType.COSINE,
                    collection_name=collections_dict['context']
                )
                context_optimizer = QdrantSearchOptimizer(
                    config=context_config,
                    qdrant_client=self.qdrant_manager.client
                )
                self.search_optimizers[f"{index_name}_context"] = context_optimizer
                
        except Exception as e:
            logger.error(f"Failed to initialize search optimizer for {index_name}: {e}")
    
    def _get_collection_names(self, index_name: str) -> Tuple[str, str]:
        """Get Qdrant collection names for content and context vectors."""
        content_collection = f"{index_name}{self.content_collection_suffix}"
        context_collection = f"{index_name}{self.context_collection_suffix}"
        return content_collection, context_collection
    
    async def create_index(
        self,
        index_name: str,
        embedding_dimension: int = 384,
        user_id: int = None,
        document_id: int = None,
        db: Session = None
    ) -> bool:
        """
        Create a new vector index with Qdrant collections.
        
        Args:
            index_name: Unique name for the index
            embedding_dimension: Dimension of the embedding vectors
            user_id: ID of the user creating the index
            document_id: Optional document ID for document-specific indices
            db: Database session for metadata storage
            
        Returns:
            True if index was created successfully
        """
        try:
            if not self.qdrant_manager:
                await self.initialize()
            
            content_collection, context_collection = self._get_collection_names(index_name)
            
            # Create Qdrant collections
            await self.qdrant_manager.create_collection(
                collection_name=content_collection,
                vector_size=embedding_dimension
            )
            
            await self.qdrant_manager.create_collection(
                collection_name=context_collection,
                vector_size=embedding_dimension
            )
            
            # Create search optimizers
            content_config = SearchConfig(
                dimension=embedding_dimension,
                metric=MetricType.COSINE,
                collection_name=content_collection
            )
            content_optimizer = QdrantSearchOptimizer(
                config=content_config,
                qdrant_client=self.qdrant_manager.client
            )
            self.search_optimizers[f"{index_name}_content"] = content_optimizer
            
            context_config = SearchConfig(
                dimension=embedding_dimension,
                metric=MetricType.COSINE,
                collection_name=context_collection
            )
            context_optimizer = QdrantSearchOptimizer(
                config=context_config,
                qdrant_client=self.qdrant_manager.client
            )
            self.search_optimizers[f"{index_name}_context"] = context_optimizer
            
            # Save metadata to database if provided
            if db and user_id:
                vector_index = VectorIndex(
                    index_name=index_name,
                    index_type="qdrant",
                    user_id=user_id,
                    document_id=document_id,
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Default
                    embedding_dimension=embedding_dimension,
                    similarity_metric="cosine",
                    # Set FAISS fields to default values for Qdrant-only mode
                    faiss_index_type="none",
                    faiss_index_path=None,
                    faiss_index_params=None,
                    # Qdrant-specific fields
                    qdrant_collection_name=content_collection,
                    build_status="ready"
                )
                
                db.add(vector_index)
                db.commit()
                
                logger.info(f"Created vector index metadata in database: {index_name}")
            
            logger.info(f"Created Qdrant-only vector index: {index_name}")
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
        chunk_ids: List[str] = None,
        validate_chunks: bool = True
    ) -> List[str]:
        """
        Add vectors to Qdrant collections with validation to prevent inconsistencies.
        
        Args:
            index_name: Name of the target index
            content_vectors: List of content embedding vectors
            context_vectors: List of context embedding vectors  
            metadata_list: List of metadata dictionaries for each vector
            chunk_ids: Optional list of chunk IDs
            validate_chunks: Whether to validate chunk existence in database
            
        Returns:
            List of point IDs that were added
        """
        try:
            if not self.qdrant_manager:
                await self.initialize()
            
            if len(content_vectors) != len(context_vectors) or len(content_vectors) != len(metadata_list):
                raise ValueError("All input lists must have the same length")
            
            # Generate chunk IDs if not provided
            if chunk_ids is None:
                import uuid
                chunk_ids = [str(uuid.uuid4()) for _ in range(len(content_vectors))]
            
            # Validate chunk consistency if requested
            valid_indices = []
            if validate_chunks and chunk_ids:
                valid_indices = await self._validate_chunk_consistency(
                    chunk_ids, metadata_list, index_name
                )
                
                if len(valid_indices) < len(chunk_ids):
                    logger.warning(
                        f"Filtered out {len(chunk_ids) - len(valid_indices)} invalid chunks "
                        f"from vector addition to prevent inconsistencies"
                    )
                
                # Filter vectors and metadata to only include valid chunks
                if valid_indices:
                    content_vectors = [content_vectors[i] for i in valid_indices]
                    context_vectors = [context_vectors[i] for i in valid_indices]
                    metadata_list = [metadata_list[i] for i in valid_indices]
                    chunk_ids = [chunk_ids[i] for i in valid_indices]
                else:
                    logger.error("No valid chunks found after validation - aborting vector addition")
                    return []
            
            # Get collection names for direct Qdrant operations
            content_collection, context_collection = self._get_collection_names(index_name)
            
            # Add enhanced metadata for content vectors
            content_metadata = []
            for i, metadata in enumerate(metadata_list):
                enhanced_metadata = metadata.copy()
                # chunk_ids now contains original string chunk_ids from document_version_manager
                enhanced_metadata.update({
                    'chunk_id': chunk_ids[i],  # Use original string chunk_id for database lookup
                    'index_name': index_name,
                    'vector_type': 'content',
                    'added_at': datetime.now(timezone.utc).isoformat()
                })
                content_metadata.append(enhanced_metadata)
                
                # Debug: Log what chunk_id is being stored
                logger.debug(f"DEBUG: Storing content metadata - chunk_id: {chunk_ids[i]}")
            
            # Add content vectors directly to Qdrant
            content_ids = await self.qdrant_manager.upsert_vectors(
                collection_name=content_collection,
                vectors=content_vectors,
                payloads=content_metadata,
                ids=chunk_ids  # QdrantManager handles ID conversion
            )
            
            # Add enhanced metadata for context vectors
            context_metadata = []
            for i, metadata in enumerate(metadata_list):
                enhanced_metadata = metadata.copy()
                # chunk_ids now contains original string chunk_ids from document_version_manager
                enhanced_metadata.update({
                    'chunk_id': chunk_ids[i],  # Use original string chunk_id for database lookup
                    'index_name': index_name,
                    'vector_type': 'context',
                    'added_at': datetime.now(timezone.utc).isoformat()
                })
                context_metadata.append(enhanced_metadata)
            
            # Add context vectors directly to Qdrant
            context_ids = await self.qdrant_manager.upsert_vectors(
                collection_name=context_collection,
                vectors=context_vectors,
                payloads=context_metadata,
                ids=chunk_ids  # QdrantManager handles ID conversion
            )
            
            logger.info(f"Added {len(content_vectors)} validated vectors to index {index_name}")
            # Return the actual IDs that were added to Qdrant (converted UUIDs)
            return content_ids if content_ids else context_ids if context_ids else chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to add vectors to index {index_name}: {e}")
            return []
    
    async def _validate_chunk_consistency(
        self, 
        chunk_ids: List[str], 
        metadata_list: List[Dict[str, Any]], 
        index_name: str
    ) -> List[int]:
        """
        Validate that chunks exist in database and have content before adding vectors.
        
        Args:
            chunk_ids: List of chunk IDs to validate
            metadata_list: Metadata for each chunk
            index_name: Name of the target index
            
        Returns:
            List of valid indices (positions in original lists)
        """
        try:
            from database.connection import get_db
            db = next(get_db())
            
            try:
                valid_indices = []
                
                for i, (chunk_id, metadata) in enumerate(zip(chunk_ids, metadata_list)):
                    try:
                        # Get original string chunk_id from metadata (UUIDs won't be in database)
                        original_chunk_id = metadata.get('chunk_id')
                        if not original_chunk_id:
                            logger.debug(f"Skipping chunk at index {i}: no original chunk_id in metadata")
                            continue
                        
                        # Check if chunk exists and has content using original chunk ID
                        chunk = db.query(DocumentChunk).filter(
                            DocumentChunk.chunk_id == original_chunk_id
                        ).first()
                        
                        if chunk and chunk.text and chunk.text.strip():
                            valid_indices.append(i)
                        else:
                            reason = "not found" if not chunk else "empty text"
                            logger.debug(f"Skipping chunk {original_chunk_id} (UUID: {chunk_id}): {reason}")
                    
                    except Exception as e:
                        logger.warning(f"Error validating chunk {original_chunk_id} (UUID: {chunk_id}): {e}")
                        continue
                
                logger.debug(f"Validated {len(valid_indices)} out of {len(chunk_ids)} chunks")
                return valid_indices
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to validate chunk consistency: {e}")
            # Return all indices as valid if validation fails - better to have some inconsistency
            # than to fail completely
            return list(range(len(chunk_ids)))
    
    async def search_vectors(
        self,
        index_name: str,
        query_vector: List[float],
        vector_type: str = "content",
        limit: int = 10,
        score_threshold: float = None,
        metadata_filters: Dict[str, Any] = None,
        use_faiss: bool = None  # Ignored - kept for backward compatibility
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using Qdrant.
        
        Args:
            index_name: Name of the index to search
            query_vector: Query vector for similarity search
            vector_type: Type of vectors to search ("content" or "context")
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            metadata_filters: Filters to apply to metadata
            use_faiss: Ignored (backward compatibility)
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            optimizer_key = f"{index_name}_{vector_type}"
            optimizer = self.search_optimizers.get(optimizer_key)
            
            if not optimizer:
                available = list(self.search_optimizers.keys())
                raise ValueError(f"Search optimizer '{optimizer_key}' not found. Available: {available}")
            
            # Use Qdrant search
            results = await optimizer.search(
                query_vector=query_vector,
                limit=limit,
                min_score=score_threshold or 0.0,
                filters=metadata_filters
            )
            
            # Format results for backward compatibility
            formatted_results = []
            for chunk, score in results:
                # Get content from chunk
                content = chunk.text if hasattr(chunk, 'text') and chunk.text else ""
                
                # If no content in chunk, try to fetch from database
                if not content and hasattr(chunk, 'chunk_id') and chunk.chunk_id:
                    content = await self._get_chunk_content(chunk.chunk_id)
                
                # Create metadata dict
                metadata = {
                    "chunk_id": getattr(chunk, 'chunk_id', 'unknown'),
                    "document_id": getattr(chunk, 'document_id', None),
                    "start_char": getattr(chunk, 'start_idx', None),
                    "end_char": getattr(chunk, 'end_idx', None)
                }
                
                # Add chunk metadata if available
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    metadata.update(chunk.metadata)
                
                formatted_results.append({
                    "score": float(score),
                    "content": content,
                    "metadata": metadata,
                    "source": "qdrant"
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in index {index_name}: {e}")
            return []
    
    async def _get_chunk_content(self, chunk_id: str) -> str:
        """
        Get chunk text content from database by chunk_id with enhanced error handling.
        
        This method is called when vector search results don't contain text content,
        serving as a fallback to retrieve content from the database.
        """
        try:
            from database.connection import get_db
            db = next(get_db())
            
            try:
                chunk = db.query(DocumentChunk).filter(DocumentChunk.chunk_id == chunk_id).first()
                
                if not chunk:
                    # Chunk doesn't exist in database - this indicates data inconsistency
                    logger.warning(
                        f"Chunk not found in database: {chunk_id}. "
                        f"This may indicate orphaned vector data. "
                        f"Run data integrity audit to identify and fix inconsistencies."
                    )
                    return f"[Chunk {chunk_id} not found in database - data inconsistency detected]"
                
                if not chunk.text or chunk.text.strip() == "":
                    # Chunk exists but has no text content
                    logger.warning(
                        f"Empty text content for chunk_id: {chunk_id} "
                        f"(document_id: {chunk.document_id}, chunk_index: {chunk.chunk_index}). "
                        f"Consider running data cleanup to remove empty chunks."
                    )
                    # Try to provide more context about the chunk
                    context_info = f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}"
                    return f"[Empty content for chunk {context_info}]"
                
                # Successfully found content
                logger.debug(f"Retrieved content for chunk_id: {chunk_id} ({len(chunk.text)} chars)")
                return chunk.text
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Database error while fetching chunk_id {chunk_id}: {e}")
            return f"[Database error for chunk {chunk_id}: {str(e)[:100]}...]"
    
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
                metadata_filters=metadata_filters
            )
            
            context_results = await self.search_vectors(
                index_name=index_name,
                query_vector=query_vector,
                vector_type="context",
                limit=limit * 2,
                metadata_filters=metadata_filters
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
                        "result": result
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
                            "result": result
                        }
            
            # Calculate combined scores
            final_results = []
            for chunk_id, scores in combined_scores.items():
                combined_score = (
                    scores["content_score"] * content_weight +
                    scores["context_score"] * context_weight
                )
                
                if score_threshold is None or combined_score >= score_threshold:
                    result = scores["result"].copy()
                    result["score"] = combined_score
                    result["content_score"] = scores["content_score"]
                    result["context_score"] = scores["context_score"]
                    final_results.append(result)
            
            # Sort by combined score and return top results
            final_results.sort(key=lambda x: x["score"], reverse=True)
            return final_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to perform contextual search in index {index_name}: {e}")
            return []
    
    async def delete_index(self, index_name: str, db: Session = None) -> bool:
        """Delete an index from Qdrant and database."""
        try:
            # Remove from search optimizers
            content_key = f"{index_name}_content"
            context_key = f"{index_name}_context"
            
            if content_key in self.search_optimizers:
                del self.search_optimizers[content_key]
            
            if context_key in self.search_optimizers:
                del self.search_optimizers[context_key]
            
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
    
    async def soft_delete_index(self, index_name: str, db: Session = None) -> bool:
        """Soft delete an index by disabling vectors but keeping data."""
        try:
            # Disable both content and context collections in Qdrant
            content_collection, context_collection = self._get_collection_names(index_name)
            
            content_disabled = await self.qdrant_manager.disable_vector_collection(content_collection)
            context_disabled = await self.qdrant_manager.disable_vector_collection(context_collection)
            
            if not (content_disabled and context_disabled):
                logger.warning(f"Partial failure soft deleting index {index_name}")
                return False
            
            # Update database vector index status if db is provided
            if db:
                vector_index = db.query(VectorIndex).filter(
                    VectorIndex.index_name == index_name
                ).first()
                if vector_index:
                    vector_index.is_active = False
                    vector_index.build_status = "soft_deleted"
                    vector_index.updated_at = datetime.now(timezone.utc)
                    db.commit()
            
            # Keep search optimizers but mark them as inactive (don't remove them)
            content_key = f"{index_name}_content"
            context_key = f"{index_name}_context"
            
            if content_key in self.search_optimizers:
                # Set a flag to indicate this optimizer is disabled
                self.search_optimizers[content_key]._disabled = True
            
            if context_key in self.search_optimizers:
                self.search_optimizers[context_key]._disabled = True
            
            logger.info(f"Soft deleted vector index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to soft delete index {index_name}: {e}")
            return False
    
    async def restore_index(self, index_name: str, db: Session = None) -> bool:
        """Restore a soft-deleted index by re-enabling vectors."""
        try:
            # Re-enable both content and context collections in Qdrant
            content_collection, context_collection = self._get_collection_names(index_name)
            
            content_enabled = await self.qdrant_manager.enable_vector_collection(content_collection)
            context_enabled = await self.qdrant_manager.enable_vector_collection(context_collection)
            
            if not (content_enabled and context_enabled):
                logger.warning(f"Partial failure restoring index {index_name}")
                return False
            
            # Update database vector index status if db is provided
            if db:
                vector_index = db.query(VectorIndex).filter(
                    VectorIndex.index_name == index_name
                ).first()
                if vector_index:
                    vector_index.is_active = True
                    vector_index.build_status = "ready"
                    vector_index.updated_at = datetime.now(timezone.utc)
                    db.commit()
            
            # Re-enable search optimizers
            content_key = f"{index_name}_content"
            context_key = f"{index_name}_context"
            
            if content_key in self.search_optimizers:
                if hasattr(self.search_optimizers[content_key], '_disabled'):
                    delattr(self.search_optimizers[content_key], '_disabled')
            
            if context_key in self.search_optimizers:
                if hasattr(self.search_optimizers[context_key], '_disabled'):
                    delattr(self.search_optimizers[context_key], '_disabled')
            
            logger.info(f"Restored vector index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore index {index_name}: {e}")
            return False
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for a vector index."""
        try:
            stats = {
                "index_name": index_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "qdrant_only"
            }
            
            # Get Qdrant collection stats
            content_collection, context_collection = self._get_collection_names(index_name)
            
            content_info = await self.qdrant_manager.get_collection_info(content_collection)
            context_info = await self.qdrant_manager.get_collection_info(context_collection)
            
            stats["collections"] = {
                "content": content_info,
                "context": context_info
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for index {index_name}: {e}")
            return {"error": str(e)}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for health monitoring."""
        try:
            stats = {
                'storage_type': 'qdrant_only',
                'search_optimizers_count': len(self.search_optimizers),
            }
            
            # Add Qdrant connection status
            if self.qdrant_manager:
                stats['qdrant_connected'] = await self.qdrant_manager.is_connected()
                stats['qdrant_collections'] = len(await self.qdrant_manager.list_collections())
            else:
                stats['qdrant_connected'] = False
                stats['qdrant_collections'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {'error': str(e)}

    async def add_collection(self, index_name: str, embedding_dimension: int) -> bool:
        """Add a new collection to the storage manager incrementally."""
        try:
            if not self._is_initialized:
                await self.initialize()
            
            content_collection, context_collection = self._get_collection_names(index_name)
            
            # Create search optimizers for new collections
            content_key = f"{index_name}_content"
            context_key = f"{index_name}_context"
            
            # Check if collections exist in Qdrant
            if await self.qdrant_manager.collection_exists(content_collection):
                if content_key not in self.search_optimizers:
                    content_config = SearchConfig(
                        dimension=embedding_dimension,
                        collection_name=content_collection
                    )
                    content_optimizer = QdrantSearchOptimizer(content_config, self.qdrant_manager)
                    self.search_optimizers[content_key] = content_optimizer
                    logger.info(f"Added search optimizer for content collection: {content_collection}")
            
            if await self.qdrant_manager.collection_exists(context_collection):
                if context_key not in self.search_optimizers:
                    context_config = SearchConfig(
                        dimension=embedding_dimension,
                        collection_name=context_collection
                    )
                    context_optimizer = QdrantSearchOptimizer(context_config, self.qdrant_manager)
                    self.search_optimizers[context_key] = context_optimizer
                    logger.info(f"Added search optimizer for context collection: {context_collection}")
            
            # Update collections state
            self._collections_state[index_name] = {
                'content': content_collection,
                'context': context_collection,
                'dimension': embedding_dimension
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add collection for index {index_name}: {e}")
            return False

    async def remove_collection(self, index_name: str) -> bool:
        """Remove a collection from the storage manager incrementally."""
        try:
            content_key = f"{index_name}_content"
            context_key = f"{index_name}_context"
            
            # Remove search optimizers
            if content_key in self.search_optimizers:
                del self.search_optimizers[content_key]
                logger.info(f"Removed search optimizer for {content_key}")
            
            if context_key in self.search_optimizers:
                del self.search_optimizers[context_key]
                logger.info(f"Removed search optimizer for {context_key}")
            
            # Remove from collections state
            if index_name in self._collections_state:
                del self._collections_state[index_name]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove collection for index {index_name}: {e}")
            return False

    async def refresh_collections(self, force: bool = False) -> bool:
        """Refresh collections discovery only if needed."""
        try:
            if not self._is_initialized and not force:
                logger.debug("Storage manager not initialized, skipping refresh")
                return True
            
            if not force:
                # Check if collections state has actually changed
                current_collections = await self.qdrant_manager.list_active_collections()
                expected_collections = set()
                for index_name, state in self._collections_state.items():
                    expected_collections.add(state['content'])
                    expected_collections.add(state['context'])
                
                if set(current_collections) == expected_collections:
                    logger.debug("Collection state unchanged, skipping refresh")
                    return True
            
            logger.info("Refreshing collections discovery...")
            await self._discover_and_init_collections()
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh collections: {e}")
            return False

    async def cleanup(self):
        """Cleanup storage manager and close connections."""
        if self.qdrant_manager:
            await self.qdrant_manager.disconnect()
        
        self.search_optimizers.clear()
        self._collections_state.clear()
        self._is_initialized = False
        logger.info("Vector storage manager cleaned up")


# Global storage manager instance
_storage_manager: Optional[VectorStorageManager] = None

def get_storage_manager() -> VectorStorageManager:
    """Get the global vector storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = VectorStorageManager()
        # Note: Initialization must be done async, caller should call initialize() if needed
    return _storage_manager

async def get_initialized_storage_manager() -> VectorStorageManager:
    """Get the global vector storage manager instance and ensure it's initialized."""
    manager = get_storage_manager()
    if not manager._is_initialized:
        await manager.initialize()
    return manager

async def init_storage_manager() -> VectorStorageManager:
    """Initialize storage manager and return instance (legacy compatibility)."""
    return await get_initialized_storage_manager()