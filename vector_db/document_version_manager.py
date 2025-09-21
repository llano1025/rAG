"""
Database-integrated document version manager.
Provides version control for documents with database persistence and vector synchronization.
"""

import logging
import hashlib
import asyncio
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database.models import Document, DocumentChunk, VectorIndex, DocumentStatusEnum
from .storage_manager import VectorStorageManager, get_storage_manager

# Lazy import for chunking
def _get_adaptive_chunker():
    """Lazy import of AdaptiveChunker."""
    try:
        from .chunking import AdaptiveChunker
        return AdaptiveChunker
    except ImportError as e:
        logging.warning(f"AdaptiveChunker not available: {e}")
        return None

# Lazy import for EmbeddingManager
def _get_embedding_manager():
    """Lazy import of EnhancedEmbeddingManager."""
    try:
        from .embedding_manager import EnhancedEmbeddingManager
        return EnhancedEmbeddingManager
    except ImportError as e:
        logging.warning(f"EnhancedEmbeddingManager not available: {e}")
        return None

logger = logging.getLogger(__name__)

class DocumentVersionError(Exception):
    """Raised when document version operation fails."""
    pass

class DocumentVersionManager:
    """
    Manages document versions with database persistence and vector synchronization.
    
    Features:
    - Database-backed version history
    - Automatic vector index updates
    - Parent-child version relationships
    - Diff tracking and metadata
    """
    
    def __init__(self, storage_manager: VectorStorageManager = None):
        """Initialize document version manager."""
        logger.info("Initializing DocumentVersionManager")
        
        try:
            self.storage_manager = storage_manager or get_storage_manager()
            if self.storage_manager is None:
                logger.error("Failed to initialize storage manager")
                raise RuntimeError("Storage manager initialization failed")
        except Exception as e:
            logger.error(f"Failed to get storage manager: {e}")
            raise RuntimeError(f"Storage manager initialization failed: {e}")
        
        self.embedding_manager = None  # Will be initialized lazily
        self.chunker = None  # Will be initialized lazily
        
        logger.info(f"DocumentVersionManager initialized - storage_manager: {self.storage_manager is not None}")
    
    def _get_embedding_manager_instance(self, embedding_model: str = None):
        """Get embedding manager instance, initializing if needed."""
        # If a specific model is requested, create a new manager for that model
        if embedding_model is not None:
            return self._create_embedding_manager_for_model(embedding_model)
        
        if self.embedding_manager is None:
            try:
                EnhancedEmbeddingManager = _get_embedding_manager()
                if not EnhancedEmbeddingManager:
                    raise ImportError("EnhancedEmbeddingManager not available")
                
                # Try different embedding providers in order of preference
                providers_to_try = [
                    ("huggingface", self._create_huggingface_manager),
                    ("ollama", self._create_ollama_manager),
                    ("openai", self._create_openai_manager)
                ]
                
                last_error = None
                for provider_name, create_func in providers_to_try:
                    try:
                        logger.info(f"Attempting to initialize {provider_name} embedding manager")
                        self.embedding_manager = create_func(EnhancedEmbeddingManager)
                        if self.embedding_manager:
                            logger.info(f"Successfully initialized {provider_name} embedding manager")
                            break
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Failed to initialize {provider_name} embedding manager: {e}")
                        continue
                
                if self.embedding_manager is None:
                    error_msg = f"Failed to initialize any embedding provider. Last error: {last_error}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                logger.error(f"Failed to get embedding manager: {e}")
                raise RuntimeError(f"Embedding manager initialization failed: {e}")
        
        return self.embedding_manager
    
    def _create_embedding_manager_for_model(self, embedding_model: str):
        """Create embedding manager for a specific model using the model registry."""
        try:
            # Get the model registry
            from .embedding_model_registry import get_embedding_model_registry
            registry = get_embedding_model_registry()
            
            # Get model metadata
            model_metadata = registry.get_model(embedding_model)
            if model_metadata is None:
                logger.warning(f"Model {embedding_model} not found in registry, falling back to default")
                return self._get_embedding_manager_instance()
            
            # Get the enhanced embedding manager class
            EnhancedEmbeddingManager = _get_embedding_manager()
            if not EnhancedEmbeddingManager:
                raise ImportError("EnhancedEmbeddingManager not available")
            
            # Create manager based on provider
            if model_metadata.provider.value == "huggingface":
                manager = EnhancedEmbeddingManager.create_huggingface_manager(
                    model_name=model_metadata.model_name,
                    batch_size=16
                )
            elif model_metadata.provider.value == "ollama":
                manager = EnhancedEmbeddingManager.create_ollama_manager(
                    model_name=model_metadata.model_name,
                    base_url="http://localhost:11434",
                    batch_size=8
                )
            elif model_metadata.provider.value == "openai":
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not set, falling back to default model")
                    return self._get_embedding_manager_instance()
                manager = EnhancedEmbeddingManager.create_openai_manager(
                    api_key=api_key,
                    model_name=model_metadata.model_name,
                    batch_size=32
                )
            else:
                logger.warning(f"Unknown provider {model_metadata.provider}, falling back to default")
                return self._get_embedding_manager_instance()
            
            logger.info(f"Created embedding manager for model {embedding_model} ({model_metadata.provider.value})")
            return manager
            
        except Exception as e:
            logger.error(f"Failed to create embedding manager for model {embedding_model}: {e}")
            logger.info("Falling back to default embedding manager")
            return self._get_embedding_manager_instance()
    
    def _create_huggingface_manager(self, EnhancedEmbeddingManager):
        """Create HuggingFace embedding manager with error handling."""
        try:
            import torch
            # Check if torch is available and working
            if not torch.cuda.is_available():
                logger.info("CUDA not available, using CPU for embeddings")
            
            # Use a lightweight model by default for better reliability
            manager = EnhancedEmbeddingManager.create_huggingface_manager(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller, faster model
                batch_size=16  # Smaller batch size for stability
            )
            
            # Validate that the manager was created successfully
            if manager is None:
                raise RuntimeError("HuggingFace embedding manager creation returned None")
            
            return manager
            
        except ImportError as e:
            logger.warning(f"PyTorch/transformers not available for HuggingFace embeddings: {e}")
            raise
        except Exception as e:
            logger.warning(f"Failed to create HuggingFace embedding manager: {e}")
            raise
    
    def _create_ollama_manager(self, EnhancedEmbeddingManager):
        """Create Ollama embedding manager with error handling."""
        try:
            # Try to create Ollama manager with default settings
            manager = EnhancedEmbeddingManager.create_ollama_manager(
                base_url="http://localhost:11434",
                model_name="nomic-embed-text",
                batch_size=8
            )
            return manager
        except Exception as e:
            logger.warning(f"Failed to create Ollama embedding manager: {e}")
            raise
    
    def _create_openai_manager(self, EnhancedEmbeddingManager):
        """Create OpenAI embedding manager with error handling."""
        try:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            manager = EnhancedEmbeddingManager.create_openai_manager(
                api_key=api_key,
                model_name="text-embedding-ada-002",
                batch_size=32
            )
            return manager
        except Exception as e:
            logger.warning(f"Failed to create OpenAI embedding manager: {e}")
            raise
    
    def _get_chunker_instance(self):
        """Get chunker instance, initializing if needed."""
        if self.chunker is None:
            AdaptiveChunker = _get_adaptive_chunker()
            if AdaptiveChunker:
                self.chunker = AdaptiveChunker()
            else:
                raise ImportError("AdaptiveChunker not available")
        return self.chunker
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of document content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def create_document_version(
        self,
        user_id: int,
        filename: str,
        content: str,
        content_type: str,
        file_size: int,
        metadata: Dict = None,
        parent_document_id: int = None,
        embedding_model: str = None,
        db: Session = None
    ) -> Document:
        """
        Create a new document version with automatic chunking and vector indexing.
        
        Args:
            user_id: ID of the user creating the document
            filename: Name of the document file
            content: Extracted text content
            content_type: MIME type of the document
            file_size: Size of the original file in bytes
            metadata: Additional metadata dictionary
            parent_document_id: ID of parent document if this is a version
            db: Database session
            
        Returns:
            Document model instance
        """
        try:
            # Calculate content hash for deduplication
            content_hash = self._calculate_content_hash(content)
            
            # Check if identical content already exists
            existing_doc = db.query(Document).filter(
                Document.file_hash == content_hash,
                Document.user_id == user_id,
                Document.is_deleted == False
            ).first()
            
            if existing_doc:
                logger.info(f"Document with identical content already exists: {existing_doc.id}")
                return existing_doc
            
            # Determine version number
            version = 1
            if parent_document_id:
                latest_version = db.query(Document).filter(
                    Document.parent_document_id == parent_document_id
                ).order_by(desc(Document.version)).first()
                
                if latest_version:
                    version = latest_version.version + 1
                else:
                    # Parent document is version 1, this is version 2
                    parent_doc = db.query(Document).filter(Document.id == parent_document_id).first()
                    if parent_doc:
                        version = parent_doc.version + 1
            
            # Create document record
            document = Document(
                user_id=user_id,
                filename=filename,
                original_filename=filename,
                file_path=metadata.get('file_path') if metadata else None,
                file_hash=metadata.get('file_hash', content_hash),  # Use provided file_hash or fallback to content_hash
                title=metadata.get('title', filename) if metadata else filename,
                description=metadata.get('description') if metadata else None,
                content_type=content_type,
                file_size=file_size,
                extracted_text=content,
                document_metadata=self._serialize_metadata(metadata) if metadata else None,
                version=version,
                parent_document_id=parent_document_id,
                status=DocumentStatusEnum.PROCESSING
            )
            
            db.add(document)
            db.flush()  # Get the document ID
            
            # Process content into chunks
            chunks_data = await self._process_document_chunks(
                document_id=document.id,
                content=content,
                db=db
            )
            
            # Commit document and chunks to database 
            document.status = DocumentStatusEnum.PROCESSING  # Keep processing until vectors succeed
            db.commit()
            logger.info(f"Document and chunks committed to database: {document.id}")
            
            # Create vector index and add embeddings
            index_name = f"doc_{document.id}"
            try:
                await self._create_vector_index(
                    document=document,
                    chunks_data=chunks_data,
                    index_name=index_name,
                    embedding_model=embedding_model,
                    db=db
                )
                
                # Only mark as completed if vector creation succeeds
                document.status = DocumentStatusEnum.COMPLETED
                document.processed_at = datetime.now(timezone.utc)
                db.commit()
                
                logger.info(f"Created document version {version} for document {document.id}")
                return document
                
            except Exception as vector_error:
                # Vector creation failed, but chunks are already saved
                logger.error(f"Vector creation failed for document {document.id}: {vector_error}")
                document.status = DocumentStatusEnum.FAILED
                db.commit()
                
                # Re-raise the error for caller to handle
                raise DocumentVersionError(f"Vector creation failed: {vector_error}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create document version: {e}")
            raise DocumentVersionError(f"Failed to create document version: {e}")
    
    async def _process_document_chunks(
        self,
        document_id: int,
        content: str,
        db: Session
    ) -> List[Dict]:
        """Process document content into chunks and store in database."""
        try:
            # Get chunker instance and chunk the content
            chunker = self._get_chunker_instance()
            chunks = chunker.chunk_document(content, {})
            
            chunks_data = []
            for i, chunk in enumerate(chunks):
                # Generate UUID for Qdrant point ID
                qdrant_point_id = str(uuid.uuid4())
                
                # Create database record for chunk
                doc_chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    chunk_id=f"doc_{document_id}_chunk_{i}",  # Keep original format for database
                    text=chunk.text,
                    text_length=len(chunk.text),
                    start_char=chunk.start_idx,
                    end_char=chunk.end_idx,
                    context_before=chunk.context_text[:200] if chunk.context_text else '',
                    context_after=chunk.context_text[-200:] if chunk.context_text else '',
                    metadata=self._serialize_metadata(chunk.metadata) if chunk.metadata else None
                )
                
                db.add(doc_chunk)
                
                # Prepare chunk data for vector processing
                # Convert context_text to context dict format
                context_dict = {}
                if chunk.context_text:
                    # Split context_text into before/after if it contains the main text
                    context_dict = {
                        'before': chunk.context_text[:200] if chunk.context_text else '',
                        'after': chunk.context_text[-200:] if chunk.context_text else ''
                    }
                
                chunks_data.append({
                    'chunk': doc_chunk,
                    'text': chunk.text,
                    'context': context_dict,
                    'metadata': chunk.metadata or {},
                    'qdrant_point_id': qdrant_point_id  # Add UUID for Qdrant
                })
            
            db.flush()  # Ensure chunk IDs are generated
            return chunks_data
            
        except Exception as e:
            logger.error(f"Failed to process document chunks: {e}")
            raise
    
    async def _create_vector_index(
        self,
        document: Document,
        chunks_data: List[Dict],
        index_name: str,
        embedding_model: str = None,
        db: Session = None
    ):
        """Create vector index and embeddings for document chunks."""
        try:
            # Check if storage manager is available
            if self.storage_manager is None:
                logger.error("Storage manager not available for vector indexing")
                raise RuntimeError("Vector storage service is currently unavailable")
            
            # Generate embeddings for all chunks
            texts = [chunk_data['text'] for chunk_data in chunks_data]
            
            # Generate content embeddings
            embedding_manager = self._get_embedding_manager_instance(embedding_model)
            if embedding_manager is None:
                logger.error("Embedding manager not available for vector generation")
                raise RuntimeError("Embedding generation service is currently unavailable")
                
            content_embeddings = await embedding_manager.generate_embeddings(texts)
            
            # Generate context embeddings (text + context)
            context_texts = []
            for chunk_data in chunks_data:
                context = chunk_data['context']
                context_text = f"{context.get('before', '')} {chunk_data['text']} {context.get('after', '')}".strip()
                context_texts.append(context_text)
            
            context_embeddings = await embedding_manager.generate_embeddings(context_texts)
            
            # Create vector index
            await self.storage_manager.create_index(
                index_name=index_name,
                embedding_dimension=len(content_embeddings[0]),
                user_id=document.user_id,
                document_id=document.id,
                db=db
            )
            
            # Prepare metadata for vector storage
            vector_metadata = []
            chunk_ids = []
            for i, chunk_data in enumerate(chunks_data):
                chunk = chunk_data['chunk']
                qdrant_point_id = chunk_data['qdrant_point_id']
                
                metadata = {
                    'document_id': document.id,
                    'chunk_id': chunk.chunk_id,  # Keep original for reference
                    'chunk_index': chunk.chunk_index,
                    'user_id': document.user_id,
                    'filename': document.filename,
                    'content_type': document.content_type,
                    'version': document.version,
                    'text_length': chunk.text_length,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                }
                
                # Add custom metadata
                if chunk_data['metadata']:
                    metadata.update(chunk_data['metadata'])
                
                vector_metadata.append(metadata)
                chunk_ids.append(chunk.chunk_id)  # Use original string chunk_id

            logger.info(f"{chunk_ids}")
            
            # Add vectors to storage
            added_ids = await self.storage_manager.add_vectors(
                index_name=index_name,
                content_vectors=content_embeddings,
                context_vectors=context_embeddings,
                metadata_list=vector_metadata,
                chunk_ids=chunk_ids
            )
            
            # Get the actual model name used for embeddings
            actual_model_name = self._get_actual_model_name(embedding_model)
            
            # Update chunks with embedding IDs
            for i, chunk_data in enumerate(chunks_data):
                chunk = chunk_data['chunk']
                chunk.content_embedding_id = added_ids[i] if i < len(added_ids) else None
                chunk.context_embedding_id = added_ids[i] if i < len(added_ids) else None
                chunk.embedding_model = actual_model_name
            
            logger.info(f"Created vector index {index_name} with {len(added_ids)} embeddings")
            
            # Add new collection to storage manager incrementally
            try:
                embedding_dimension = len(content_embeddings[0]) if content_embeddings else 768
                await self.storage_manager.add_collection(index_name, embedding_dimension)
                logger.info(f"Added collection {index_name} to storage manager")
            except Exception as e:
                logger.warning(f"Failed to add collection to storage manager: {e}")
                # Continue - this is for performance optimization only
            
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            raise
    
    def get_document_versions(
        self,
        document_id: int,
        db: Session,
        include_deleted: bool = False
    ) -> List[Document]:
        """Get all versions of a document."""
        try:
            query = db.query(Document).filter(
                (Document.id == document_id) | (Document.parent_document_id == document_id)
            )
            
            if not include_deleted:
                query = query.filter(Document.is_deleted == False)
            
            versions = query.order_by(Document.version).all()
            return versions
            
        except Exception as e:
            logger.error(f"Failed to get document versions: {e}")
            return []
    
    def get_latest_version(self, document_id: int, db: Session) -> Optional[Document]:
        """Get the latest version of a document."""
        try:
            latest_version = db.query(Document).filter(
                (Document.id == document_id) | (Document.parent_document_id == document_id),
                Document.is_deleted == False
            ).order_by(desc(Document.version)).first()
            
            return latest_version
            
        except Exception as e:
            logger.error(f"Failed to get latest version: {e}")
            return None
    
    async def delete_document_version(
        self,
        document_id: int,
        user_id: int,
        db: Session,
        hard_delete: bool = False
    ) -> bool:
        """Delete a document version and its associated vectors."""
        try:
            document = db.query(Document).filter(
                Document.id == document_id,
                Document.user_id == user_id
            ).first()
            
            if not document:
                raise DocumentVersionError("Document not found or access denied")
            
            if hard_delete:
                # Try to delete vector index if storage manager is available
                index_name = f"doc_{document.id}"
                if self.storage_manager is not None:
                    try:
                        await self.storage_manager.delete_index(index_name, db)
                        logger.info(f"Successfully deleted vector index {index_name}")
                        
                        # Remove collection from storage manager
                        try:
                            await self.storage_manager.remove_collection(index_name)
                            logger.info(f"Removed collection {index_name} from storage manager")
                        except Exception as e:
                            logger.warning(f"Failed to remove collection from storage manager: {e}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to delete vector index {index_name}: {e}")
                        # Continue with document deletion even if vector deletion fails
                else:
                    logger.warning(f"Storage manager not available, skipping vector index deletion for {index_name}")
                
                # Hard delete from database
                # First delete chunks
                db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document_id
                ).delete()
                
                # Delete vector indices
                db.query(VectorIndex).filter(
                    VectorIndex.document_id == document_id
                ).delete()
                
                # Delete document
                db.delete(document)
            else:
                # Soft delete document
                document.is_deleted = True
                document.deleted_at = datetime.now(timezone.utc)
                
                # Soft delete associated vector indices
                vector_indices = db.query(VectorIndex).filter(
                    VectorIndex.document_id == document_id
                ).all()
                
                for vector_index in vector_indices:
                    vector_index.is_active = False
                    vector_index.build_status = "soft_deleted"
                    vector_index.updated_at = datetime.now(timezone.utc)
                
                # Soft delete vector storage (remove from memory but keep files)
                index_name = f"doc_{document.id}"
                if self.storage_manager:
                    try:
                        await self.storage_manager.soft_delete_index(index_name)
                        logger.info(f"Soft deleted vector storage for {index_name}")
                    except Exception as e:
                        logger.warning(f"Failed to soft delete vector storage {index_name}: {e}")
                        # Continue with document deletion even if vector soft deletion fails
            
            # Invalidate search cache for this document
            try:
                from .search_engine import get_search_engine
                search_engine = get_search_engine()
                await search_engine.invalidate_cache_for_documents([document_id])
            except Exception as e:
                logger.warning(f"Failed to invalidate search cache for document {document_id}: {e}")
            
            db.commit()
            
            logger.info(f"Deleted document {document_id} ({'hard' if hard_delete else 'soft'} delete)")
            return True
            
        except DocumentVersionError:
            # Re-raise specific document version errors
            db.rollback()
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete document version {document_id}: {e}")
            # Provide more specific error information
            if "storage_manager" in str(e).lower() or "vector" in str(e).lower():
                logger.error("Vector storage operations failed, but document deletion may still proceed")
            return False
    
    async def restore_document_version(
        self,
        document_id: int,
        user_id: int,
        db: Session
    ) -> bool:
        """Restore a soft-deleted document version."""
        try:
            document = db.query(Document).filter(
                Document.id == document_id,
                Document.user_id == user_id,
                Document.is_deleted == True
            ).first()
            
            if not document:
                raise DocumentVersionError("Document not found, access denied, or not deleted")
            
            # Restore document
            document.is_deleted = False
            document.deleted_at = None
            
            # Restore associated vector indices
            vector_indices = db.query(VectorIndex).filter(
                VectorIndex.document_id == document_id
            ).all()
            
            for vector_index in vector_indices:
                vector_index.is_active = True
                vector_index.build_status = "ready"
                vector_index.updated_at = datetime.now(timezone.utc)
            
            # Reload FAISS indices into memory if they exist on disk
            index_name = f"doc_{document_id}"
            if self.storage_manager:
                try:
                    # Try to load the index if FAISS files exist
                    embedding_dimension = vector_indices[0].embedding_dimension if vector_indices else 768
                    await self.storage_manager.load_index(index_name, embedding_dimension)
                    logger.info(f"Reloaded vector storage for {index_name}")
                except Exception as e:
                    logger.warning(f"Failed to reload vector storage {index_name}: {e}")
                    # Continue with document restoration even if vector loading fails
            
            # Invalidate search cache since document is now available again
            try:
                from .search_engine import get_search_engine
                search_engine = get_search_engine()
                await search_engine.invalidate_cache_for_documents([document_id])
            except Exception as e:
                logger.warning(f"Failed to invalidate search cache for document {document_id}: {e}")
            
            db.commit()
            
            logger.info(f"Restored document {document_id} with {len(vector_indices)} vector indices")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to restore document version: {e}")
            return False
    
    def get_version_diff(
        self,
        version1_id: int,
        version2_id: int,
        db: Session
    ) -> Dict:
        """Get differences between two document versions."""
        try:
            doc1 = db.query(Document).filter(Document.id == version1_id).first()
            doc2 = db.query(Document).filter(Document.id == version2_id).first()
            
            if not doc1 or not doc2:
                raise DocumentVersionError("One or both document versions not found")
            
            # Basic diff information
            diff = {
                'version1': {
                    'id': doc1.id,
                    'version': doc1.version,
                    'created_at': doc1.created_at.isoformat(),
                    'file_size': doc1.file_size,
                    'content_length': len(doc1.extracted_text) if doc1.extracted_text else 0
                },
                'version2': {
                    'id': doc2.id,
                    'version': doc2.version,
                    'created_at': doc2.created_at.isoformat(),
                    'file_size': doc2.file_size,
                    'content_length': len(doc2.extracted_text) if doc2.extracted_text else 0
                },
                'changes': {
                    'content_changed': doc1.file_hash != doc2.file_hash,
                    'size_diff': doc2.file_size - doc1.file_size,
                    'content_length_diff': (len(doc2.extracted_text) if doc2.extracted_text else 0) - 
                                         (len(doc1.extracted_text) if doc1.extracted_text else 0)
                }
            }
            
            # Get chunk differences
            chunks1 = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == version1_id
            ).count()
            
            chunks2 = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == version2_id
            ).count()
            
            diff['changes']['chunks_diff'] = chunks2 - chunks1
            
            return diff
            
        except Exception as e:
            logger.error(f"Failed to get version diff: {e}")
            return {'error': str(e)}
    
    def _get_actual_model_name(self, embedding_model: str = None) -> str:
        """Get the actual model name used for embeddings."""
        if embedding_model is None:
            # Default fallback model
            return "sentence-transformers/all-MiniLM-L6-v2"
        
        try:
            # Get the model registry
            from .embedding_model_registry import get_embedding_model_registry
            registry = get_embedding_model_registry()
            
            # Get model metadata
            model_metadata = registry.get_model(embedding_model)
            if model_metadata is not None:
                return model_metadata.model_name
            else:
                logger.warning(f"Model {embedding_model} not found in registry, using default")
                return "sentence-transformers/all-MiniLM-L6-v2"
                
        except Exception as e:
            logger.error(f"Error getting model name for {embedding_model}: {e}")
            return "sentence-transformers/all-MiniLM-L6-v2"
    
    def _serialize_metadata(self, metadata: Dict) -> str:
        """Serialize metadata dictionary to JSON string."""
        import json
        return json.dumps(metadata, default=str)
    
    def _deserialize_metadata(self, metadata_str: str) -> Dict:
        """Deserialize JSON string to metadata dictionary."""
        import json
        if metadata_str:
            return json.loads(metadata_str)
        return {}


# Global version manager instance
_version_manager: Optional[DocumentVersionManager] = None

def get_version_manager() -> DocumentVersionManager:
    """Get the global document version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = DocumentVersionManager()
    return _version_manager