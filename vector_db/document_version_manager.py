"""
Database-integrated document version manager.
Provides version control for documents with database persistence and vector synchronization.
"""

import logging
import hashlib
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
    """Lazy import of EmbeddingManager."""
    try:
        from .embedding_manager import EmbeddingManager
        return EmbeddingManager
    except ImportError as e:
        logging.warning(f"EmbeddingManager not available: {e}")
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
        self.storage_manager = storage_manager or get_storage_manager()
        self.embedding_manager = None  # Will be initialized lazily
        self.chunker = None  # Will be initialized lazily
    
    def _get_embedding_manager_instance(self):
        """Get embedding manager instance, initializing if needed."""
        if self.embedding_manager is None:
            EmbeddingManager = _get_embedding_manager()
            if EmbeddingManager:
                self.embedding_manager = EmbeddingManager()
            else:
                raise ImportError("EmbeddingManager not available")
        return self.embedding_manager
    
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
                file_hash=content_hash,
                title=metadata.get('title', filename) if metadata else filename,
                description=metadata.get('description') if metadata else None,
                content_type=content_type,
                file_size=file_size,
                extracted_text=content,
                metadata=self._serialize_metadata(metadata) if metadata else None,
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
            
            # Create vector index and add embeddings
            index_name = f"doc_{document.id}"
            await self._create_vector_index(
                document=document,
                chunks_data=chunks_data,
                index_name=index_name,
                db=db
            )
            
            # Update document status
            document.status = DocumentStatusEnum.COMPLETED
            document.processed_at = datetime.now(timezone.utc)
            
            db.commit()
            
            logger.info(f"Created document version {version} for document {document.id}")
            return document
            
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
            # Chunk the content
            chunks = self.chunker.chunk_text(content)
            
            chunks_data = []
            for i, chunk in enumerate(chunks):
                # Create database record for chunk
                doc_chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    chunk_id=f"doc_{document_id}_chunk_{i}",
                    text=chunk.text,
                    text_length=len(chunk.text),
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    context_before=chunk.context.get('before', '') if chunk.context else '',
                    context_after=chunk.context.get('after', '') if chunk.context else '',
                    metadata=self._serialize_metadata(chunk.metadata) if chunk.metadata else None
                )
                
                db.add(doc_chunk)
                
                # Prepare chunk data for vector processing
                chunks_data.append({
                    'chunk': doc_chunk,
                    'text': chunk.text,
                    'context': chunk.context or {},
                    'metadata': chunk.metadata or {}
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
        db: Session
    ):
        """Create vector index and embeddings for document chunks."""
        try:
            # Generate embeddings for all chunks
            texts = [chunk_data['text'] for chunk_data in chunks_data]
            
            # Generate content embeddings
            embedding_manager = self._get_embedding_manager_instance()
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
                metadata = {
                    'document_id': document.id,
                    'chunk_id': chunk.chunk_id,
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
                chunk_ids.append(chunk.chunk_id)
            
            # Add vectors to storage
            added_ids = await self.storage_manager.add_vectors(
                index_name=index_name,
                content_vectors=content_embeddings,
                context_vectors=context_embeddings,
                metadata_list=vector_metadata,
                chunk_ids=chunk_ids
            )
            
            # Update chunks with embedding IDs
            for i, chunk_data in enumerate(chunks_data):
                chunk = chunk_data['chunk']
                chunk.content_embedding_id = added_ids[i] if i < len(added_ids) else None
                chunk.context_embedding_id = added_ids[i] if i < len(added_ids) else None
                chunk.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Default model
            
            logger.info(f"Created vector index {index_name} with {len(added_ids)} embeddings")
            
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
                # Delete vector index
                index_name = f"doc_{document.id}"
                await self.storage_manager.delete_index(index_name, db)
                
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
                # Soft delete
                document.is_deleted = True
                document.deleted_at = datetime.now(timezone.utc)
            
            db.commit()
            
            logger.info(f"Deleted document {document_id} ({'hard' if hard_delete else 'soft'} delete)")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete document version: {e}")
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
            
            document.is_deleted = False
            document.deleted_at = None
            
            db.commit()
            
            logger.info(f"Restored document {document_id}")
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