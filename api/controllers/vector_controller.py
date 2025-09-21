"""
Vector controller for document processing and vector operations.
Integrates with authentication system and provides user-aware document management.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any, Union
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from dataclasses import asdict

from database.models import User, Document, DocumentChunk, VectorIndex, DocumentStatusEnum, PermissionEnum
from vector_db.storage_manager import VectorStorageManager, get_storage_manager, init_storage_manager
from vector_db.document_version_manager import DocumentVersionManager, get_version_manager
from vector_db.search_engine import EnhancedSearchEngine, get_search_engine, SearchFilter, SearchType
from file_processor.text_extractor import TextExtractor
from file_processor.type_detector import FileTypeDetector
from file_processor.metadata_extractor import MetadataExtractor
from utils.security.audit_logger import AuditLogger
from utils.file_storage import get_file_manager
import hashlib

logger = logging.getLogger(__name__)

class VectorControllerError(Exception):
    """Raised when vector controller operation fails."""
    pass

class VectorController:
    """
    Vector operations controller with user authentication and access control.
    
    Handles:
    - Document upload and processing
    - Vector indexing and storage
    - Document search and retrieval
    - User access control
    - Version management
    """
    
    def __init__(
        self,
        storage_manager: VectorStorageManager = None,
        version_manager: DocumentVersionManager = None,
        search_engine: EnhancedSearchEngine = None,
        audit_logger: AuditLogger = None
    ):
        """Initialize vector controller with dependencies."""
        logger.info("Initializing VectorController with dependencies")
        
        self.storage_manager = storage_manager or get_storage_manager()
        self.version_manager = version_manager or get_version_manager()
        self.search_engine = search_engine or get_search_engine()
        self.audit_logger = audit_logger
        self._initialized = False
        
        # Log dependency initialization status
        logger.info(f"VectorController initialized - storage_manager: {self.storage_manager is not None}, "
                   f"version_manager: {self.version_manager is not None}, "
                   f"search_engine: {self.search_engine is not None}, "
                   f"audit_logger: {self.audit_logger is not None}")
        
        # File processing components
        self.text_extractor = TextExtractor()
        self.type_detector = FileTypeDetector()
        self.metadata_extractor = MetadataExtractor()
        self.file_manager = get_file_manager()
    
    async def _ensure_initialized(self):
        """Ensure storage manager is properly initialized."""
        if not self._initialized:
            await self.storage_manager.initialize()
            self._initialized = True
            logger.info("Storage manager initialized successfully")
    
    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        user: User,
        metadata: Dict[str, Any] = None,
        embedding_model: str = None,
        tags: Optional[List[str]] = None,
        db: Session = None,
        websocket_manager = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Upload and process a document for a user.
        
        Args:
            file_content: Raw file content bytes
            filename: Original filename
            user: User uploading the document
            metadata: Additional metadata
            db: Database session
            
        Returns:
            Dictionary with document processing results
        """
        try:
            # Ensure storage manager is initialized
            await self._ensure_initialized()
            
            # Check if dependencies are available
            if self.storage_manager is None or self.version_manager is None:
                logger.error("Storage or version manager not available for document upload")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Document upload service is currently unavailable"
                )
            
            # Validate user permissions
            if not user.has_permission(PermissionEnum.WRITE_DOCUMENTS):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User does not have permission to upload documents"
                )
            
            # Detect file type
            content_type = self.type_detector.detect_type(file_content=file_content, filename=filename)
            file_size = len(file_content)
            
            # Validate file type and size
            if not self._is_supported_file_type(content_type):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {content_type}"
                )
            
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File size exceeds 50MB limit"
                )
            
            # Emit text extraction progress
            if websocket_manager and user_id:
                await websocket_manager.emit_document_progress(
                    user_id=user_id,
                    document_id=0,
                    filename=filename,
                    stage="extracting_text",
                    progress=30
                )

            # Extract text content with OCR settings from metadata
            ocr_method = metadata.get('ocr_method') if metadata else None
            ocr_language = metadata.get('ocr_language') if metadata else None
            vision_provider = metadata.get('vision_provider') if metadata else None

            extracted_text = await self.text_extractor.extract_text(
                file_content, content_type, filename,
                ocr_method=ocr_method,
                ocr_language=ocr_language,
                vision_provider=vision_provider
            )

            # Emit text extraction complete
            if websocket_manager and user_id:
                await websocket_manager.emit_document_progress(
                    user_id=user_id,
                    document_id=0,
                    filename=filename,
                    stage="text_extracted",
                    progress=40
                )
            
            # More lenient validation - allow shorter text for certain document types
            min_length = 5 if ocr_method == "vision_llm" else 10
            if not extracted_text or len(extracted_text.strip()) < min_length:
                # Provide more specific error message
                extraction_method = "Vision LLM OCR" if ocr_method == "vision_llm" else "regular text extraction"
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not extract meaningful text content from file using {extraction_method}. "
                           f"Try using a different OCR method or ensure the document contains readable text."
                )
            
            # Calculate file hash for storage and deduplication
            file_hash = hashlib.sha256(file_content).hexdigest()
            
            # Save original file to storage
            file_path = None
            try:
                logger.info(f"VectorController: Attempting to save original file to storage...")
                file_path = self.file_manager.save_file(file_content, filename, file_hash)
                logger.info(f"VectorController: Original file saved to: {file_path}")
            except Exception as e:
                logger.error(f"VectorController: Failed to save original file: {e}")
                logger.error(f"File: {filename}, Size: {len(file_content)} bytes, Type: {content_type}")
                
                # Don't continue processing if we can't store the original file
                if content_type and (content_type.startswith('image/') or 
                                   content_type == 'application/pdf' or
                                   content_type.startswith('application/vnd.') or
                                   content_type == 'application/msword'):
                    logger.error(f"VectorController: File storage is critical for binary files, aborting upload")
                    raise HTTPException(
                        status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
                        detail=f"Failed to store original file: {str(e)}. Original file storage is required for {content_type} files."
                    )
                else:
                    logger.warning(f"VectorController: Continuing without file storage for text-based file")
                    file_path = None
            
            # Extract metadata
            file_metadata = await self.metadata_extractor.extract_metadata(file_content, content_type)
            
            # Convert DocumentMetadata dataclass to dictionary
            file_metadata_dict = asdict(file_metadata)
            
            # Convert datetime objects to ISO format strings for JSON serialization
            if 'creation_date' in file_metadata_dict and file_metadata_dict['creation_date']:
                file_metadata_dict['creation_date'] = file_metadata_dict['creation_date'].isoformat()
            if 'modification_date' in file_metadata_dict and file_metadata_dict['modification_date']:
                file_metadata_dict['modification_date'] = file_metadata_dict['modification_date'].isoformat()
            
            # Combine metadata
            combined_metadata = metadata or {}
            combined_metadata.update(file_metadata_dict)
            combined_metadata.update({
                'original_filename': filename,
                'uploaded_at': datetime.utcnow().isoformat(),
                'content_type': content_type,
                'file_size': file_size,
                'text_length': len(extracted_text),
                'user_id': user.id,
                'file_hash': file_hash,
                'file_path': file_path
            })
            
            # Add OCR metadata for image files
            if content_type and content_type.startswith('image/'):
                combined_metadata.update({
                    'ocr_processing': {
                        'method_used': ocr_method or 'tesseract',
                        'language': ocr_language or 'eng',
                        'vision_provider': vision_provider if ocr_method == 'vision_llm' else None,
                        'processed_at': datetime.utcnow().isoformat()
                    }
                })
            
            # Emit chunking and embedding generation progress
            if websocket_manager and user_id:
                await websocket_manager.emit_document_progress(
                    user_id=user_id,
                    document_id=0,
                    filename=filename,
                    stage="generating_embeddings",
                    progress=45
                )

            # Create document version with vector indexing
            document = await self.version_manager.create_document_version(
                user_id=user.id,
                filename=filename,
                content=extracted_text,
                content_type=content_type,
                file_size=file_size,
                metadata=combined_metadata,
                embedding_model=embedding_model,
                db=db,
                websocket_manager=websocket_manager,  # Pass websocket manager
                progress_user_id=user_id  # Pass user_id for progress tracking
            )

            # Emit embedding generation complete
            if websocket_manager and user_id:
                await websocket_manager.emit_document_progress(
                    user_id=user_id,
                    document_id=document.id,
                    filename=filename,
                    stage="storing_vectors",
                    progress=95
                )
            
            # Set tags if provided
            if tags:
                # Refresh the document object to ensure it's in this session
                db.refresh(document)
                document.set_tag_list(tags)
                db.add(document)  # Ensure the document is tracked in this session
                db.commit()  # Commit the tag changes
            
            # Log successful upload
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="user_action",
                    user_id=str(user.id),
                    action="document_uploaded",
                    resource_type="document",
                    resource_id=str(document.id),
                    details={
                        'filename': filename,
                        'content_type': content_type,
                        'file_size': file_size,
                        'text_length': len(extracted_text)
                    }
                )
            
            return {
                'document_id': document.id,
                'filename': document.filename,
                'status': document.status.value,
                'content_type': content_type,
                'file_size': file_size,
                'text_length': len(extracted_text),
                'chunks_count': len(document.chunks),
                'version': document.version,
                'created_at': document.created_at.isoformat(),
                'processed_at': document.processed_at.isoformat() if document.processed_at else None
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document upload failed"
            )
    
    async def search_documents(
        self,
        query: str,
        user: User,
        search_type: str = SearchType.SEMANTIC,
        limit: int = 10,
        filters: Dict[str, Any] = None,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents for a user with access control.
        
        Args:
            query: Search query text
            user: User performing the search
            search_type: Type of search (semantic, keyword, hybrid, contextual)
            limit: Maximum number of results
            filters: Additional search filters
            db: Database session
            
        Returns:
            List of search results
        """
        try:
            # Validate user permissions
            if not user.has_permission(PermissionEnum.READ_DOCUMENTS):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User does not have permission to search documents"
                )
            
            # Create search filter
            search_filter = SearchFilter()
            if filters:
                if 'content_types' in filters:
                    search_filter.content_types = filters['content_types']
                if 'date_range' in filters:
                    search_filter.date_range = filters['date_range']
                if 'tags' in filters:
                    search_filter.tags = filters['tags']
                if 'language' in filters:
                    search_filter.language = filters['language']
                if 'min_score' in filters:
                    search_filter.min_score = filters['min_score']
            
            # Perform search
            results = await self.search_engine.search(
                query=query,
                user=user,
                search_type=search_type,
                filters=search_filter,
                limit=limit,
                db=db
            )
            
            # Format results for API response
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'chunk_id': result.chunk_id,
                    'document_id': result.document_id,
                    'text': result.text,
                    'score': result.score,
                    'highlight': result.highlight,
                    'metadata': result.metadata,
                    'document_metadata': result.document_metadata,
                    'timestamp': result.timestamp.isoformat()
                })
            
            # Log search
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="user_action",
                    user_id=str(user.id),
                    action="documents_searched",
                    resource_type="search",
                    details={
                        'query': query,
                        'search_type': search_type,
                        'results_count': len(results),
                        'limit': limit
                    }
                )
            
            return formatted_results
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document search failed"
            )
    
    async def get_document(
        self,
        document_id: int,
        user: User,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Get document details with access control.
        
        Args:
            document_id: ID of the document to retrieve
            user: User requesting the document
            db: Database session
            
        Returns:
            Document details dictionary
        """
        try:
            # Get document from database
            document = db.query(Document).filter(
                Document.id == document_id,
                Document.is_deleted == False
            ).first()
            
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
            
            # Check access permissions
            if not document.can_access(user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this document"
                )
            
            # Get document chunks
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index).all()
            
            # Format response
            document_data = {
                'id': document.id,
                'filename': document.filename,
                'title': document.title,
                'description': document.description,
                'content_type': document.content_type,
                'file_size': document.file_size,
                'status': document.status.value,
                'version': document.version,
                'is_public': document.is_public,
                'user_id': document.user_id,
                'created_at': document.created_at.isoformat(),
                'updated_at': document.updated_at.isoformat(),
                'processed_at': document.processed_at.isoformat() if document.processed_at else None,
                'extracted_text': document.extracted_text,
                'language': document.language,
                'tags': document.get_tag_list(),
                'chunks_count': len(chunks),
                'chunks': [
                    {
                        'id': chunk.id,
                        'chunk_id': chunk.chunk_id,
                        'chunk_index': chunk.chunk_index,
                        'text': chunk.text,
                        'text_length': chunk.text_length,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        'context_before': chunk.context_before,
                        'context_after': chunk.context_after,
                        'embedding_model': chunk.embedding_model
                    }
                    for chunk in chunks
                ]
            }
            
            return document_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve document"
            )
    
    async def list_user_documents(
        self,
        user: User,
        skip: int = 0,
        limit: int = 50,
        include_public: bool = True,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        List documents accessible to a user.
        
        Args:
            user: User requesting the list
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            include_public: Whether to include public documents
            db: Database session
            
        Returns:
            Dictionary with documents list and metadata
        """
        try:
            # Build query for accessible documents
            query = db.query(Document).filter(Document.is_deleted == False)
            
            if user.is_superuser or user.has_role("admin"):
                # Admins can see all documents
                pass
            else:
                # Regular users see their own documents and public ones
                if include_public:
                    query = query.filter(
                        (Document.user_id == user.id) | (Document.is_public == True)
                    )
                else:
                    query = query.filter(Document.user_id == user.id)
            
            # Get total count
            total_count = query.count()
            
            # Get paginated documents
            documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
            
            # Format documents
            documents_data = []
            for doc in documents:
                # Get embedding model from first chunk (all chunks should use same model)
                embedding_model = None
                if doc.chunks:
                    # Get the first chunk's embedding model
                    embedding_model = doc.chunks[0].embedding_model

                doc_data = {
                    'id': doc.id,
                    'filename': doc.filename,
                    'title': doc.title,
                    'description': doc.description,
                    'content_type': doc.content_type,
                    'file_size': doc.file_size,
                    'status': doc.status.value,
                    'version': doc.version,
                    'is_public': doc.is_public,
                    'user_id': doc.user_id,
                    'created_at': doc.created_at.isoformat(),
                    'updated_at': doc.updated_at.isoformat(),
                    'processed_at': doc.processed_at.isoformat() if doc.processed_at else None,
                    'language': doc.language,
                    'tags': doc.get_tag_list(),
                    'chunks_count': len(doc.chunks),
                    'embedding_model': embedding_model
                }
                documents_data.append(doc_data)
            
            return {
                'documents': documents_data,
                'total_count': total_count,
                'skip': skip,
                'limit': limit,
                'has_more': skip + len(documents) < total_count
            }
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list documents"
            )
    
    async def delete_document(
        self,
        document_id: int,
        user: User,
        hard_delete: bool = False,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Delete a document with access control.
        
        Args:
            document_id: ID of the document to delete
            user: User requesting the deletion
            hard_delete: Whether to permanently delete or soft delete
            db: Database session
            
        Returns:
            Dictionary with deletion status
        """
        try:
            # Ensure storage manager is initialized
            await self._ensure_initialized()
            
            # Check if version manager is available
            if self.version_manager is None:
                logger.error("Version manager not available for document deletion")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Document deletion service is currently unavailable"
                )
            
            # Get document
            document = db.query(Document).filter(
                Document.id == document_id,
                Document.is_deleted == False
            ).first()
            
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
            
            # Check permissions
            if document.user_id != user.id and not user.has_permission(PermissionEnum.DELETE_DOCUMENTS):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: cannot delete this document"
                )
            
            # Delete document
            success = await self.version_manager.delete_document_version(
                document_id=document_id,
                user_id=user.id,
                db=db,
                hard_delete=hard_delete
            )
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to delete document"
                )
            
            # Log deletion
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="user_action",
                    user_id=str(user.id),
                    action="document_deleted",
                    resource_type="document",
                    resource_id=str(document_id),
                    details={
                        'filename': document.filename,
                        'hard_delete': hard_delete
                    }
                )
            
            return {
                'document_id': document_id,
                'deleted': True,
                'hard_delete': hard_delete,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document deletion failed"
            )
    
    async def get_vector_stats(
        self,
        user: User,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Get vector database statistics for a user.
        
        Args:
            user: User requesting the statistics
            db: Database session
            
        Returns:
            Dictionary with vector statistics
        """
        try:
            # Ensure storage manager is initialized
            await self._ensure_initialized()
            
            # Get user's documents
            user_docs = db.query(Document).filter(
                Document.user_id == user.id,
                Document.is_deleted == False
            ).all()
            
            # Get vector indices
            user_indices = db.query(VectorIndex).filter(
                VectorIndex.user_id == user.id
            ).all()
            
            # Calculate statistics
            total_documents = len(user_docs)
            total_chunks = sum(len(doc.chunks) for doc in user_docs)
            total_indices = len(user_indices)
            
            # Get index statistics
            index_stats = {}
            for doc in user_docs:
                index_name = f"doc_{doc.id}"
                try:
                    stats = await self.storage_manager.get_index_stats(index_name)
                    index_stats[index_name] = stats
                except Exception as e:
                    logger.warning(f"Failed to get stats for index {index_name}: {e}")
            
            return {
                'user_id': user.id,
                'total_documents': total_documents,
                'total_chunks': total_chunks,
                'total_indices': total_indices,
                'documents_by_status': self._get_documents_by_status(user_docs),
                'documents_by_type': self._get_documents_by_type(user_docs),
                'index_statistics': index_stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get vector stats: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve vector statistics"
            )
    
    def _is_supported_file_type(self, content_type: str) -> bool:
        """Check if file type is supported for processing."""
        supported_types = [
            # Document types
            'text/plain',
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/html',
            'application/json',
            'text/markdown',
            'text/csv',
            'text/rtf',
            'application/rtf',
            # Image types (for OCR processing)
            'image/jpeg',
            'image/jpg',
            'image/png',
            'image/tiff',
            'image/tif',
            'image/gif',
            'image/pjpeg',
            'image/x-png',
            'image/webp',
            'image/bmp',
            'image/x-ms-bmp',
            # Additional variations
            'image/jp2',
            'image/jpx',
            'image/jpm',
            'image/jpeg2000',
            'image/jpeg2000-image',
        ]
        return content_type in supported_types
    
    def _get_documents_by_status(self, documents: List[Document]) -> Dict[str, int]:
        """Get document count by status."""
        status_counts = {}
        for doc in documents:
            status = doc.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts
    
    def _get_documents_by_type(self, documents: List[Document]) -> Dict[str, int]:
        """Get document count by content type."""
        type_counts = {}
        for doc in documents:
            content_type = doc.content_type
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        return type_counts


# Dependency injection
def get_vector_controller(audit_logger: AuditLogger = None) -> VectorController:
    """Get vector controller instance with dependencies."""
    try:
        # Initialize all dependencies first to ensure they work
        storage_manager = get_storage_manager()
        version_manager = get_version_manager()
        search_engine = get_search_engine()
        
        # Create controller with verified dependencies
        return VectorController(
            storage_manager=storage_manager,
            version_manager=version_manager,
            search_engine=search_engine,
            audit_logger=audit_logger
        )
    except Exception as e:
        logger.error(f"Failed to initialize vector controller: {e}")
        raise VectorControllerError(f"Vector controller initialization failed: {e}")