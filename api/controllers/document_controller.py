"""
Document controller for handling document CRUD operations.
Integrates file processing, vector operations, and database management.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime, timezone
from fastapi import UploadFile, status
from sqlalchemy.orm import Session
from io import BytesIO

from database.models import User, Document, DocumentStatusEnum
from api.controllers.vector_controller import VectorController, get_vector_controller
from file_processor.type_detector import FileTypeDetector
from file_processor.text_extractor import TextExtractor
from file_processor.metadata_extractor import MetadataExtractor
from utils.security.audit_logger import AuditLogger
from api.schemas.document_schemas import DocumentUpdate, Document, DocumentCreate
from utils.exceptions import (
    DocumentNotFoundException,
    DocumentProcessingException,
    InvalidFileTypeException,
    FileTooLargeException,
    DocumentAccessDeniedException,
    DuplicateResourceException,
    ExternalServiceException,
    ConfigurationException
)
from api.schemas.responses import StandardResponse, create_success_response, create_paginated_response

logger = logging.getLogger(__name__)

# Removed DocumentProcessingError - using unified exception system

class DocumentController:
    """
    Document controller for handling document CRUD operations.
    
    Handles:
    - Document upload and processing
    - Document retrieval and export
    - Document metadata management
    - Document versioning
    - Batch operations
    """
    
    def __init__(
        self,
        vector_controller: VectorController = None,
        audit_logger: AuditLogger = None
    ):
        """Initialize document controller with dependencies."""
        logger.info("DocumentController.__init__: Starting DocumentController initialization...")
        logger.info(f"DocumentController.__init__: vector_controller provided: {vector_controller is not None}")
        logger.info(f"DocumentController.__init__: audit_logger provided: {audit_logger is not None}")
        
        self.vector_controller = vector_controller or get_vector_controller(audit_logger=audit_logger)
        self.audit_logger = audit_logger
        
        # File processing components
        logger.info("DocumentController: Initializing file processing components...")
        try:
            self.type_detector = FileTypeDetector()
            logger.info("DocumentController: FileTypeDetector initialized successfully")
        except Exception as e:
            logger.error(f"DocumentController: Failed to initialize FileTypeDetector: {e}")
            raise
            
        try:
            self.text_extractor = TextExtractor()
            logger.info("DocumentController: TextExtractor initialized successfully")
        except Exception as e:
            logger.error(f"DocumentController: Failed to initialize TextExtractor: {e}")
            raise
            
        try:
            self.metadata_extractor = MetadataExtractor()
            logger.info("DocumentController: MetadataExtractor initialized successfully")
        except Exception as e:
            logger.error(f"DocumentController: Failed to initialize MetadataExtractor: {e}")
            raise
        
        # Configuration
        logger.info("DocumentController.__init__: Setting up configuration...")
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        # Ensure we have comprehensive MIME type support
        self.allowed_content_types = [
            'text/plain',
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/html',
            'application/json',
            'text/markdown',
            'text/csv',
            # Image types for OCR processing - comprehensive list
            'image/jpeg',
            'image/jpg',  # Common MIME type variation
            'image/png',
            'image/tiff',
            'image/tif',  # Common MIME type variation
            'image/gif',
            # Additional variations that might be detected
            'image/pjpeg',  # Progressive JPEG
            'image/x-png',  # Alternative PNG
        ]
        
        # Log the final configuration
        logger.info(f"DocumentController.__init__: Max file size: {self.max_file_size / (1024*1024)}MB")
        logger.info(f"DocumentController.__init__: Allowed content types ({len(self.allowed_content_types)} total):")
        for i, content_type in enumerate(self.allowed_content_types, 1):
            logger.info(f"DocumentController.__init__:   {i:2d}. {content_type}")
        logger.info(f"DocumentController.__init__: 'image/jpeg' in allowed types: {'image/jpeg' in self.allowed_content_types}")
        logger.info("DocumentController.__init__: DocumentController initialization completed successfully!")
    
    async def process_upload(
        self,
        file: UploadFile,
        user: User,
        folder_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[str] = None,
        ocr_method: Optional[str] = None,
        ocr_language: Optional[str] = None,
        vision_provider: Optional[str] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Process a single document upload.
        
        Args:
            file: Uploaded file
            user: User uploading the document
            folder_id: Optional folder ID
            tags: Optional list of tags
            metadata: Optional additional metadata
            embedding_model: Optional embedding model to use
            ocr_method: Optional OCR method ('tesseract' or 'vision_llm')
            ocr_language: Optional OCR language code
            vision_provider: Optional vision provider ('openai', 'gemini', 'claude')
            db: Database session
            
        Returns:
            Dictionary with upload results
        """
        try:
            logger.info(f"DocumentController: Starting file validation for {file.filename}")
            
            # Validate file
            try:
                await self._validate_file(file)
                logger.info(f"DocumentController: File validation successful for {file.filename}")
            except Exception as ve:
                logger.error(f"DocumentController: File validation failed for {file.filename}")
                logger.error(f"DocumentController: Validation error type: {type(ve).__name__}")
                logger.error(f"DocumentController: Validation error message: {str(ve)}")
                logger.exception(f"DocumentController: Full validation exception traceback:")
                raise  # Re-raise the original exception
            
            # Check if vector controller is available
            if self.vector_controller is None:
                logger.error("Vector controller not available for document processing")
                raise ExternalServiceException(
                    message="Document processing service is currently unavailable",
                    service_name="vector_controller"
                )
            
            # Read file content
            file_content = await file.read()
            await file.seek(0)  # Reset file pointer
            
            # Prepare metadata with OCR settings for image files
            enhanced_metadata = metadata or {}
            if file.content_type and file.content_type.startswith('image/'):
                enhanced_metadata.update({
                    'ocr_method': ocr_method or 'tesseract',
                    'ocr_language': ocr_language or 'eng',
                    'vision_provider': vision_provider if ocr_method == 'vision_llm' else None,
                    'requires_ocr': True
                })
            
            # Process upload using vector controller
            result = await self.vector_controller.upload_document(
                file_content=file_content,
                filename=file.filename,
                user=user,
                metadata=enhanced_metadata,
                embedding_model=embedding_model,
                tags=tags,
                db=db
            )
            
            # Log successful upload
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="user_action",
                    user_id=str(user.id),
                    action="document_uploaded",
                    resource_type="document",
                    resource_id=str(result['document_id']),
                    details={
                        'filename': file.filename,
                        'file_size': result['file_size'],
                        'content_type': result['content_type']
                    }
                )
            
            return result
            
        except (DocumentProcessingException, InvalidFileTypeException, FileTooLargeException, 
                ExternalServiceException) as e:
            # Re-raise our unified exceptions
            raise
        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            raise DocumentProcessingException(
                message="Document upload failed",
                processing_stage="upload",
                context={"error": str(e)}
            )
    
    async def process_batch_upload(
        self,
        files: List[UploadFile],
        user: User,
        folder_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        embedding_model: Optional[str] = None,
        ocr_method: Optional[str] = None,
        ocr_language: Optional[str] = None,
        vision_provider: Optional[str] = None,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple document uploads simultaneously.
        
        Args:
            files: List of uploaded files
            user: User uploading the documents
            folder_id: Optional folder ID
            tags: Optional list of tags
            db: Database session
            
        Returns:
            List of upload results
        """
        try:
            if len(files) > 10:  # Limit batch size
                raise DocumentProcessingException(
                    message="Batch upload limited to 10 files maximum",
                    processing_stage="batch_validation",
                    context={"file_count": len(files), "max_allowed": 10}
                )
            
            # Process uploads concurrently
            tasks = []
            for file in files:
                task = self.process_upload(
                    file=file,
                    user=user,
                    folder_id=folder_id,
                    tags=tags,
                    embedding_model=embedding_model,
                    ocr_method=ocr_method,
                    ocr_language=ocr_language,
                    vision_provider=vision_provider,
                    db=db
                )
                tasks.append(task)
            
            # Wait for all uploads to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle any exceptions
            successful_uploads = []
            failed_uploads = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_uploads.append({
                        'filename': files[i].filename,
                        'error': str(result)
                    })
                else:
                    successful_uploads.append(result)
            
            # Log batch upload results
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="user_action",
                    user_id=str(user.id),
                    action="batch_documents_uploaded",
                    resource_type="batch_upload",
                    details={
                        'total_files': len(files),
                        'successful_uploads': len(successful_uploads),
                        'failed_uploads': len(failed_uploads),
                        'failed_files': [f['filename'] for f in failed_uploads]
                    }
                )
            
            return create_success_response(
                data={
                    'successful_uploads': successful_uploads,
                    'failed_uploads': failed_uploads,
                    'summary': {
                        'total': len(files),
                        'successful': len(successful_uploads),
                        'failed': len(failed_uploads)
                    }
                },
                message=f"Batch upload completed: {len(successful_uploads)} successful, {len(failed_uploads)} failed"
            )
            
        except (DocumentProcessingException, InvalidFileTypeException, FileTooLargeException) as e:
            raise
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            raise DocumentProcessingException(
                message="Batch upload operation failed",
                processing_stage="batch_processing",
                context={"error": str(e)}
            )
    
    async def get_document(
        self,
        document_id: int,
        user: User,
        db: Session = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: Document ID
            user: User requesting the document
            db: Database session
            
        Returns:
            Document data or None if not found
        """
        try:
            # Use vector controller to get document
            document_data = await self.vector_controller.get_document(
                document_id=document_id,
                user=user,
                db=db
            )
            
            return document_data
            
        except (DocumentNotFoundException, DocumentAccessDeniedException) as e:
            raise
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise DocumentProcessingException(
                message="Failed to retrieve document", 
                document_id=str(document_id),
                processing_stage="retrieval",
                context={"error": str(e)}
            )
    
    async def update_document(
        self,
        document_id: int,
        update_data: DocumentUpdate,
        user: User,
        db: Session = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update document metadata.
        
        Args:
            document_id: Document ID
            update_data: Update data
            user: User updating the document
            db: Database session
            
        Returns:
            Updated document data
        """
        try:
            # Get existing document first
            document = db.query(Document).filter(
                Document.id == document_id,
                Document.is_deleted == False
            ).first()
            
            if not document:
                raise DocumentNotFoundException(
                    message="Document not found",
                    document_id=str(document_id)
                )
            
            # Check permissions
            if not document.can_access(user):
                raise DocumentAccessDeniedException(
                    message="Access denied to this document",
                    document_id=str(document_id)
                )
            
            # Check if user can edit this document
            if document.user_id != user.id and not user.is_superuser:
                raise DocumentAccessDeniedException(
                    message="Permission denied: cannot edit this document",
                    document_id=str(document_id),
                    context={"required_permission": "edit"}
                )
            
            # Update fields
            update_fields = {}
            if update_data.title is not None:
                document.title = update_data.title
                update_fields['title'] = update_data.title
            
            if update_data.description is not None:
                document.description = update_data.description
                update_fields['description'] = update_data.description
            
            if update_data.tags is not None:
                document.set_tag_list(update_data.tags)
                update_fields['tags'] = update_data.tags
            
            if update_data.metadata is not None:
                # Merge with existing metadata
                current_metadata = document.get_metadata_dict()
                current_metadata.update(update_data.metadata)
                document.set_metadata(current_metadata)
                update_fields['metadata'] = update_data.metadata
            
            document.updated_at = datetime.now(timezone.utc)
            
            db.commit()
            
            # Log document update
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="user_action",
                    user_id=str(user.id),
                    action="document_updated",
                    resource_type="document",
                    resource_id=str(document_id),
                    details={
                        'updated_fields': list(update_fields.keys()),
                        'changes': update_fields
                    }
                )
            
            # Return updated document data
            return await self.get_document(document_id, user, db)
            
        except (DocumentNotFoundException, DocumentAccessDeniedException) as e:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update document {document_id}: {e}")
            raise DocumentProcessingException(
                message="Document update failed",
                document_id=str(document_id), 
                processing_stage="update",
                context={"error": str(e)}
            )
    
    async def delete_document(
        self,
        document_id: int,
        user: User,
        hard_delete: bool = False,
        db: Session = None
    ) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: Document ID
            user: User deleting the document
            hard_delete: Whether to permanently delete
            db: Database session
            
        Returns:
            True if successful
        """
        try:
            # Use vector controller to delete document
            result = await self.vector_controller.delete_document(
                document_id=document_id,
                user=user,
                hard_delete=hard_delete,
                db=db
            )
            
            return result.get('deleted', False)
            
        except (DocumentNotFoundException, DocumentAccessDeniedException) as e:
            raise
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise DocumentProcessingException(
                message="Document deletion failed",
                document_id=str(document_id),
                processing_stage="deletion",
                context={"error": str(e)}
            )
    
    async def batch_delete_documents(
        self,
        document_ids: List[int],
        user: User,
        hard_delete: bool = False,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Delete multiple documents.
        
        Args:
            document_ids: List of document IDs
            user: User deleting the documents
            hard_delete: Whether to permanently delete
            db: Database session
            
        Returns:
            Deletion results
        """
        try:
            if len(document_ids) > 50:  # Limit batch size
                raise DocumentProcessingException(
                    message="Batch delete limited to 50 documents maximum",
                    processing_stage="batch_validation",
                    context={"document_count": len(document_ids), "max_allowed": 50}
                )
            
            successful_deletes = []
            failed_deletes = []
            
            for doc_id in document_ids:
                try:
                    success = await self.delete_document(
                        document_id=doc_id,
                        user=user,
                        hard_delete=hard_delete,
                        db=db
                    )
                    if success:
                        successful_deletes.append(doc_id)
                    else:
                        failed_deletes.append({'id': doc_id, 'error': 'Delete failed'})
                except Exception as e:
                    failed_deletes.append({'id': doc_id, 'error': str(e)})
            
            # Log batch delete
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="user_action",
                    user_id=str(user.id),
                    action="batch_documents_deleted",
                    resource_type="batch_delete",
                    details={
                        'total_documents': len(document_ids),
                        'successful_deletes': len(successful_deletes),
                        'failed_deletes': len(failed_deletes),
                        'hard_delete': hard_delete
                    }
                )
            
            return create_success_response(
                data={
                    'successful_deletes': successful_deletes,
                    'failed_deletes': failed_deletes,
                    'summary': {
                        'total': len(document_ids),
                        'deleted': len(successful_deletes),
                        'failed': len(failed_deletes)
                    }
                },
                message=f"Batch delete completed: {len(successful_deletes)} deleted, {len(failed_deletes)} failed"
            )
            
        except DocumentProcessingException as e:
            raise
        except Exception as e:
            logger.error(f"Batch delete failed: {e}")
            raise DocumentProcessingException(
                message="Batch delete operation failed",
                processing_stage="batch_delete",
                context={"error": str(e)}
            )
    
    async def get_document_versions(
        self,
        document_id: int,
        user: User,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """
        Get version history of a document.
        
        Args:
            document_id: Document ID
            user: User requesting versions
            db: Database session
            
        Returns:
            List of document versions
        """
        try:
            # Get document first to check permissions
            document = db.query(Document).filter(
                Document.id == document_id,
                Document.is_deleted == False
            ).first()
            
            if not document:
                raise DocumentNotFoundException(
                    message="Document not found",
                    document_id=str(document_id)
                )
            
            # Check permissions
            if not document.can_access(user):
                raise DocumentAccessDeniedException(
                    message="Access denied to this document",
                    document_id=str(document_id)
                )
            
            # Get all versions from vector controller's version manager
            from vector_db.document_version_manager import get_version_manager
            version_manager = get_version_manager()
            
            versions = version_manager.get_document_versions(
                document_id=document_id,
                db=db,
                include_deleted=False
            )
            
            # Format versions
            version_list = []
            for version in versions:
                version_data = {
                    'id': version.id,
                    'version': version.version,
                    'filename': version.filename,
                    'created_at': version.created_at.isoformat(),
                    'file_size': version.file_size,
                    'status': version.status.value,
                    'chunks_count': len(version.chunks),
                    'content_length': len(version.extracted_text) if version.extracted_text else 0
                }
                version_list.append(version_data)
            
            return version_list
            
        except (DocumentNotFoundException, DocumentAccessDeniedException) as e:
            raise
        except Exception as e:
            logger.error(f"Failed to get document versions {document_id}: {e}")
            raise DocumentProcessingException(
                message="Failed to retrieve document versions",
                document_id=str(document_id),
                processing_stage="version_retrieval",
                context={"error": str(e)}
            )
    
    async def list_user_documents(
        self,
        user: User,
        skip: int = 0,
        limit: int = 50,
        include_public: bool = True,
        folder_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search_query: Optional[str] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        List documents accessible to a user.
        
        Args:
            user: User requesting the list
            skip: Number of documents to skip
            limit: Maximum number of documents
            include_public: Whether to include public documents
            folder_id: Filter by folder ID
            tags: Filter by tags
            search_query: Filter by search query
            db: Database session
            
        Returns:
            Dictionary with documents list and metadata
        """
        try:
            # Use vector controller to list documents
            result = await self.vector_controller.list_user_documents(
                user=user,
                skip=skip,
                limit=limit,
                include_public=include_public,
                db=db
            )
            
            # Apply additional filters if provided
            documents = result.get('documents', [])
            
            if folder_id:
                documents = [doc for doc in documents if doc.get('folder_id') == folder_id]
            
            if tags:
                documents = [
                    doc for doc in documents 
                    if any(tag in doc.get('tags', []) for tag in tags)
                ]
            
            if search_query:
                query_lower = search_query.lower()
                documents = [
                    doc for doc in documents
                    if query_lower in doc.get('filename', '').lower() or
                       query_lower in doc.get('title', '').lower() or
                       query_lower in doc.get('description', '').lower()
                ]
            
            return create_success_response(
                data={
                    "documents": documents,
                    "total_count": len(documents),
                    "skip": skip,
                    "limit": limit,
                    "filters": {
                        "folder_id": folder_id,
                        "tags": tags,
                        "search_query": search_query
                    }
                },
                message="Documents retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise DocumentProcessingException(
                message="Failed to list documents",
                processing_stage="document_listing",
                context={"error": str(e)}
            )
    
    async def _validate_file(self, file: UploadFile):
        """Validate uploaded file."""
        logger.info(f"_validate_file: Starting validation for {file.filename}")
        logger.info(f"_validate_file: File content_type from FastAPI: {file.content_type}")
        
        # Check file size
        logger.info(f"_validate_file: Reading file content...")
        file_content = await file.read()
        await file.seek(0)  # Reset file pointer
        logger.info(f"_validate_file: File content read - Size: {len(file_content)} bytes")
        
        if len(file_content) > self.max_file_size:
            logger.error(f"_validate_file: File size check failed - {len(file_content)} > {self.max_file_size}")
            raise FileTooLargeException(
                message=f"File size exceeds {self.max_file_size / (1024*1024)}MB limit",
                file_size=len(file_content),
                max_size=self.max_file_size
            )
        
        logger.info(f"_validate_file: File size check passed")
        logger.info(f"_validate_file: About to call type_detector.detect_type...")
        logger.info(f"_validate_file: TypeDetector instance: {self.type_detector}")
        
        # Check content type
        try:
            content_type = self.type_detector.detect_type(file_content=file_content, filename=file.filename)
            logger.info(f"_validate_file: Type detection successful - Detected: {content_type}")
            logger.info(f"_validate_file: Content type type: {type(content_type)}")
            logger.info(f"_validate_file: Content type repr: {repr(content_type)}")
        except Exception as te:
            logger.error(f"_validate_file: Type detection failed")
            logger.error(f"_validate_file: Type detection error type: {type(te).__name__}")
            logger.error(f"_validate_file: Type detection error message: {str(te)}")
            logger.exception(f"_validate_file: Full type detection exception traceback:")
            
            # Convert UnsupportedFileType to unified exception
            from file_processor.type_detector import UnsupportedFileType
            if isinstance(te, UnsupportedFileType):
                raise InvalidFileTypeException(
                    message=str(te),
                    file_type=getattr(te, 'file_type', None)
                )
            else:
                raise  # Re-raise other exceptions
        
        # Debug logging for file type validation
        logger.info(f"File upload validation - Filename: {file.filename}")
        logger.info(f"File upload validation - Detected MIME type: {content_type}")
        logger.info(f"File upload validation - File size: {len(file_content)} bytes")
        logger.info(f"File upload validation - Allowed content types: {self.allowed_content_types}")
        logger.info(f"File upload validation - Number of allowed types: {len(self.allowed_content_types)}")
        logger.info(f"File upload validation - MIME type in allowed list: {content_type in self.allowed_content_types}")
        
        # Additional debugging - check each allowed type
        logger.info("File upload validation - Checking each allowed type:")
        for i, allowed_type in enumerate(self.allowed_content_types):
            matches = content_type == allowed_type
            logger.info(f"  {i+1:2d}. {repr(allowed_type)} == {repr(content_type)}: {matches}")
        
        # Check if the content type is in allowed types, with fallback for image files
        is_allowed = content_type in self.allowed_content_types
        
        # Fallback: If it's an image type but not in allowed types, check if it's a reasonable image type
        if not is_allowed and content_type and content_type.startswith('image/'):
            # Check if this is a reasonable image extension that we should support
            filename_lower = file.filename.lower() if file.filename else ''
            if any(filename_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif']):
                logger.warning(f"Allowing image file with MIME type '{content_type}' based on filename '{file.filename}'")
                is_allowed = True
                # Add this MIME type to allowed types for future uploads
                self.allowed_content_types.append(content_type)
        
        if not is_allowed:
            # Enhanced error logging for debugging
            logger.error(f"DocumentController validation failed - Content type '{content_type}' not in allowed types")
            logger.error(f"Available allowed_content_types: {self.allowed_content_types}")
            logger.error(f"Content type '{content_type}' in allowed list: {content_type in self.allowed_content_types}")
            logger.error(f"Type of content_type: {type(content_type)}")
            logger.error(f"Repr of content_type: {repr(content_type)}")
            
            # Get supported file types for error message
            supported_formats = []
            for mime_type in self.allowed_content_types:
                if mime_type.startswith('text/'):
                    supported_formats.append('Text files')
                elif mime_type.startswith('application/pdf'):
                    supported_formats.append('PDF')
                elif 'word' in mime_type:
                    supported_formats.append('Word documents')
                elif mime_type.startswith('image/'):
                    supported_formats.append('Images (PNG, JPG, GIF, TIFF)')
                elif mime_type.startswith('text/html'):
                    supported_formats.append('HTML')
                elif mime_type.startswith('application/json'):
                    supported_formats.append('JSON')
                elif mime_type.startswith('text/markdown'):
                    supported_formats.append('Markdown')
                elif mime_type.startswith('text/csv'):
                    supported_formats.append('CSV')
            
            unique_formats = list(set(supported_formats))
            supported_text = ', '.join(unique_formats)
            
            raise InvalidFileTypeException(
                message=f"Unsupported file type: {content_type}",
                file_type=content_type,
                supported_types=unique_formats
            )


# Dependency injection
def get_audit_logger() -> AuditLogger:
    """Get audit logger instance."""
    from utils.security.audit_logger import AuditLoggerConfig
    from pathlib import Path
    
    config = AuditLoggerConfig(
        log_path=Path("data/audit_logs"),
        rotation_size_mb=10,
        retention_days=90
    )
    # Ensure audit log directory exists
    Path("data/audit_logs").mkdir(parents=True, exist_ok=True)
    return AuditLogger(config)

def get_document_controller() -> DocumentController:
    """Get document controller instance with dependencies."""
    logger.info("get_document_controller: Creating new DocumentController instance...")
    try:
        # Get audit logger
        logger.info("get_document_controller: Getting audit logger...")
        audit_logger = get_audit_logger()
        
        # Get vector controller with proper initialization
        logger.info("get_document_controller: Getting vector controller...")
        vector_controller = get_vector_controller(audit_logger=audit_logger)
        
        logger.info("get_document_controller: Creating DocumentController with dependencies...")
        controller = DocumentController(
            vector_controller=vector_controller,
            audit_logger=audit_logger
        )
        logger.info("get_document_controller: DocumentController created successfully")
        return controller
    except Exception as e:
        logger.error(f"get_document_controller: Failed to initialize document controller: {e}")
        logger.exception("get_document_controller: Full exception traceback:")
        # Return a minimal controller without vector functionality
        logger.info("get_document_controller: Creating minimal DocumentController fallback...")
        return DocumentController(audit_logger=get_audit_logger())