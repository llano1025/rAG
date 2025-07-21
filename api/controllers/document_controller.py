"""
Document controller for handling document CRUD operations.
Integrates file processing, vector operations, and database management.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime, timezone
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from io import BytesIO

from database.models import User, Document, DocumentStatusEnum
from api.controllers.vector_controller import VectorController, get_vector_controller
from file_processor.type_detector import FileTypeDetector
from file_processor.text_extractor import TextExtractor
from file_processor.metadata_extractor import MetadataExtractor
from utils.security.audit_logger import AuditLogger
from api.schemas.document_schemas import DocumentUpdate, Document, DocumentCreate

logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass

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
        self.vector_controller = vector_controller or get_vector_controller()
        self.audit_logger = audit_logger
        
        # File processing components
        self.type_detector = FileTypeDetector()
        self.text_extractor = TextExtractor()
        self.metadata_extractor = MetadataExtractor()
        
        # Configuration
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.allowed_content_types = [
            'text/plain',
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/html',
            'application/json',
            'text/markdown',
            'text/csv'
        ]
    
    async def process_upload(
        self,
        file: UploadFile,
        user: User,
        folder_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
            db: Database session
            
        Returns:
            Dictionary with upload results
        """
        try:
            # Validate file
            await self._validate_file(file)
            
            # Read file content
            file_content = await file.read()
            await file.seek(0)  # Reset file pointer
            
            # Process upload using vector controller
            result = await self.vector_controller.upload_document(
                file_content=file_content,
                filename=file.filename,
                user=user,
                metadata=metadata or {},
                db=db
            )
            
            # Log successful upload
            if self.audit_logger:
                await self.audit_logger.log(
                    action="document_uploaded",
                    resource_type="document",
                    resource_id=str(result['document_id']),
                    user_id=user.id,
                    details={
                        'filename': file.filename,
                        'file_size': result['file_size'],
                        'content_type': result['content_type']
                    }
                )
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document upload failed"
            )
    
    async def process_batch_upload(
        self,
        files: List[UploadFile],
        user: User,
        folder_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
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
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Batch upload limited to 10 files maximum"
                )
            
            # Process uploads concurrently
            tasks = []
            for file in files:
                task = self.process_upload(
                    file=file,
                    user=user,
                    folder_id=folder_id,
                    tags=tags,
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
                await self.audit_logger.log(
                    action="batch_documents_uploaded",
                    resource_type="batch_upload",
                    user_id=user.id,
                    details={
                        'total_files': len(files),
                        'successful_uploads': len(successful_uploads),
                        'failed_uploads': len(failed_uploads),
                        'failed_files': [f['filename'] for f in failed_uploads]
                    }
                )
            
            return {
                'successful_uploads': successful_uploads,
                'failed_uploads': failed_uploads,
                'summary': {
                    'total': len(files),
                    'successful': len(successful_uploads),
                    'failed': len(failed_uploads)
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch upload failed"
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
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve document"
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
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
            
            # Check permissions
            if not document.can_access(user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this document"
                )
            
            # Check if user can edit this document
            if document.user_id != user.id and not user.has_permission("edit_documents"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission denied: cannot edit this document"
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
                document.set_tags(update_data.tags)
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
                await self.audit_logger.log(
                    action="document_updated",
                    resource_type="document",
                    resource_id=str(document_id),
                    user_id=user.id,
                    details={
                        'updated_fields': list(update_fields.keys()),
                        'changes': update_fields
                    }
                )
            
            # Return updated document data
            return await self.get_document(document_id, user, db)
            
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update document {document_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document update failed"
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
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document deletion failed"
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
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Batch delete limited to 50 documents maximum"
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
                await self.audit_logger.log(
                    action="batch_documents_deleted",
                    resource_type="batch_delete",
                    user_id=user.id,
                    details={
                        'total_documents': len(document_ids),
                        'successful_deletes': len(successful_deletes),
                        'failed_deletes': len(failed_deletes),
                        'hard_delete': hard_delete
                    }
                )
            
            return {
                'successful_deletes': successful_deletes,
                'failed_deletes': failed_deletes,
                'summary': {
                    'total': len(document_ids),
                    'deleted': len(successful_deletes),
                    'failed': len(failed_deletes)
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch delete failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch delete failed"
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
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
            
            # Check permissions
            if not document.can_access(user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this document"
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
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get document versions {document_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve document versions"
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
            
            return {
                'documents': documents,
                'total_count': len(documents),
                'skip': skip,
                'limit': limit,
                'filters': {
                    'folder_id': folder_id,
                    'tags': tags,
                    'search_query': search_query,
                    'include_public': include_public
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list documents"
            )
    
    async def _validate_file(self, file: UploadFile):
        """Validate uploaded file."""
        # Check file size
        file_content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        if len(file_content) > self.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds {self.max_file_size / (1024*1024)}MB limit"
            )
        
        # Check content type
        content_type = self.type_detector.detect_type(file_content, file.filename)
        if content_type not in self.allowed_content_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type '{content_type}' is not supported"
            )


# Dependency injection
def get_audit_logger() -> AuditLogger:
    """Get audit logger instance."""
    return AuditLogger()

def get_document_controller() -> DocumentController:
    """Get document controller instance with dependencies."""
    return DocumentController(audit_logger=get_audit_logger())