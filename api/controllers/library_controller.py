# api/controllers/library_controller.py
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import logging

from database.models import Document, User, DocumentChunk
from utils.security.encryption import EncryptionManager
from utils.security.audit_logger import AuditLogger
from utils.security.pii_detector import PIIDetector, PIIConfig

logger = logging.getLogger(__name__)

class LibraryController:
    """Controller for document library management and organization."""
    
    def __init__(
        self,
        encryption: EncryptionManager,
        audit_logger: AuditLogger,
        pii_detector: PIIDetector
    ):
        self.encryption = encryption
        self.audit_logger = audit_logger
        self.pii_detector = pii_detector

    async def create_folder(
        self,
        folder_path: str,
        user: User,
        db: Session,
        description: str = None
    ) -> dict:
        """Create a new folder for document organization."""
        try:
            # Validate folder path
            if not folder_path or folder_path.startswith('/'):
                raise ValueError("Invalid folder path")
            
            # Check if folder already exists for user
            existing_folder = db.query(Document).filter(
                Document.user_id == user.id,
                Document.folder_path == folder_path,
                Document.is_deleted == False
            ).first()
            
            if existing_folder:
                raise HTTPException(status_code=409, detail="Folder already exists")
            
            # Audit log
            await self.audit_logger.log(
                action="create_folder",
                resource_type="folder",
                user_id=str(user.id),
                resource_id=folder_path,
                details={"folder_path": folder_path, "description": description}
            )
            
            return {
                "folder_path": folder_path,
                "description": description,
                "created_at": datetime.utcnow().isoformat(),
                "status": "created"
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def list_folders(
        self,
        user: User,
        db: Session,
        parent_path: str = None
    ) -> List[Dict[str, Any]]:
        """List all folders for a user."""
        try:
            # Get all unique folder paths for user's documents
            query = db.query(
                Document.folder_path,
                func.count(Document.id).label('document_count'),
                func.max(Document.created_at).label('last_updated')
            ).filter(
                Document.user_id == user.id,
                Document.is_deleted == False,
                Document.folder_path.isnot(None)
            )
            
            if parent_path:
                query = query.filter(Document.folder_path.like(f"{parent_path}%"))
            
            folders = query.group_by(Document.folder_path).all()
            
            folder_list = []
            for folder in folders:
                folder_list.append({
                    "folder_path": folder.folder_path,
                    "document_count": folder.document_count,
                    "last_updated": folder.last_updated.isoformat() if folder.last_updated else None
                })
            
            return folder_list
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def move_documents_to_folder(
        self,
        document_ids: List[int],
        folder_path: str,
        user: User,
        db: Session
    ) -> dict:
        """Move documents to a specific folder."""
        try:
            # Validate user owns all documents
            documents = db.query(Document).filter(
                Document.id.in_(document_ids),
                Document.user_id == user.id,
                Document.is_deleted == False
            ).all()
            
            if len(documents) != len(document_ids):
                raise HTTPException(status_code=404, detail="Some documents not found")
            
            # Update folder path for all documents
            moved_count = 0
            for document in documents:
                document.folder_path = folder_path
                moved_count += 1
            
            db.commit()
            
            # Audit log
            await self.audit_logger.log(
                action="move_documents",
                resource_type="document_batch",
                user_id=str(user.id),
                resource_id=f"batch_{len(document_ids)}",
                details={
                    "document_ids": document_ids,
                    "target_folder": folder_path,
                    "moved_count": moved_count
                }
            )
            
            return {
                "moved_count": moved_count,
                "target_folder": folder_path,
                "status": "success"
            }
            
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail=str(e))

    async def create_tag(
        self,
        tag_name: str,
        user: User,
        db: Session,
        description: str = None,
        color: str = None
    ) -> dict:
        """Create a new tag for document organization."""
        try:
            # Validate tag name
            if not tag_name or len(tag_name.strip()) == 0:
                raise ValueError("Tag name cannot be empty")
            
            tag_name = tag_name.strip().lower()
            
            # Check if user already has documents with this tag
            existing_docs = db.query(Document).filter(
                Document.user_id == user.id,
                Document.tags.like(f'%"{tag_name}"%'),
                Document.is_deleted == False
            ).first()
            
            # Audit log
            await self.audit_logger.log(
                action="create_tag",
                resource_type="tag",
                user_id=str(user.id),
                resource_id=tag_name,
                details={
                    "tag_name": tag_name,
                    "description": description,
                    "color": color,
                    "exists": bool(existing_docs)
                }
            )
            
            return {
                "tag_name": tag_name,
                "description": description,
                "color": color,
                "created_at": datetime.utcnow().isoformat(),
                "status": "created" if not existing_docs else "already_exists"
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def list_tags(
        self,
        user: User,
        db: Session
    ) -> List[Dict[str, Any]]:
        """List all tags used by user's documents."""
        try:
            # Get all documents with tags
            documents = db.query(Document).filter(
                Document.user_id == user.id,
                Document.is_deleted == False,
                Document.tags.isnot(None)
            ).all()
            
            # Extract and count tags
            tag_counts = {}
            for document in documents:
                try:
                    tags = document.get_tag_list()
                    for tag in tags:
                        if tag and isinstance(tag, str):
                            tag = tag.strip().lower()
                            if tag:  # Only count non-empty tags
                                if tag in tag_counts:
                                    tag_counts[tag] += 1
                                else:
                                    tag_counts[tag] = 1
                except Exception as e:
                    logger.warning(f"Failed to extract tags from document {document.id}: {e}")
                    continue
            
            # Convert to list format
            tag_list = []
            for tag_name, count in tag_counts.items():
                tag_list.append({
                    "tag_name": tag_name,
                    "document_count": count
                })
            
            return sorted(tag_list, key=lambda x: x["document_count"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list tags for user {user.id}: {str(e)}")
            # Return empty list instead of raising exception to avoid breaking the frontend
            return []

    async def add_tags_to_documents(
        self,
        document_ids: List[int],
        tags: List[str],
        user: User,
        db: Session
    ) -> dict:
        """Add tags to multiple documents."""
        try:
            # Validate user owns all documents
            documents = db.query(Document).filter(
                Document.id.in_(document_ids),
                Document.user_id == user.id,
                Document.is_deleted == False
            ).all()
            
            if len(documents) != len(document_ids):
                raise HTTPException(status_code=404, detail="Some documents not found")
            
            # Clean and validate tags
            clean_tags = [tag.strip().lower() for tag in tags if tag.strip()]
            if not clean_tags:
                raise ValueError("No valid tags provided")
            
            # Add tags to documents
            updated_count = 0
            for document in documents:
                existing_tags = set(document.get_tag_list())
                new_tags = existing_tags.union(set(clean_tags))
                document.set_tags(list(new_tags))
                updated_count += 1
            
            db.commit()
            
            # Audit log
            await self.audit_logger.log(
                action="add_tags",
                resource_type="document_batch",
                user_id=str(user.id),
                resource_id=f"batch_{len(document_ids)}",
                details={
                    "document_ids": document_ids,
                    "tags": clean_tags,
                    "updated_count": updated_count
                }
            )
            
            return {
                "updated_count": updated_count,
                "tags_added": clean_tags,
                "status": "success"
            }
            
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail=str(e))

    async def remove_tags_from_documents(
        self,
        document_ids: List[int],
        tags: List[str],
        user: User,
        db: Session
    ) -> dict:
        """Remove tags from multiple documents."""
        try:
            # Validate user owns all documents
            documents = db.query(Document).filter(
                Document.id.in_(document_ids),
                Document.user_id == user.id,
                Document.is_deleted == False
            ).all()
            
            if len(documents) != len(document_ids):
                raise HTTPException(status_code=404, detail="Some documents not found")
            
            # Clean tags to remove
            tags_to_remove = set(tag.strip().lower() for tag in tags if tag.strip())
            if not tags_to_remove:
                raise ValueError("No valid tags provided")
            
            # Remove tags from documents
            updated_count = 0
            for document in documents:
                existing_tags = set(document.get_tag_list())
                new_tags = existing_tags - tags_to_remove
                document.set_tags(list(new_tags))
                updated_count += 1
            
            db.commit()
            
            # Audit log
            await self.audit_logger.log(
                action="remove_tags",
                resource_type="document_batch",
                user_id=str(user.id),
                resource_id=f"batch_{len(document_ids)}",
                details={
                    "document_ids": document_ids,
                    "tags_removed": list(tags_to_remove),
                    "updated_count": updated_count
                }
            )
            
            return {
                "updated_count": updated_count,
                "tags_removed": list(tags_to_remove),
                "status": "success"
            }
            
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail=str(e))

    async def bulk_add_tags(
        self,
        document_ids: List[int],
        tags: List[str],
        user: User,
        db: Session
    ) -> dict:
        """Add tags to multiple documents (bulk operation)."""
        try:
            successful = 0
            failed = 0
            errors = []
            
            # Validate documents exist and user has access
            documents = db.query(Document).filter(
                Document.id.in_(document_ids),
                Document.user_id == user.id,
                Document.is_deleted == False
            ).all()
            
            found_doc_ids = {doc.id for doc in documents}
            missing_doc_ids = set(document_ids) - found_doc_ids
            
            if missing_doc_ids:
                errors.append(f"Documents not found: {list(missing_doc_ids)}")
                failed += len(missing_doc_ids)
            
            # Clean and validate tags
            clean_tags = [tag.strip().lower() for tag in tags if tag.strip()]
            if not clean_tags:
                raise ValueError("No valid tags provided")
            
            # Add tags to each document
            for document in documents:
                try:
                    existing_tags = set(document.get_tag_list())
                    new_tags = existing_tags.union(set(clean_tags))
                    document.set_tag_list(list(new_tags))
                    successful += 1
                except Exception as e:
                    errors.append(f"Failed to add tags to document {document.id}: {str(e)}")
                    failed += 1
            
            if successful > 0:
                db.commit()
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="user_action",
                    user_id=str(user.id),
                    action="bulk_add_tags",
                    resource_type="document_batch",
                    details={
                        "document_ids": document_ids,
                        "tags": clean_tags,
                        "successful": successful,
                        "failed": failed
                    }
                )
            
            return {
                "successful": successful,
                "failed": failed,
                "errors": errors,
                "tags_added": clean_tags
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Bulk add tags failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    async def bulk_remove_tags(
        self,
        document_ids: List[int],
        tags: List[str],
        user: User,
        db: Session
    ) -> dict:
        """Remove tags from multiple documents (bulk operation)."""
        try:
            successful = 0
            failed = 0
            errors = []
            
            # Validate documents exist and user has access
            documents = db.query(Document).filter(
                Document.id.in_(document_ids),
                Document.user_id == user.id,
                Document.is_deleted == False
            ).all()
            
            found_doc_ids = {doc.id for doc in documents}
            missing_doc_ids = set(document_ids) - found_doc_ids
            
            if missing_doc_ids:
                errors.append(f"Documents not found: {list(missing_doc_ids)}")
                failed += len(missing_doc_ids)
            
            # Clean tags to remove
            tags_to_remove = set(tag.strip().lower() for tag in tags if tag.strip())
            if not tags_to_remove:
                raise ValueError("No valid tags provided")
            
            # Remove tags from each document
            for document in documents:
                try:
                    existing_tags = set(document.get_tag_list())
                    new_tags = existing_tags - tags_to_remove
                    document.set_tag_list(list(new_tags))
                    successful += 1
                except Exception as e:
                    errors.append(f"Failed to remove tags from document {document.id}: {str(e)}")
                    failed += 1
            
            if successful > 0:
                db.commit()
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="user_action",
                    user_id=str(user.id),
                    action="bulk_remove_tags",
                    resource_type="document_batch",
                    details={
                        "document_ids": document_ids,
                        "tags": list(tags_to_remove),
                        "successful": successful,
                        "failed": failed
                    }
                )
            
            return {
                "successful": successful,
                "failed": failed,
                "errors": errors,
                "tags_removed": list(tags_to_remove)
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Bulk remove tags failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    async def apply_tags_to_documents(
        self,
        document_ids: List[int],
        tags: List[str],
        user: User,
        db: Session
    ) -> dict:
        """Apply/set tags to documents (replaces existing tags)."""
        try:
            successful = 0
            failed = 0
            errors = []
            
            # Validate documents exist and user has access
            documents = db.query(Document).filter(
                Document.id.in_(document_ids),
                Document.user_id == user.id,
                Document.is_deleted == False
            ).all()
            
            found_doc_ids = {doc.id for doc in documents}
            missing_doc_ids = set(document_ids) - found_doc_ids
            
            if missing_doc_ids:
                errors.append(f"Documents not found: {list(missing_doc_ids)}")
                failed += len(missing_doc_ids)
            
            # Clean and validate tags
            clean_tags = [tag.strip().lower() for tag in tags if tag.strip()]
            
            # Apply tags to each document (replace existing)
            for document in documents:
                try:
                    document.set_tag_list(clean_tags)
                    successful += 1
                except Exception as e:
                    errors.append(f"Failed to apply tags to document {document.id}: {str(e)}")
                    failed += 1
            
            if successful > 0:
                db.commit()
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="user_action",
                    user_id=str(user.id),
                    action="apply_tags",
                    resource_type="document_batch",
                    details={
                        "document_ids": document_ids,
                        "tags": clean_tags,
                        "successful": successful,
                        "failed": failed
                    }
                )
            
            return {
                "successful": successful,
                "failed": failed,
                "errors": errors,
                "tags_applied": clean_tags
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Apply tags failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    async def organize_documents_by_content_type(
        self,
        user: User,
        db: Session
    ) -> dict:
        """Automatically organize documents into folders by content type."""
        try:
            # Get all user documents without folder organization
            documents = db.query(Document).filter(
                Document.user_id == user.id,
                Document.is_deleted == False,
                or_(
                    Document.folder_path.is_(None),
                    Document.folder_path == ""
                )
            ).all()
            
            if not documents:
                return {"organized_count": 0, "status": "no_documents"}
            
            # Organize by content type
            content_type_mapping = {
                "application/pdf": "PDFs",
                "text/plain": "Text Files",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word Documents",
                "application/msword": "Word Documents",
                "text/html": "HTML Files",
                "image/": "Images"
            }
            
            organized_count = 0
            organization_summary = {}
            
            for document in documents:
                folder_path = None
                
                # Find appropriate folder based on content type
                for content_type, folder_name in content_type_mapping.items():
                    if document.content_type.startswith(content_type):
                        folder_path = folder_name
                        break
                
                if not folder_path:
                    folder_path = "Other Files"
                
                # Update document folder path
                document.folder_path = folder_path
                organized_count += 1
                
                # Track organization summary
                if folder_path in organization_summary:
                    organization_summary[folder_path] += 1
                else:
                    organization_summary[folder_path] = 1
            
            db.commit()
            
            # Audit log
            await self.audit_logger.log(
                action="auto_organize",
                resource_type="document_batch",
                user_id=str(user.id),
                resource_id=f"auto_organize_{organized_count}",
                details={
                    "organized_count": organized_count,
                    "organization_summary": organization_summary
                }
            )
            
            return {
                "organized_count": organized_count,
                "organization_summary": organization_summary,
                "status": "success"
            }
            
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail=str(e))

    async def get_library_statistics(
        self,
        user: User,
        db: Session
    ) -> dict:
        """Get comprehensive library statistics for user."""
        try:
            # Basic document counts
            total_documents = db.query(Document).filter(
                Document.user_id == user.id,
                Document.is_deleted == False
            ).count()
            
            # Documents by status
            status_counts = db.query(
                Document.status,
                func.count(Document.id)
            ).filter(
                Document.user_id == user.id,
                Document.is_deleted == False
            ).group_by(Document.status).all()
            
            # Documents by content type
            content_type_counts = db.query(
                Document.content_type,
                func.count(Document.id)
            ).filter(
                Document.user_id == user.id,
                Document.is_deleted == False
            ).group_by(Document.content_type).all()
            
            # Folder statistics
            folder_counts = db.query(
                Document.folder_path,
                func.count(Document.id)
            ).filter(
                Document.user_id == user.id,
                Document.is_deleted == False,
                Document.folder_path.isnot(None)
            ).group_by(Document.folder_path).all()
            
            # Tag statistics
            documents_with_tags = db.query(Document).filter(
                Document.user_id == user.id,
                Document.is_deleted == False,
                Document.tags.isnot(None)
            ).all()
            
            tag_counts = {}
            for doc in documents_with_tags:
                tags = doc.get_tag_list()
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # File size statistics
            size_stats = db.query(
                func.sum(Document.file_size),
                func.avg(Document.file_size),
                func.max(Document.file_size),
                func.min(Document.file_size)
            ).filter(
                Document.user_id == user.id,
                Document.is_deleted == False
            ).first()
            
            return {
                "total_documents": total_documents,
                "status_breakdown": {status: count for status, count in status_counts},
                "content_type_breakdown": {content_type: count for content_type, count in content_type_counts},
                "folder_breakdown": {folder or "Uncategorized": count for folder, count in folder_counts},
                "tag_breakdown": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                "file_size_stats": {
                    "total_bytes": size_stats[0] or 0,
                    "average_bytes": int(size_stats[1] or 0),
                    "largest_bytes": size_stats[2] or 0,
                    "smallest_bytes": size_stats[3] or 0
                } if any(size_stats) else {},
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def manage_folder(
        self,
        operation: str,
        folder_data: dict,
        user_id: str,
        db: Session = None
    ) -> dict:
        """Manage folder operations - create, update, delete folders."""
        try:
            from database.connection import get_db
            
            if db is None:
                db = next(get_db())
            
            # Get user object
            user = db.query(User).filter(User.id == int(user_id)).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Check for PII in folder data
            pii_check = self.pii_detector.detect_pii(str(folder_data))
            if pii_check:
                # Redact PII from folder data
                sensitive_fields = ['description', 'name']
                for field in sensitive_fields:
                    if field in folder_data:
                        folder_data[field] = self.pii_detector.redact_pii(folder_data[field])
            
            # Perform operation based on type
            if operation == "create":
                folder_path = folder_data.get("folder_path") or folder_data.get("name")
                description = folder_data.get("description")
                
                if not folder_path:
                    raise ValueError("Folder path/name is required for create operation")
                
                result = await self.create_folder(
                    folder_path=folder_path,
                    user=user,
                    db=db,
                    description=description
                )
                
            elif operation == "update":
                # For folder updates, we need to move documents to new folder path
                old_folder = folder_data.get("old_folder_path")
                new_folder = folder_data.get("new_folder_path") or folder_data.get("folder_path")
                
                if not old_folder or not new_folder:
                    raise ValueError("Both old and new folder paths are required for update operation")
                
                # Get all documents in the old folder
                documents_in_folder = db.query(Document).filter(
                    Document.user_id == user.id,
                    Document.folder_path == old_folder,
                    Document.is_deleted == False
                ).all()
                
                document_ids = [doc.id for doc in documents_in_folder]
                
                if document_ids:
                    result = await self.move_documents_to_folder(
                        document_ids=document_ids,
                        folder_path=new_folder,
                        user=user,
                        db=db
                    )
                    result.update({
                        "operation": "update",
                        "old_folder": old_folder,
                        "new_folder": new_folder
                    })
                else:
                    result = {
                        "operation": "update",
                        "old_folder": old_folder,
                        "new_folder": new_folder,
                        "moved_count": 0,
                        "status": "success"
                    }
                
            elif operation == "delete":
                folder_path = folder_data.get("folder_path") or folder_data.get("name")
                force_delete = folder_data.get("force_delete", False)
                
                if not folder_path:
                    raise ValueError("Folder path is required for delete operation")
                
                # Check if folder has documents
                documents_in_folder = db.query(Document).filter(
                    Document.user_id == user.id,
                    Document.folder_path == folder_path,
                    Document.is_deleted == False
                ).count()
                
                if documents_in_folder > 0 and not force_delete:
                    raise ValueError(f"Folder '{folder_path}' contains {documents_in_folder} documents. Use force_delete=true to delete anyway.")
                
                # Remove folder path from all documents in folder
                if documents_in_folder > 0:
                    db.query(Document).filter(
                        Document.user_id == user.id,
                        Document.folder_path == folder_path,
                        Document.is_deleted == False
                    ).update({"folder_path": None})
                    db.commit()
                
                result = {
                    "operation": "delete",
                    "folder_path": folder_path,
                    "documents_moved": documents_in_folder,
                    "status": "success"
                }
                
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Audit logging
            self.audit_logger.log_event(
                event_type="user_action",
                user_id=user_id,
                action=f"folder_{operation}",
                resource_type="folder",
                resource_id=folder_data.get("folder_path", "unknown"),
                details={
                    "operation": operation,
                    "folder_data": folder_data,
                    "pii_detected": bool(pii_check),
                    "result": result
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Folder management operation '{operation}' failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    async def _execute_folder_operation(self, operation: str, encrypted_data: dict) -> dict:
        """Execute folder operation based on operation type - legacy method."""
        # This method is deprecated in favor of the main manage_folder method
        # Kept for backward compatibility
        logger.warning("_execute_folder_operation is deprecated, use manage_folder instead")
        
        try:
            # Decrypt the data
            folder_data = self.encryption.decrypt_data(encrypted_data)
            
            # Basic validation and return format for compatibility
            if operation == "create":
                folder_name = folder_data.get("name", folder_data.get("folder_path", "unknown"))
                return {
                    "id": f"folder_{datetime.utcnow().timestamp()}",
                    "name": folder_name,
                    "operation": "create",
                    "status": "success"
                }
            elif operation == "update":
                return {
                    "id": folder_data.get("id", "unknown"),
                    "operation": "update", 
                    "status": "success"
                }
            elif operation == "delete":
                return {
                    "id": folder_data.get("id", "unknown"),
                    "operation": "delete",
                    "status": "success"
                }
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"Legacy folder operation failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Operation failed: {str(e)}")

    async def get_folder_hierarchy(
        self,
        user: User,
        db: Session,
        root_path: str = None
    ) -> Dict[str, Any]:
        """Get folder hierarchy as a tree structure."""
        try:
            # Get all folders for user
            folders = await self.list_folders(user, db, parent_path=root_path)
            
            # Build hierarchy tree
            hierarchy = {}
            for folder in folders:
                folder_path = folder["folder_path"]
                parts = folder_path.split("/")
                
                current_level = hierarchy
                for part in parts:
                    if part not in current_level:
                        current_level[part] = {
                            "name": part,
                            "path": "/".join(parts[:parts.index(part)+1]),
                            "document_count": 0,
                            "children": {}
                        }
                    current_level = current_level[part]["children"]
                
                # Set document count for the leaf folder
                if parts:
                    leaf_path = parts
                    current_level = hierarchy
                    for part in leaf_path[:-1]:
                        current_level = current_level[part]["children"]
                    if leaf_path[-1] in current_level:
                        current_level[leaf_path[-1]]["document_count"] = folder["document_count"]
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Failed to get folder hierarchy: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    async def bulk_folder_operations(
        self,
        operations: List[Dict[str, Any]],
        user: User,
        db: Session
    ) -> Dict[str, Any]:
        """Perform multiple folder operations in batch."""
        try:
            results = []
            successful_operations = 0
            failed_operations = 0
            
            for operation in operations:
                try:
                    op_type = operation.get("operation")
                    folder_data = operation.get("data", {})
                    
                    result = await self.manage_folder(
                        operation=op_type,
                        folder_data=folder_data,
                        user_id=str(user.id),
                        db=db
                    )
                    
                    results.append({
                        "operation": operation,
                        "result": result,
                        "status": "success"
                    })
                    successful_operations += 1
                    
                except Exception as e:
                    results.append({
                        "operation": operation,
                        "error": str(e),
                        "status": "failed"
                    })
                    failed_operations += 1
            
            # Audit log
            self.audit_logger.log_event(
                event_type="user_action",
                user_id=str(user.id),
                action="bulk_folder_operations",
                resource_type="folder_batch",
                details={
                    "total_operations": len(operations),
                    "successful": successful_operations,
                    "failed": failed_operations
                }
            )
            
            return {
                "total_operations": len(operations),
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Bulk folder operations failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))


# Dependency injection
def get_library_controller() -> LibraryController:
    """Get library controller instance with dependencies."""
    # Import global dependency functions from main.py
    from main import get_encryption_manager, get_audit_logger
    
    # Create PII detector with default configuration
    pii_config = PIIConfig()
    pii_detector = PIIDetector(pii_config)
    
    return LibraryController(
        encryption=get_encryption_manager(),
        audit_logger=get_audit_logger(),
        pii_detector=pii_detector
    )