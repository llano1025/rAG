# api/controllers/library_controller.py
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from database.models import Document, User, DocumentChunk
from utils.security.encryption import EncryptionManager
from utils.security.audit_logger import AuditLogger
from utils.security.pii_detector import PIIDetector, PIIConfig

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
                tags = document.get_tag_list()
                for tag in tags:
                    if tag in tag_counts:
                        tag_counts[tag] += 1
                    else:
                        tag_counts[tag] = 1
            
            # Convert to list format
            tag_list = []
            for tag_name, count in tag_counts.items():
                tag_list.append({
                    "tag_name": tag_name,
                    "document_count": count
                })
            
            return sorted(tag_list, key=lambda x: x["document_count"], reverse=True)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

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
        user_id: str
    ) -> dict:
        """Legacy method - kept for compatibility."""
        try:
            # Check for PII in folder data
            pii_check = self.pii_detector.detect_pii(str(folder_data))
            if pii_check:
                folder_data = self.pii_detector.redact_pii(str(folder_data))
            
            # Encrypt sensitive data
            encrypted_data = self.encryption.encrypt_data(folder_data)
            
            # Perform operation
            result = await self._execute_folder_operation(operation, encrypted_data)
            
            # Audit logging
            await self.audit_logger.log(
                action=operation,
                resource_type="folder",
                user_id=user_id,
                resource_id=result.get("id"),
                details={"pii_detected": bool(pii_check)}
            )
            
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def _execute_folder_operation(self, operation: str, encrypted_data: dict) -> dict:
        """Execute folder operation based on operation type."""
        try:
            if operation == "create":
                return {
                    "id": f"folder_{datetime.utcnow().timestamp()}",
                    "operation": "create",
                    "status": "success"
                }
            elif operation == "update":
                return {
                    "id": encrypted_data.get("id", "unknown"),
                    "operation": "update", 
                    "status": "success"
                }
            elif operation == "delete":
                return {
                    "id": encrypted_data.get("id", "unknown"),
                    "operation": "delete",
                    "status": "success"
                }
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Operation failed: {str(e)}")


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