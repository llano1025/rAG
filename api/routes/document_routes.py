from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Form, status
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import io
import json

from database.connection import get_db
from api.middleware.auth import get_current_active_user
from api.controllers.document_controller import DocumentController, get_document_controller
from api.schemas.document_schemas import DocumentUpdate
from database.models import User, Document, DocumentShare

router = APIRouter(prefix="/documents", tags=["documents"])

class BatchDeleteRequest(BaseModel):
    document_ids: List[int]

class DocumentShareRequest(BaseModel):
    share_with_users: Optional[List[int]] = None
    make_public: Optional[bool] = None
    permissions: Optional[Dict[str, bool]] = None  # {"can_read": True, "can_write": False, etc.}
    share_message: Optional[str] = None
    expires_at: Optional[datetime] = None

class DocumentPermissionRequest(BaseModel):
    user_id: int
    permissions: List[str]  # read, write, delete

class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    status: str
    content_type: str
    file_size: int
    text_length: int
    chunks_count: int
    version: int
    created_at: str
    processed_at: Optional[str] = None

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    folder_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # JSON string of tags
    metadata: Optional[str] = Form(None),  # JSON string of metadata
    embedding_model: Optional[str] = Form(None),  # Embedding model ID
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Upload a new document with optional folder and tags."""
    try:
        # Parse tags and metadata if provided
        parsed_tags = []
        parsed_metadata = {}
        
        if tags:
            import json
            try:
                parsed_tags = json.loads(tags)
            except json.JSONDecodeError:
                parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        if metadata:
            import json
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                parsed_metadata = {}
        
        document = await controller.process_upload(
            file=file,
            user=current_user,
            folder_id=folder_id,
            tags=parsed_tags,
            metadata=parsed_metadata,
            embedding_model=embedding_model,
            db=db
        )
        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/batch-upload")
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    folder_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    embedding_model: Optional[str] = Form(None),  # Embedding model ID
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Upload multiple documents simultaneously."""
    try:
        # Parse tags if provided
        parsed_tags = []
        if tags:
            import json
            try:
                parsed_tags = json.loads(tags)
            except json.JSONDecodeError:
                parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        result = await controller.process_batch_upload(
            files=files,
            user=current_user,
            folder_id=folder_id,
            tags=parsed_tags,
            embedding_model=embedding_model,
            db=db
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{document_id}")
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Retrieve a specific document by ID."""
    document = await controller.get_document(
        document_id=document_id,
        user=current_user,
        db=db
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.put("/{document_id}")
async def update_document(
    document_id: int,
    update_data: DocumentUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Update document metadata."""
    document = await controller.update_document(
        document_id=document_id,
        update_data=update_data,
        user=current_user,
        db=db
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    hard_delete: bool = Query(False, description="Permanently delete the document"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Delete a specific document."""
    success = await controller.delete_document(
        document_id=document_id,
        user=current_user,
        hard_delete=hard_delete,
        db=db
    )
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
        "hard_delete": hard_delete
    }

@router.post("/batch-delete")
async def batch_delete_documents(
    request: BatchDeleteRequest,
    hard_delete: bool = Query(False, description="Permanently delete the documents"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Delete multiple documents in a single request."""
    result = await controller.batch_delete_documents(
        document_ids=request.document_ids,
        user=current_user,
        hard_delete=hard_delete,
        db=db
    )
    return result

@router.get("/{document_id}/versions")
async def get_document_versions(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Retrieve version history of a document."""
    versions = await controller.get_document_versions(
        document_id=document_id,
        user=current_user,
        db=db
    )
    return versions

@router.get("/")
async def list_documents(
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of documents to return"),
    include_public: bool = Query(True, description="Include public documents"),
    folder_id: Optional[str] = Query(None, description="Filter by folder ID"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    search: Optional[str] = Query(None, description="Search query for document content"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """List documents accessible to the current user."""
    # Parse tags if provided
    parsed_tags = []
    if tags:
        parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    
    result = await controller.list_user_documents(
        user=current_user,
        skip=skip,
        limit=limit,
        include_public=include_public,
        folder_id=folder_id,
        tags=parsed_tags if parsed_tags else None,
        search_query=search,
        db=db
    )
    return result

@router.get("/{document_id}/download")
async def download_document(
    document_id: int,
    format: str = Query("original", description="Download format: original, text, json"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Download a document in various formats."""
    try:
        # Get document data
        document = await controller.get_document(
            document_id=document_id,
            user=current_user,
            db=db
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        filename = document.get('filename', f'document_{document_id}')
        
        if format == "text":
            # Return extracted text content
            content = document.get('extracted_text', '')
            content_bytes = content.encode('utf-8')
            
            return StreamingResponse(
                io.BytesIO(content_bytes),
                media_type='text/plain',
                headers={
                    "Content-Disposition": f"attachment; filename={filename}.txt"
                }
            )
        
        elif format == "json":
            # Return full document metadata and content as JSON
            json_content = json.dumps(document, indent=2, default=str)
            content_bytes = json_content.encode('utf-8')
            
            return StreamingResponse(
                io.BytesIO(content_bytes),
                media_type='application/json',
                headers={
                    "Content-Disposition": f"attachment; filename={filename}.json"
                }
            )
        
        elif format == "original":
            # Note: For original file download, we would need to store the original file
            # For now, return the extracted text with original filename
            content = document.get('extracted_text', '')
            content_bytes = content.encode('utf-8')
            
            # Try to maintain original extension or default to .txt
            original_filename = filename
            if not any(filename.endswith(ext) for ext in ['.txt', '.pdf', '.docx', '.html']):
                original_filename += '.txt'
            
            return StreamingResponse(
                io.BytesIO(content_bytes),
                media_type='application/octet-stream',
                headers={
                    "Content-Disposition": f"attachment; filename={original_filename}"
                }
            )
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid format. Supported formats: original, text, json"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download document: {str(e)}"
        )

@router.get("/{document_id}/export")
async def export_document(
    document_id: int,
    include_chunks: bool = Query(False, description="Include document chunks in export"),
    include_vectors: bool = Query(False, description="Include vector information"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Export document with comprehensive metadata and analysis."""
    try:
        # Get document data
        document = await controller.get_document(
            document_id=document_id,
            user=current_user,
            db=db
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Build export data
        export_data = {
            "document_info": {
                "id": document.get('id'),
                "filename": document.get('filename'),
                "title": document.get('title'),
                "description": document.get('description'),
                "content_type": document.get('content_type'),
                "file_size": document.get('file_size'),
                "status": document.get('status'),
                "version": document.get('version'),
                "created_at": document.get('created_at'),
                "updated_at": document.get('updated_at'),
                "language": document.get('language'),
                "tags": document.get('tags', [])
            },
            "content": {
                "extracted_text": document.get('extracted_text', ''),
                "text_length": len(document.get('extracted_text', '')),
                "chunks_count": document.get('chunks_count', 0)
            }
        }
        
        if include_chunks:
            export_data["chunks"] = document.get('chunks', [])
        
        if include_vectors:
            export_data["vector_info"] = {
                "chunks_count": document.get('chunks_count', 0),
                "embedding_model": document.get('chunks', [{}])[0].get('embedding_model') if document.get('chunks') else None
            }
        
        # Add export metadata
        export_data["export_metadata"] = {
            "exported_by": current_user.username,
            "exported_at": datetime.now().isoformat(),
            "export_version": "1.0"
        }
        
        return export_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export document: {str(e)}"
        )

@router.get("/{document_id}/content")
async def get_document_content(
    document_id: int,
    chunk_id: Optional[str] = Query(None, description="Get specific chunk content"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Get document content, optionally filtered by chunk."""
    try:
        # Get document data
        document = await controller.get_document(
            document_id=document_id,
            user=current_user,
            db=db
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if chunk_id:
            # Return specific chunk
            chunks = document.get('chunks', [])
            for chunk in chunks:
                if chunk.get('chunk_id') == chunk_id:
                    return {
                        "chunk_id": chunk_id,
                        "chunk_index": chunk.get('chunk_index'),
                        "text": chunk.get('text'),
                        "text_length": chunk.get('text_length'),
                        "start_char": chunk.get('start_char'),
                        "end_char": chunk.get('end_char'),
                        "context_before": chunk.get('context_before'),
                        "context_after": chunk.get('context_after')
                    }
            
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        else:
            # Return full document content
            return {
                "document_id": document_id,
                "filename": document.get('filename'),
                "extracted_text": document.get('extracted_text', ''),
                "text_length": len(document.get('extracted_text', '')),
                "chunks_count": document.get('chunks_count', 0),
                "language": document.get('language'),
                "content_type": document.get('content_type')
            }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document content: {str(e)}"
        )

@router.post("/{document_id}/share")
async def share_document(
    document_id: int,
    share_request: DocumentShareRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Share a document with other users or make it public."""
    try:
        # Get document first to check ownership
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted == False
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user can share the document
        if not document.can_share(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to share this document"
            )
        
        sharing_results = {
            "document_id": document_id,
            "shared_with": [],
            "public_status_changed": False,
            "message": "Document sharing updated"
        }
        
        # Update public status if requested
        if share_request.make_public is not None:
            old_public_status = document.is_public
            document.is_public = share_request.make_public
            if old_public_status != share_request.make_public:
                sharing_results["public_status_changed"] = True
                sharing_results["is_public"] = share_request.make_public
        
        # Share with specific users
        if share_request.share_with_users:
            shared_users = []
            
            # Get default permissions
            default_permissions = {
                "can_read": True,
                "can_write": False,
                "can_delete": False,
                "can_share": False
            }
            
            # Override with requested permissions
            if share_request.permissions:
                default_permissions.update(share_request.permissions)
            
            for user_id in share_request.share_with_users:
                # Check if user exists
                target_user = db.query(User).filter(User.id == user_id, User.is_deleted == False).first()
                if not target_user:
                    continue
                
                # Check if share already exists
                existing_share = db.query(DocumentShare).filter(
                    DocumentShare.document_id == document_id,
                    DocumentShare.shared_with_user_id == user_id
                ).first()
                
                if existing_share:
                    # Update existing share
                    existing_share.set_permissions(default_permissions)
                    existing_share.share_message = share_request.share_message
                    existing_share.expires_at = share_request.expires_at
                    existing_share.is_active = True
                    existing_share.shared_by_user_id = current_user.id
                else:
                    # Create new share
                    new_share = DocumentShare(
                        document_id=document_id,
                        shared_with_user_id=user_id,
                        shared_by_user_id=current_user.id,
                        share_message=share_request.share_message,
                        expires_at=share_request.expires_at,
                        **default_permissions
                    )
                    db.add(new_share)
                
                shared_users.append({
                    "user_id": user_id,
                    "username": target_user.username,
                    "permissions": default_permissions
                })
            
            sharing_results["shared_with"] = shared_users
        
        db.commit()
        return sharing_results
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to share document: {str(e)}"
        )

@router.get("/{document_id}/permissions")
async def get_document_permissions(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get document sharing permissions and access information."""
    try:
        # Get document
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted == False
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user can view permissions
        if not document.can_access(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this document"
            )
        
        # Get owner information
        owner = db.query(User).filter(User.id == document.user_id).first()
        
        # Get document shares
        shares = db.query(DocumentShare).filter(
            DocumentShare.document_id == document_id,
            DocumentShare.is_active == True
        ).all()
        
        shared_with = []
        for share in shares:
            if share.is_valid():
                shared_user = db.query(User).filter(User.id == share.shared_with_user_id).first()
                if shared_user:
                    shared_with.append({
                        "user_id": shared_user.id,
                        "username": shared_user.username,
                        "email": shared_user.email,
                        "permissions": share.get_permissions(),
                        "shared_by": share.shared_by_user_id,
                        "share_message": share.share_message,
                        "expires_at": share.expires_at.isoformat() if share.expires_at else None,
                        "shared_at": share.created_at.isoformat()
                    })
        
        permissions_info = {
            "document_id": document_id,
            "owner": {
                "id": owner.id,
                "username": owner.username,
                "email": owner.email
            } if owner else None,
            "is_public": document.is_public,
            "current_user_permissions": {
                "can_read": document.can_access(current_user),
                "can_write": document.can_edit(current_user),
                "can_delete": document.can_delete(current_user),
                "can_share": document.can_share(current_user)
            },
            "shared_with": shared_with,
            "total_shares": len(shared_with)
        }
        
        return permissions_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document permissions: {str(e)}"
        )

@router.delete("/{document_id}/share/{user_id}")
async def unshare_document(
    document_id: int,
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Remove sharing access for a specific user."""
    try:
        # Get document
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted == False
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user can share the document
        if not document.can_share(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to unshare this document"
            )
        
        # Find the share record
        share = db.query(DocumentShare).filter(
            DocumentShare.document_id == document_id,
            DocumentShare.shared_with_user_id == user_id,
            DocumentShare.is_active == True
        ).first()
        
        if not share:
            raise HTTPException(
                status_code=404,
                detail="Document share not found"
            )
        
        # Remove the share (soft delete by setting is_active to False)
        share.is_active = False
        
        # Get shared user info for response
        shared_user = db.query(User).filter(User.id == user_id).first()
        
        db.commit()
        
        return {
            "message": f"Document unshared from user {shared_user.username if shared_user else user_id}",
            "document_id": document_id,
            "unshared_user_id": user_id,
            "unshared_username": shared_user.username if shared_user else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unshare document: {str(e)}"
        )

@router.get("/shared-with-me")
async def get_shared_documents(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """Get documents shared with current user."""
    try:
        # Get active shares for current user
        shares = db.query(DocumentShare).filter(
            DocumentShare.shared_with_user_id == current_user.id,
            DocumentShare.is_active == True
        ).offset(skip).limit(limit).all()
        
        shared_documents = []
        for share in shares:
            if share.is_valid():
                document = db.query(Document).filter(
                    Document.id == share.document_id,
                    Document.is_deleted == False
                ).first()
                
                if document:
                    owner = db.query(User).filter(User.id == document.user_id).first()
                    
                    shared_documents.append({
                        "document_id": document.id,
                        "filename": document.filename,
                        "title": document.title,
                        "content_type": document.content_type,
                        "file_size": document.file_size,
                        "created_at": document.created_at.isoformat(),
                        "owner": {
                            "id": owner.id,
                            "username": owner.username
                        } if owner else None,
                        "permissions": share.get_permissions(),
                        "shared_at": share.created_at.isoformat(),
                        "share_message": share.share_message,
                        "expires_at": share.expires_at.isoformat() if share.expires_at else None
                    })
        
        return {
            "documents": shared_documents,
            "total": len(shared_documents),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get shared documents: {str(e)}"
        )

@router.put("/{document_id}/share/{user_id}/permissions")
async def update_share_permissions(
    document_id: int,
    user_id: int,
    permission_request: DocumentPermissionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update permissions for a specific document share."""
    try:
        # Get document
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted == False
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user can share the document
        if not document.can_share(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to modify document shares"
            )
        
        # Find the share record
        share = db.query(DocumentShare).filter(
            DocumentShare.document_id == document_id,
            DocumentShare.shared_with_user_id == user_id,
            DocumentShare.is_active == True
        ).first()
        
        if not share:
            raise HTTPException(
                status_code=404,
                detail="Document share not found"
            )
        
        # Convert permission list to dictionary
        permissions = {
            "can_read": "read" in permission_request.permissions,
            "can_write": "write" in permission_request.permissions,
            "can_delete": "delete" in permission_request.permissions,
            "can_share": "share" in permission_request.permissions
        }
        
        # Update permissions
        share.set_permissions(permissions)
        
        db.commit()
        
        return {
            "message": "Share permissions updated successfully",
            "document_id": document_id,
            "user_id": user_id,
            "permissions": permissions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update share permissions: {str(e)}"
        )

@router.post("/{document_id}/public")
async def make_document_public(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Make a document publicly accessible."""
    try:
        # Get document
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted == False
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user owns the document or has admin privileges
        if document.user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only document owner or admin can make documents public"
            )
        
        document.is_public = True
        db.commit()
        
        return {
            "message": "Document is now public",
            "document_id": document_id,
            "is_public": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to make document public: {str(e)}"
        )

@router.delete("/{document_id}/public")
async def make_document_private(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Make a document private (remove public access)."""
    try:
        # Get document
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted == False
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user owns the document or has admin privileges
        if document.user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only document owner or admin can make documents private"
            )
        
        document.is_public = False
        db.commit()
        
        return {
            "message": "Document is now private",
            "document_id": document_id,
            "is_public": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to make document private: {str(e)}"
        )