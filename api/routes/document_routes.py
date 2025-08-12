from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Form, status
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import io
import json
import mimetypes
from pathlib import Path
from utils.file_storage import get_file_manager

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
    ocr_method: Optional[str] = Form(None),  # OCR method for images
    ocr_language: Optional[str] = Form(None),  # OCR language
    vision_provider: Optional[str] = Form(None),  # Vision provider for OCR
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Upload a new document with optional folder and tags."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== UPLOAD ROUTE START ===")
    logger.info(f"Route: Upload document - Filename: {file.filename}")
    logger.info(f"Route: File content type: {file.content_type}")
    logger.info(f"Route: File size: {file.size if hasattr(file, 'size') else 'unknown'}")
    logger.info(f"Route: OCR method: {ocr_method}, OCR language: {ocr_language}, Vision provider: {vision_provider}")
    
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
        
        logger.info(f"Route: Calling controller.process_upload...")
        
        document = await controller.process_upload(
            file=file,
            user=current_user,
            folder_id=folder_id,
            tags=parsed_tags,
            metadata=parsed_metadata,
            embedding_model=embedding_model,
            ocr_method=ocr_method,
            ocr_language=ocr_language,
            vision_provider=vision_provider,
            db=db
        )
        
        logger.info(f"Route: Upload successful - Document ID: {document.get('document_id', 'unknown')}")
        return document
    except HTTPException as he:
        logger.error(f"Route: HTTPException caught - Status: {he.status_code}, Detail: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Route: Generic exception caught - Type: {type(e).__name__}, Message: {str(e)}")
        logger.exception(f"Route: Full exception traceback:")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/batch-upload")
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    folder_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    embedding_model: Optional[str] = Form(None),  # Embedding model ID
    ocr_method: Optional[str] = Form(None),  # OCR method for images
    ocr_language: Optional[str] = Form(None),  # OCR language
    vision_provider: Optional[str] = Form(None),  # Vision provider for OCR
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Upload multiple documents simultaneously."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== BATCH UPLOAD ROUTE START ===")
    logger.info(f"Route: Batch upload - Number of files: {len(files)}")
    logger.info(f"Route: First file name: {files[0].filename if files else 'None'}")
    logger.info(f"Route: OCR method: {ocr_method}, OCR language: {ocr_language}, Vision provider: {vision_provider}")
    
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
            ocr_method=ocr_method,
            ocr_language=ocr_language,
            vision_provider=vision_provider,
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
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔄 Download request - Document ID: {document_id}, Format: {format}, User: {current_user.username}")
    
    try:
        # Get document data
        logger.info(f"📄 Fetching document data for ID: {document_id}")
        document = await controller.get_document(
            document_id=document_id,
            user=current_user,
            db=db
        )
        
        if not document:
            logger.error(f"❌ Document not found - ID: {document_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        filename = document.get('filename', f'document_{document_id}')
        logger.info(f"📋 Document found - Filename: {filename}, Content-Type: {document.get('content_type')}, Size: {document.get('file_size')}")
        
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
            logger.info(f"📁 Processing original file request")
            # Serve the actual original file
            file_path = document.get('file_path')
            logger.info(f"🗂️ File path from document: {file_path}")
            
            if not file_path:
                content_type = document.get('content_type', '')
                logger.error(f"❌ No file_path found in document record")
                logger.error(f"Document: {filename}, Content-Type: {content_type}")
                logger.error(f"Document record keys: {list(document.keys()) if isinstance(document, dict) else 'Not a dict'}")
                
                # For binary files (images, PDFs), don't fall back to extracted text
                if content_type and (content_type.startswith('image/') or 
                                   content_type == 'application/pdf' or
                                   content_type.startswith('application/vnd.') or
                                   content_type == 'application/msword'):
                    logger.error(f"💥 Binary file missing from storage - cannot serve extracted text as {content_type}")
                    
                    # Provide helpful error message with recovery guidance
                    file_type = "image" if content_type.startswith('image/') else "document"
                    error_detail = {
                        "error": f"Original {file_type} file missing from storage",
                        "message": f"The original {content_type} file for '{filename}' was not properly stored and cannot be displayed.",
                        "suggestions": [
                            f"Re-upload the original {filename} file",
                            "Contact administrator if this issue persists with new uploads", 
                            "Check system storage configuration",
                            "Try using the OCR Preview feature which may still have the file"
                        ],
                        "document_id": document_id,
                        "filename": filename,
                        "content_type": content_type,
                        "has_extracted_text": bool(document.get('extracted_text')),
                        "file_path_missing": True,
                        "recovery_action": "re-upload"
                    }
                    
                    raise HTTPException(
                        status_code=404,
                        detail=error_detail
                    )
                
                logger.warning(f"⚠️ Text-based file missing from storage - using fallback to extracted text")
                # Fallback to extracted text only for text-based files
                content = document.get('extracted_text', '')
                if not content:
                    raise HTTPException(
                        status_code=404,
                        detail="Neither original file nor extracted text is available"
                    )
                    
                content_bytes = content.encode('utf-8')
                
                return StreamingResponse(
                    io.BytesIO(content_bytes),
                    media_type='text/plain',
                    headers={
                        "Content-Disposition": f"inline; filename={filename}.txt"
                    }
                )
            
            # Get file manager and retrieve original file
            logger.info(f"🔧 Getting file manager instance")
            file_manager = get_file_manager()
            
            # Check if file exists
            logger.info(f"🔍 Checking if file exists: {file_path}")
            file_exists = file_manager.file_exists(file_path)
            logger.info(f"📂 File exists check result: {file_exists}")
            
            if not file_exists:
                logger.error(f"❌ Original file not found in storage - Path: {file_path}")
                logger.error(f"File manager storage root: {file_manager.storage_root.absolute()}")
                logger.error(f"Full file path would be: {file_manager.storage_root / file_path}")
                
                # Check if the file exists with different casing or location
                storage_files = []
                try:
                    import os
                    for root, dirs, files in os.walk(file_manager.storage_root):
                        for file in files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, file_manager.storage_root)
                            storage_files.append(rel_path)
                            if len(storage_files) >= 10:  # Limit to first 10 files
                                break
                        if len(storage_files) >= 10:
                            break
                    logger.info(f"📂 Found {len(storage_files)} files in storage: {storage_files[:5]}...")
                except Exception as scan_err:
                    logger.warning(f"Could not scan storage directory: {scan_err}")
                
                # Provide detailed error for file not found
                content_type = document.get('content_type', '')
                file_type = "image" if content_type.startswith('image/') else "document"
                error_detail = {
                    "error": f"Original {file_type} file not found in storage",
                    "message": f"The {content_type} file '{filename}' exists in the database but the actual file is missing from storage.",
                    "suggestions": [
                        f"Re-upload the original {filename} file",
                        "Check if storage directory was moved or changed",
                        "Contact administrator for file recovery"
                    ],
                    "document_id": document_id,
                    "filename": filename,
                    "content_type": content_type,
                    "expected_path": file_path,
                    "storage_root": str(file_manager.storage_root.absolute()),
                    "files_in_storage": len(storage_files) if 'storage_files' in locals() else 0,
                    "recovery_action": "re-upload"
                }
                
                raise HTTPException(
                    status_code=404,
                    detail=error_detail
                )
            
            # Get file content
            logger.info(f"📖 Opening file stream for: {file_path}")
            file_stream = file_manager.get_file_stream(file_path)
            
            if not file_stream:
                logger.error(f"❌ Failed to open file stream - Path: {file_path}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to open original file: {file_path}"
                )
            
            # Get file size for validation
            file_size = file_manager.get_file_size(file_path)
            logger.info(f"📏 File size: {file_size} bytes")
            
            # Determine MIME type
            content_type = document.get('content_type', 'application/octet-stream')
            logger.info(f"📝 Content type: {content_type}")
            
            # For preview in browser (PDFs, images), use inline disposition
            # For downloads, use attachment
            if content_type in ['application/pdf'] or content_type.startswith('image/'):
                disposition = f"inline; filename={filename}"
                logger.info(f"🖼️ Using inline disposition for preview: {disposition}")
            else:
                disposition = f"attachment; filename={filename}"
                logger.info(f"📎 Using attachment disposition for download: {disposition}")
            
            # Prepare headers
            headers = {
                "Content-Disposition": disposition,
                "Content-Length": str(file_size or 0),
                "Cache-Control": "no-cache"  # Prevent caching issues
            }
            
            logger.info(f"📨 Response headers: {headers}")
            logger.info(f"🎯 Creating StreamingResponse with media_type: {content_type}")
            
            return StreamingResponse(
                file_stream,
                media_type=content_type,
                headers=headers
            )
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid format. Supported formats: original, text, json"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions (they already have proper error handling)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in download endpoint - Document ID: {document_id}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.exception("Full exception traceback:")
        
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

@router.get("/debug/storage")
async def debug_storage_system(
    current_user: User = Depends(get_current_active_user)
):
    """Debug endpoint to test file storage system health."""
    import os
    from pathlib import Path
    
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        file_manager = get_file_manager()
        
        # Test basic storage info
        storage_root = file_manager.storage_root
        storage_info = {
            "storage_root": str(storage_root.absolute()),
            "exists": storage_root.exists(),
            "is_dir": storage_root.is_dir() if storage_root.exists() else False,
            "readable": os.access(storage_root, os.R_OK) if storage_root.exists() else False,
            "writable": os.access(storage_root, os.W_OK) if storage_root.exists() else False,
            "executable": os.access(storage_root, os.X_OK) if storage_root.exists() else False,
        }
        
        # Test file operations
        test_results = {
            "directory_creation": False,
            "file_write": False,
            "file_read": False,
            "file_delete": False,
            "error": None
        }
        
        try:
            # Test writing a small file
            test_content = b"File storage test content"
            test_filename = "storage_test.txt"
            
            file_path = file_manager.save_file(test_content, test_filename)
            test_results["file_write"] = True
            
            # Test reading the file back
            if file_manager.file_exists(file_path):
                file_stream = file_manager.get_file_stream(file_path)
                if file_stream:
                    read_content = file_stream.read()
                    file_stream.close()
                    if read_content == test_content:
                        test_results["file_read"] = True
            
            # Clean up test file
            full_path = storage_root / file_path
            if full_path.exists():
                full_path.unlink()
                test_results["file_delete"] = True
                
        except Exception as test_err:
            test_results["error"] = str(test_err)
        
        return {
            "storage_info": storage_info,
            "test_results": test_results,
            "status": "healthy" if all([
                storage_info["exists"],
                storage_info["is_dir"], 
                storage_info["writable"],
                test_results["file_write"],
                test_results["file_read"]
            ]) else "unhealthy"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "storage_info": None,
            "test_results": None
        }

@router.get("/debug/missing-files")
async def find_documents_missing_files(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Find documents that are missing their original files."""
    
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Query documents with missing file paths for binary file types
        binary_content_types = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/tiff', 
            'image/webp', 'image/bmp', 'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword'
        ]
        
        # Find documents with binary content types but no file_path
        missing_files = db.query(Document).filter(
            Document.content_type.in_(binary_content_types),
            (Document.file_path.is_(None) | (Document.file_path == "")),
            Document.is_deleted == False
        ).all()
        
        results = []
        for doc in missing_files:
            results.append({
                "id": doc.id,
                "filename": doc.filename,
                "title": doc.title or doc.filename,
                "content_type": doc.content_type,
                "file_size": doc.file_size,
                "user_id": doc.user_id,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "status": doc.status.value if doc.status else None,
                "file_path": doc.file_path,
                "has_extracted_text": bool(doc.extracted_text)
            })
        
        # Also check for documents with file_path but files don't exist on disk
        file_manager = get_file_manager()
        documents_with_paths = db.query(Document).filter(
            Document.content_type.in_(binary_content_types),
            Document.file_path.isnot(None),
            Document.file_path != "",
            Document.is_deleted == False
        ).all()
        
        missing_on_disk = []
        for doc in documents_with_paths:
            if not file_manager.file_exists(doc.file_path):
                missing_on_disk.append({
                    "id": doc.id,
                    "filename": doc.filename,
                    "title": doc.title or doc.filename,
                    "content_type": doc.content_type,
                    "file_path": doc.file_path,
                    "user_id": doc.user_id,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                })
        
        return {
            "missing_file_path": {
                "count": len(results),
                "documents": results
            },
            "missing_on_disk": {
                "count": len(missing_on_disk),
                "documents": missing_on_disk
            },
            "total_affected": len(results) + len(missing_on_disk),
            "binary_content_types_checked": binary_content_types
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "missing_file_path": None,
            "missing_on_disk": None,
            "total_affected": 0
        }

@router.post("/{document_id}/reprocess-ocr")
async def reprocess_document_ocr(
    document_id: int,
    ocr_method: str = Form("tesseract"),
    ocr_language: str = Form("eng"),
    vision_provider: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Reprocess a document with improved OCR settings."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔄 OCR reprocess request for document {document_id} by user {current_user.username}")
    logger.info(f"🔧 OCR settings - Method: {ocr_method}, Language: {ocr_language}, Provider: {vision_provider}")
    
    try:
        # Get the document
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted == False
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        # Check permissions
        if not document.can_edit(current_user):
            raise HTTPException(status_code=403, detail="Permission denied")
            
        # Ensure this is an image document
        if not document.content_type or not document.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="OCR reprocessing is only available for image documents"
            )
            
        # Check if original file exists
        if not document.file_path:
            raise HTTPException(
                status_code=400, 
                detail="Cannot reprocess OCR: original file path missing. Use file recovery first."
            )
            
        file_manager = get_file_manager()
        if not file_manager.file_exists(document.file_path):
            raise HTTPException(
                status_code=404, 
                detail="Cannot reprocess OCR: original file not found in storage"
            )
            
        logger.info(f"📄 Reprocessing OCR for document: {document.filename}")
        
        # Read the original file
        file_content = file_manager.get_file(document.file_path)
        if not file_content:
            raise HTTPException(
                status_code=500, 
                detail="Failed to read original file for OCR reprocessing"
            )
            
        # Initialize text extractor with new OCR settings
        from file_processor.text_extractor import TextExtractor
        text_extractor = TextExtractor()
        
        # Extract text with new settings
        new_extracted_text = await text_extractor.extract_text(
            file_content=file_content,
            content_type=document.content_type,
            filename=document.filename,
            ocr_method=ocr_method,
            ocr_language=ocr_language,
            vision_provider=vision_provider
        )
        
        if not new_extracted_text or len(new_extracted_text.strip()) < 10:
            raise HTTPException(
                status_code=422,
                detail="OCR reprocessing failed to extract meaningful text"
            )
            
        # Store the old extracted text for comparison
        old_text_length = len(document.extracted_text) if document.extracted_text else 0
        new_text_length = len(new_extracted_text)
        
        # Update document with new extracted text
        document.extracted_text = new_extracted_text
        document.updated_at = datetime.now()
        
        # Update metadata to reflect new OCR processing
        metadata_dict = document.get_metadata_dict()
        metadata_dict['ocr_processing'] = {
            'method_used': ocr_method,
            'language': ocr_language,
            'vision_provider': vision_provider if ocr_method == 'vision_llm' else None,
            'reprocessed_at': datetime.now().isoformat(),
            'previous_text_length': old_text_length,
            'new_text_length': new_text_length
        }
        document.set_metadata(metadata_dict)
        
        # TODO: Optionally reprocess document chunks and update vector embeddings
        # This would require updating the document_version_manager integration
        
        db.commit()
        
        logger.info(f"✅ OCR reprocessing successful - Text length: {old_text_length} -> {new_text_length}")
        
        return {
            "message": "OCR reprocessing completed successfully",
            "document_id": document_id,
            "ocr_settings": {
                "method": ocr_method,
                "language": ocr_language,
                "vision_provider": vision_provider if ocr_method == 'vision_llm' else None
            },
            "text_stats": {
                "previous_length": old_text_length,
                "new_length": new_text_length,
                "improvement": new_text_length - old_text_length
            },
            "filename": document.filename,
            "content_type": document.content_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ OCR reprocessing failed for document {document_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"OCR reprocessing failed: {str(e)}"
        )

@router.post("/{document_id}/recover-file")
async def recover_document_file(
    document_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: DocumentController = Depends(get_document_controller)
):
    """Recover a document by re-uploading its original file."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"📁 Recovery request for document {document_id} by user {current_user.username}")
    
    try:
        # Get the existing document
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted == False
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        # Check permissions
        if document.user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Permission denied")
            
        # Check if this document actually needs recovery
        if document.file_path and get_file_manager().file_exists(document.file_path):
            raise HTTPException(
                status_code=400, 
                detail="Document file already exists and doesn't need recovery"
            )
            
        logger.info(f"🔧 Document needs recovery - Original: {document.filename}, New: {file.filename}")
        
        # Validate file type matches
        if file.content_type != document.content_type:
            logger.warning(f"⚠️ Content type mismatch - Original: {document.content_type}, New: {file.content_type}")
            # Allow it but warn the user
        
        # Read file content
        file_content = await file.read()
        await file.seek(0)
        
        # Use the file manager to save the file
        import hashlib
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        try:
            file_path = get_file_manager().save_file(file_content, file.filename, file_hash)
            logger.info(f"✅ Recovery file saved to: {file_path}")
            
            # Update the document record
            document.file_path = file_path
            document.file_hash = file_hash
            document.file_size = len(file_content)
            document.updated_at = datetime.now()
            
            # If content type changed, update it
            if file.content_type and file.content_type != document.content_type:
                document.content_type = file.content_type
                
            db.commit()
            
            logger.info(f"✅ Document {document_id} recovered successfully")
            
            return {
                "message": "Document file recovered successfully",
                "document_id": document_id,
                "original_filename": document.filename,
                "recovered_filename": file.filename,
                "file_path": file_path,
                "file_size": len(file_content),
                "content_type": document.content_type,
                "content_type_changed": file.content_type != document.content_type if file.content_type else False
            }
            
        except Exception as storage_err:
            logger.error(f"❌ Failed to save recovery file: {storage_err}")
            raise HTTPException(
                status_code=507,
                detail=f"Failed to save recovery file: {str(storage_err)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Recovery failed for document {document_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Document recovery failed: {str(e)}"
        )