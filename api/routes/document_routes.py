from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from ..middleware.auth import get_current_user
from ..controllers import document_controller
from ..schemas.document_schemas import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    DocumentMetadata
)

router = APIRouter(prefix="/documents", tags=["documents"])

class BatchDeleteRequest(BaseModel):
    document_ids: List[str]

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    folder_id: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    current_user = Depends(get_current_user)
):
    """Upload a new document with optional folder and tags."""
    try:
        document = await document_controller.process_upload(
            file=file,
            folder_id=folder_id,
            tags=tags,
            user_id=current_user.id
        )
        return document
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/batch-upload", response_model=List[DocumentResponse])
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    folder_id: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    current_user = Depends(get_current_user)
):
    """Upload multiple documents simultaneously."""
    try:
        documents = await document_controller.process_batch_upload(
            files=files,
            folder_id=folder_id,
            tags=tags,
            user_id=current_user.id
        )
        return documents
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user = Depends(get_current_user)
):
    """Retrieve a specific document by ID."""
    document = await document_controller.get_document(
        document_id=document_id,
        user_id=current_user.id
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    update_data: DocumentUpdate,
    current_user = Depends(get_current_user)
):
    """Update document metadata."""
    document = await document_controller.update_document(
        document_id=document_id,
        update_data=update_data,
        user_id=current_user.id
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user = Depends(get_current_user)
):
    """Delete a specific document."""
    success = await document_controller.delete_document(
        document_id=document_id,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}

@router.post("/batch-delete")
async def batch_delete_documents(
    request: BatchDeleteRequest,
    current_user = Depends(get_current_user)
):
    """Delete multiple documents in a single request."""
    result = await document_controller.batch_delete_documents(
        document_ids=request.document_ids,
        user_id=current_user.id
    )
    return {"deleted": result.deleted_count, "failed": result.failed_count}

@router.get("/{document_id}/versions", response_model=List[DocumentMetadata])
async def get_document_versions(
    document_id: str,
    current_user = Depends(get_current_user)
):
    """Retrieve version history of a document."""
    versions = await document_controller.get_document_versions(
        document_id=document_id,
        user_id=current_user.id
    )
    return versions