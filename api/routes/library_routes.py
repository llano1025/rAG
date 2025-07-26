from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..middleware.auth import get_current_active_user
from ..controllers.library_controller import LibraryController, get_library_controller
from ..schemas.library_schemas import (
    FolderCreate,
    FolderUpdate,
    Folder,
    TagCreate,
    TagUpdate,
    Tag,
    TagSummary,
    LibraryStats
)
from database.connection import get_db

router = APIRouter(prefix="/library", tags=["library"])

# Folder Routes
@router.post("/folders", response_model=Folder)
async def create_folder(
    folder: FolderCreate,
    current_user = Depends(get_current_active_user),
    controller: LibraryController = Depends(get_library_controller),
    db: Session = Depends(get_db)
):
    """Create a new folder in the library."""
    try:
        created_folder = await controller.create_folder(
            folder_data=folder,
            user_id=current_user.id
        )
        return created_folder
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/folders", response_model=List[Folder])
async def list_folders(
    parent_id: Optional[str] = Query(None),
    current_user = Depends(get_current_active_user),
    controller: LibraryController = Depends(get_library_controller),
    db: Session = Depends(get_db)
):
    """List all folders, optionally filtered by parent folder."""
    folders = await controller.list_folders(
        parent_id=parent_id,
        user_id=current_user.id
    )
    return folders

@router.put("/folders/{folder_id}", response_model=Folder)
async def update_folder(
    folder_id: str,
    folder_update: FolderUpdate,
    current_user = Depends(get_current_active_user),
    controller: LibraryController = Depends(get_library_controller),
    db: Session = Depends(get_db)
):
    """Update folder details."""
    updated_folder = await controller.update_folder(
        folder_id=folder_id,
        folder_data=folder_update,
        user_id=current_user.id
    )
    if not updated_folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    return updated_folder

@router.delete("/folders/{folder_id}")
async def delete_folder(
    folder_id: str,
    recursive: bool = Query(False),
    current_user = Depends(get_current_active_user),
    controller: LibraryController = Depends(get_library_controller),
    db: Session = Depends(get_db)
):
    """Delete a folder and optionally its contents."""
    success = await controller.delete_folder(
        folder_id=folder_id,
        recursive=recursive,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Folder not found")
    return {"message": "Folder deleted successfully"}

# Tag Routes
@router.post("/tags", response_model=Tag)
async def create_tag(
    tag: TagCreate,
    current_user = Depends(get_current_active_user),
    controller: LibraryController = Depends(get_library_controller),
    db: Session = Depends(get_db)
):
    """Create a new tag."""
    try:
        created_tag = await controller.create_tag(
            tag_data=tag,
            user_id=current_user.id
        )
        return created_tag
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/tags", response_model=List[TagSummary])
async def list_tags(
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    controller: LibraryController = Depends(get_library_controller)
):
    """List all available tags."""
    tags = await controller.list_tags(
        user=current_user,
        db=db
    )
    return tags

@router.put("/tags/{tag_id}", response_model=Tag)
async def update_tag(
    tag_id: str,
    tag_update: TagUpdate,
    current_user = Depends(get_current_active_user),
    controller: LibraryController = Depends(get_library_controller),
    db: Session = Depends(get_db)
):
    """Update tag details."""
    updated_tag = await controller.update_tag(
        tag_id=tag_id,
        tag_data=tag_update,
        user_id=current_user.id
    )
    if not updated_tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    return updated_tag

@router.delete("/tags/{tag_id}")
async def delete_tag(
    tag_id: str,
    current_user = Depends(get_current_active_user),
    controller: LibraryController = Depends(get_library_controller),
    db: Session = Depends(get_db)
):
    """Delete a tag."""
    success = await controller.delete_tag(
        tag_id=tag_id,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Tag not found")
    return {"message": "Tag deleted successfully"}

# Document Organization Routes

class DocumentMoveRequest(BaseModel):
    document_ids: List[int]
    target_folder_id: Optional[str] = None

@router.post("/move-documents")
async def move_documents(
    move_request: DocumentMoveRequest,
    current_user = Depends(get_current_active_user),
    controller: LibraryController = Depends(get_library_controller),
    db: Session = Depends(get_db)
):
    """Move documents to a different folder."""
    try:
        result = await controller.move_documents(
            document_ids=move_request.document_ids,
            target_folder_id=move_request.target_folder_id,
            user_id=current_user.id
        )
        return {"message": "Documents moved successfully", "moved_count": len(move_request.document_ids)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/documents/{document_id}/tags/{tag_id}")
async def add_tag_to_document(
    document_id: str,
    tag_id: str,
    current_user = Depends(get_current_active_user),
    controller: LibraryController = Depends(get_library_controller),
    db: Session = Depends(get_db)
):
    """Add a tag to a document."""
    success = await controller.add_document_tag(
        document_id=document_id,
        tag_id=tag_id,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Document or tag not found")
    return {"message": "Tag added successfully"}

@router.delete("/documents/{document_id}/tags/{tag_id}")
async def remove_tag_from_document(
    document_id: str,
    tag_id: str,
    current_user = Depends(get_current_active_user),
    controller: LibraryController = Depends(get_library_controller),
    db: Session = Depends(get_db)
):
    """Remove a tag from a document."""
    success = await controller.remove_document_tag(
        document_id=document_id,
        tag_id=tag_id,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Document or tag not found")
    return {"message": "Tag removed successfully"}

@router.get("/stats", response_model=LibraryStats)
async def get_library_stats(
    current_user = Depends(get_current_active_user),
    controller: LibraryController = Depends(get_library_controller),
    db: Session = Depends(get_db)
):
    """Get statistics about the library (document count, storage used, etc.)."""
    stats = await controller.get_library_stats(
        user_id=current_user.id
    )
    return stats