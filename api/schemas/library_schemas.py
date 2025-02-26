from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class FolderBase(BaseModel):
    name: str = Field(..., description="Name of the folder")
    description: Optional[str] = None
    parent_id: Optional[str] = Field(None, description="ID of parent folder")
    metadata: dict = Field(default_factory=dict)

class FolderCreate(FolderBase):
    pass

class FolderUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Optional[dict] = None

class Folder(FolderBase):
    id: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    document_count: int
    child_folders: List[str] = Field(default_factory=list)

    class Config:
        from_attributes = True

class TagBase(BaseModel):
    name: str = Field(..., description="Name of the tag")
    color: str = Field(..., description="Color code for the tag")
    description: Optional[str] = None

class TagCreate(TagBase):
    pass

class TagUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None

class Tag(TagBase):
    id: str
    created_at: datetime
    created_by: str
    document_count: int

    class Config:
        from_attributes = True

class LibraryStats(BaseModel):
    total_documents: int
    total_folders: int
    total_tags: int
    storage_used_bytes: int
    document_types_count: dict
    documents_by_language: dict
    recent_activity: List[dict]
    folder_hierarchy: dict