from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl

class DocumentBase(BaseModel):
    title: str = Field(..., description="Title of the document")
    description: Optional[str] = Field(None, description="Optional description of the document")
    file_type: str = Field(..., description="MIME type of the document")
    language: Optional[str] = Field(None, description="Document language code (ISO 639-1)")
    tags: List[str] = Field(default_factory=list, description="List of tag IDs associated with the document")
    folder_id: Optional[str] = Field(None, description="ID of the folder containing the document")
    metadata: dict = Field(default_factory=dict, description="Additional metadata for the document")

class DocumentCreate(DocumentBase):
    content: str = Field(..., description="Raw text content of the document")
    source_url: Optional[HttpUrl] = Field(None, description="Original source URL if applicable")

class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    folder_id: Optional[str] = None
    metadata: Optional[dict] = None

class DocumentVersion(BaseModel):
    version_id: str = Field(..., description="Unique identifier for this version")
    created_at: datetime
    created_by: str
    changes: dict = Field(..., description="Changes made in this version")

class Document(DocumentBase):
    id: str = Field(..., description="Unique identifier for the document")
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    version: int = Field(..., description="Current version number")
    versions: List[DocumentVersion] = Field(default_factory=list)
    embedding_id: Optional[str] = Field(None, description="ID of associated vector embedding")
    status: str = Field(..., description="Processing status of the document")
    file_size: int = Field(..., description="Size of the document in bytes")
    chunk_count: Optional[int] = Field(None, description="Number of chunks document is split into")

    class Config:
        from_attributes = True