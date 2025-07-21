"""
Database package for RAG system.
Contains SQLAlchemy models, database connection, and migrations.
"""

from .connection import get_db, engine, SessionLocal
from .models import (
    User, UserSession, APIKey, UserRole, Permission,
    Document, DocumentChunk, VectorIndex, SearchQuery, DocumentStatusEnum
)
from .base import Base

__all__ = [
    "get_db",
    "engine", 
    "SessionLocal",
    "User",
    "UserSession", 
    "APIKey",
    "UserRole",
    "Permission",
    "Document",
    "DocumentChunk",
    "VectorIndex", 
    "SearchQuery",
    "DocumentStatusEnum",
    "Base"
]