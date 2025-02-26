# api/routes/__init__.py
from fastapi import APIRouter
from .document_routes import router as document_router
from .search_routes import router as search_router
from .vector_routes import router as vector_router
from .library_routes import router as library_router

__all__ = [
    'document_router',
    'search_router',
    'vector_router',
    'library_router'
]