# api/routes/__init__.py
from fastapi import APIRouter
from .document_routes import router as document_router
from .search_routes import router as search_router
from .vector_routes import router as vector_router
from .library_routes import router as library_router
from .user_routes import router as user_router
from .auth_routes import router as auth_router
from .health_routes import router as health_router
from .admin_routes import router as admin_router
from .advanced_routes import router as advanced_router

__all__ = [
    'document_router',
    'search_router',
    'vector_router',
    'library_router',
    'user_router',
    'auth_router',
    'health_router',
    'admin_router',
    'advanced_router'
]