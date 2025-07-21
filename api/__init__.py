# api/__init__.py
from .routes import document_router, search_router, vector_router, library_router
from .controllers import (
    DocumentController,
    SearchController,
    VectorController,
    LibraryController,
    AuthController
)
from .schemas import (
    Document,
    SearchQuery,
    VectorEntry,
    LibraryStats,
    User,
    APIKey
)

__all__ = [
    'document_router',
    'search_router',
    'vector_router',
    'library_router',
    'DocumentController',
    'SearchController',
    'VectorController',
    'LibraryController',
    'AuthController',
    'Document',
    'SearchQuery',
    'VectorEntry',
    'LibraryStats',
    'User',
    'APIKey'
]