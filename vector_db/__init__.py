# vector_db/__init__.py
# Lazy import approach to avoid failing on missing dependencies at import time

import logging

# Default availability flags
EMBEDDING_AVAILABLE = False
SEARCH_AVAILABLE = False

def get_embedding_manager():
    """Lazy import of EmbeddingManager."""
    global EMBEDDING_AVAILABLE
    try:
        from .embedding_manager import EmbeddingManager
        EMBEDDING_AVAILABLE = True
        return EmbeddingManager
    except ImportError as e:
        logging.warning(f"EmbeddingManager not available: {e}")
        EMBEDDING_AVAILABLE = False
        return None

def get_search_optimizer():
    """Lazy import of SearchOptimizer.""" 
    global SEARCH_AVAILABLE
    try:
        from .search_optimizer import SearchOptimizer
        SEARCH_AVAILABLE = True
        return SearchOptimizer
    except ImportError as e:
        logging.warning(f"SearchOptimizer not available: {e}")
        SEARCH_AVAILABLE = False 
        return None

# Set module-level variables for backward compatibility
EmbeddingManager = None
SearchOptimizer = None

__all__ = [
    'get_embedding_manager',
    'get_search_optimizer',
    'EMBEDDING_AVAILABLE',
    'SEARCH_AVAILABLE',
]