# vector_db/__init__.py
# Lazy import approach to avoid failing on missing dependencies at import time

import logging

# Default availability flags - will be updated by lazy imports
EMBEDDING_AVAILABLE = False
SEARCH_AVAILABLE = False
STORAGE_AVAILABLE = False

# Initialize flags on module load
def _initialize_flags():
    """Initialize availability flags."""
    global SEARCH_AVAILABLE, STORAGE_AVAILABLE
    try:
        from .search_optimizer import SEARCH_AVAILABLE as _SEARCH_AVAILABLE
        SEARCH_AVAILABLE = _SEARCH_AVAILABLE
    except:
        SEARCH_AVAILABLE = False
    
    try:
        from .storage_manager import VECTOR_DEPENDENCIES_AVAILABLE
        STORAGE_AVAILABLE = VECTOR_DEPENDENCIES_AVAILABLE
    except:
        STORAGE_AVAILABLE = False

_initialize_flags()

def get_embedding_manager():
    """Lazy import of EmbeddingManager."""
    global EMBEDDING_AVAILABLE
    try:
        from .embedding_manager import EnhancedEmbeddingManager as EmbeddingManager
        EMBEDDING_AVAILABLE = True
        return EmbeddingManager
    except ImportError as e:
        logging.warning(f"EmbeddingManager not available: {e}. Install required dependencies: numpy, transformers, sentence-transformers")
        EMBEDDING_AVAILABLE = False
        return None
    except Exception as e:
        logging.error(f"Failed to import EmbeddingManager: {e}")
        EMBEDDING_AVAILABLE = False
        return None

def get_search_optimizer():
    """Lazy import of SearchOptimizer.""" 
    global SEARCH_AVAILABLE
    try:
        from .search_optimizer import SearchOptimizer, SEARCH_AVAILABLE as _SEARCH_AVAILABLE
        SEARCH_AVAILABLE = _SEARCH_AVAILABLE
        return SearchOptimizer if SEARCH_AVAILABLE else None
    except ImportError as e:
        logging.warning(f"SearchOptimizer not available: {e}. Install required dependencies: numpy, faiss-cpu")
        SEARCH_AVAILABLE = False 
        return None
    except Exception as e:
        logging.error(f"Failed to import SearchOptimizer: {e}")
        SEARCH_AVAILABLE = False
        return None

def get_storage_manager():
    """Lazy import of VectorStorageManager."""
    global STORAGE_AVAILABLE
    try:
        from .storage_manager import VectorStorageManager, get_storage_manager, VECTOR_DEPENDENCIES_AVAILABLE
        STORAGE_AVAILABLE = VECTOR_DEPENDENCIES_AVAILABLE
        return get_storage_manager if STORAGE_AVAILABLE else None
    except ImportError as e:
        logging.warning(f"VectorStorageManager not available: {e}. Install required dependencies: numpy, faiss-cpu, qdrant-client")
        STORAGE_AVAILABLE = False
        return None
    except Exception as e:
        logging.error(f"Failed to import VectorStorageManager: {e}")
        STORAGE_AVAILABLE = False
        return None

def get_chunking():
    """Lazy import of chunking components."""
    try:
        from .chunking import Chunk, AdaptiveChunker, CHUNKING_DEPENDENCIES_AVAILABLE
        return Chunk, AdaptiveChunker, CHUNKING_DEPENDENCIES_AVAILABLE
    except ImportError as e:
        logging.warning(f"Chunking components not available: {e}")
        return None, None, False
    except Exception as e:
        logging.error(f"Failed to import chunking components: {e}")
        return None, None, False

# Set module-level variables for backward compatibility
EmbeddingManager = None
SearchOptimizer = None
VectorStorageManager = None

__all__ = [
    'get_embedding_manager',
    'get_search_optimizer', 
    'get_storage_manager',
    'get_chunking',
    'EMBEDDING_AVAILABLE',
    'SEARCH_AVAILABLE',
    'STORAGE_AVAILABLE',
]