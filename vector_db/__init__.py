# vector_db/__init__.py
# from .chunking import DocumentChunker  # Currently commented out
from .embedding_manager import EmbeddingManager
from .search_optimizer import SearchOptimizer
# from .version_controller import VersionController  # May need to be implemented

__all__ = [
    # 'DocumentChunker',
    'EmbeddingManager',
    'SearchOptimizer',
    # 'VersionController'
]