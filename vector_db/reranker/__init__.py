"""
Reranker module for improving search result relevance using Hugging Face models.

This module provides reranking capabilities to enhance search results by re-scoring
initial search results using specialized cross-encoder models from Hugging Face.
"""

from .base_reranker import BaseReranker, RerankResult
from .huggingface_reranker import HuggingFaceReranker
from .reranker_manager import RerankerManager, get_reranker_manager
from .reranker_config import RerankerConfig, get_reranker_config

__all__ = [
    'BaseReranker',
    'RerankResult',
    'HuggingFaceReranker', 
    'RerankerManager',
    'get_reranker_manager',
    'RerankerConfig',
    'get_reranker_config'
]

__version__ = '1.0.0'