"""
Base reranker interface and data classes for search result reranking.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Represents a reranked search result with enhanced scoring."""
    
    document_id: Union[int, str]
    chunk_id: str
    text: str
    original_score: float
    rerank_score: float
    combined_score: float
    metadata: Dict[str, Any]
    document_metadata: Optional[Dict[str, Any]] = None
    highlight: Optional[str] = None
    reranker_model: Optional[str] = None
    rerank_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'document_id': self.document_id,
            'chunk_id': self.chunk_id,
            'text': self.text,
            'original_score': self.original_score,
            'rerank_score': self.rerank_score,
            'combined_score': self.combined_score,
            'metadata': self.metadata,
            'document_metadata': self.document_metadata,
            'highlight': self.highlight,
            'reranker_model': self.reranker_model,
            'rerank_time_ms': self.rerank_time_ms
        }


@dataclass
class SearchResult:
    """Input search result for reranking."""
    
    document_id: Union[int, str]
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    document_metadata: Optional[Dict[str, Any]] = None
    highlight: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'document_id': self.document_id,
            'chunk_id': self.chunk_id,
            'text': self.text,
            'score': self.score,
            'metadata': self.metadata,
            'document_metadata': self.document_metadata,
            'highlight': self.highlight
        }


class BaseReranker(ABC):
    """
    Abstract base class for search result rerankers.
    
    Rerankers take initial search results and re-score them using
    specialized models to improve relevance ranking.
    """
    
    def __init__(self, model_name: str, **config):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name/path of the reranker model
            **config: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = config
        self.is_loaded = False
        self.model = None
        
    @abstractmethod
    async def load_model(self) -> bool:
        """
        Load the reranker model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def unload_model(self) -> bool:
        """
        Unload the reranker model to free memory.
        
        Returns:
            True if model unloaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: List[Union[SearchResult, Dict[str, Any]]],
        top_k: Optional[int] = None,
        score_weight: float = 0.5,
        min_rerank_score: Optional[float] = None
    ) -> List[RerankResult]:
        """
        Rerank search results based on query-document relevance.
        
        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Maximum number of results to return (None for all)
            score_weight: Weight for combining original and rerank scores (0-1)
            min_rerank_score: Minimum rerank score threshold
            
        Returns:
            List of reranked results sorted by combined score
        """
        pass
    
    @abstractmethod
    async def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[Union[SearchResult, Dict[str, Any]]]],
        top_k: Optional[int] = None,
        score_weight: float = 0.5,
        min_rerank_score: Optional[float] = None
    ) -> List[List[RerankResult]]:
        """
        Rerank multiple sets of search results efficiently.
        
        Args:
            queries: List of search queries
            results_list: List of result lists to rerank
            top_k: Maximum number of results per query
            score_weight: Weight for combining scores
            min_rerank_score: Minimum rerank score threshold
            
        Returns:
            List of reranked result lists
        """
        pass
    
    @abstractmethod
    async def get_relevance_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Get relevance scores for query-document pairs.
        
        Args:
            query: The search query
            documents: List of document texts
            
        Returns:
            List of relevance scores (higher = more relevant)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            'model_name': self.model_name,
            'is_loaded': self.is_loaded,
            'config': self.config,
            'model_type': self.__class__.__name__
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the reranker.
        
        Returns:
            Dictionary containing health status
        """
        try:
            if not self.is_loaded:
                return {
                    'status': 'error',
                    'message': 'Model not loaded',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Test with simple query
            test_query = "test query"
            test_documents = ["test document"]
            
            start_time = datetime.now()
            scores = await self.get_relevance_scores(test_query, test_documents)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'status': 'healthy',
                'model_name': self.model_name,
                'response_time_ms': response_time,
                'test_score': scores[0] if scores else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed for {self.model_name}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _normalize_search_result(self, result: Union[SearchResult, Dict[str, Any]]) -> SearchResult:
        """
        Normalize input result to SearchResult object.
        
        Args:
            result: Search result as SearchResult object or dictionary
            
        Returns:
            SearchResult object
        """
        if isinstance(result, SearchResult):
            return result
        
        # Convert dictionary to SearchResult
        return SearchResult(
            document_id=result.get('document_id'),
            chunk_id=result.get('chunk_id'),
            text=result.get('text', ''),
            score=result.get('score', 0.0),
            metadata=result.get('metadata', {}),
            document_metadata=result.get('document_metadata'),
            highlight=result.get('highlight')
        )
    
    def _combine_scores(
        self,
        original_score: float,
        rerank_score: float,
        weight: float = 0.5
    ) -> float:
        """
        Combine original and rerank scores.
        
        Args:
            original_score: Original search score
            rerank_score: Reranker model score
            weight: Weight for rerank score (0-1)
            
        Returns:
            Combined score
        """
        # Normalize scores to 0-1 range if needed
        original_normalized = max(0.0, min(1.0, original_score))
        rerank_normalized = max(0.0, min(1.0, rerank_score))
        
        # Weighted combination
        combined = (1 - weight) * original_normalized + weight * rerank_normalized
        
        return combined
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}', loaded={self.is_loaded})"
    
    def __repr__(self) -> str:
        return self.__str__()