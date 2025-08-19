"""
Reranker manager for loading, caching, and managing multiple reranker models.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import threading
import weakref
from collections import defaultdict

from .base_reranker import BaseReranker, RerankResult, SearchResult
from .huggingface_reranker import HuggingFaceReranker

logger = logging.getLogger(__name__)


class RerankerCache:
    """Cache for loaded reranker models with memory management."""
    
    def __init__(self, max_size: int = 3, ttl_hours: int = 24):
        """
        Initialize reranker cache.
        
        Args:
            max_size: Maximum number of models to keep loaded
            ttl_hours: Time-to-live for cached models in hours
        """
        self.max_size = max_size
        self.ttl_delta = timedelta(hours=ttl_hours)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
    
    def get(self, model_name: str) -> Optional[BaseReranker]:
        """Get a cached reranker model."""
        with self._lock:
            if model_name not in self._cache:
                return None
            
            entry = self._cache[model_name]
            
            # Check TTL
            if datetime.now() - entry['loaded_at'] > self.ttl_delta:
                self._remove(model_name)
                return None
            
            # Update access order (move to end)
            if model_name in self._access_order:
                self._access_order.remove(model_name)
            self._access_order.append(model_name)
            
            entry['last_accessed'] = datetime.now()
            return entry['reranker']
    
    def put(self, model_name: str, reranker: BaseReranker):
        """Add a reranker to the cache."""
        with self._lock:
            # Remove if already exists
            if model_name in self._cache:
                self._remove(model_name)
            
            # Make space if needed
            while len(self._cache) >= self.max_size:
                # Remove least recently used
                if self._access_order:
                    oldest = self._access_order[0]
                    self._remove(oldest)
                else:
                    break
            
            # Add new entry
            now = datetime.now()
            self._cache[model_name] = {
                'reranker': reranker,
                'loaded_at': now,
                'last_accessed': now
            }
            self._access_order.append(model_name)
            
            logger.info(f"Cached reranker model: {model_name}")
    
    def remove(self, model_name: str) -> bool:
        """Remove a model from cache."""
        with self._lock:
            return self._remove(model_name)
    
    def _remove(self, model_name: str) -> bool:
        """Internal remove method (assumes lock is held)."""
        if model_name not in self._cache:
            return False
        
        try:
            entry = self._cache[model_name]
            reranker = entry['reranker']
            
            # Unload the model
            if hasattr(reranker, 'unload_model'):
                asyncio.create_task(reranker.unload_model())
            
            del self._cache[model_name]
            
            if model_name in self._access_order:
                self._access_order.remove(model_name)
            
            logger.info(f"Removed reranker model from cache: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing model {model_name} from cache: {e}")
            return False
    
    def clear(self):
        """Clear all cached models."""
        with self._lock:
            model_names = list(self._cache.keys())
            for model_name in model_names:
                self._remove(model_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = {
                'cached_models': len(self._cache),
                'max_size': self.max_size,
                'ttl_hours': self.ttl_delta.total_seconds() / 3600,
                'models': {}
            }
            
            for model_name, entry in self._cache.items():
                stats['models'][model_name] = {
                    'loaded_at': entry['loaded_at'].isoformat(),
                    'last_accessed': entry['last_accessed'].isoformat(),
                    'is_loaded': entry['reranker'].is_loaded
                }
            
            return stats


class RerankerManager:
    """Manager for loading and using multiple reranker models."""
    
    def __init__(
        self,
        cache_size: int = 3,
        cache_ttl_hours: int = 24,
        default_model: str = 'ms-marco-MiniLM-L-6-v2'
    ):
        """
        Initialize reranker manager.
        
        Args:
            cache_size: Maximum number of models to keep in cache
            cache_ttl_hours: Cache TTL in hours
            default_model: Default reranker model to use
        """
        self.cache = RerankerCache(cache_size, cache_ttl_hours)
        self.default_model = default_model
        self._usage_stats = defaultdict(int)
        self._lock = threading.RLock()
        
        logger.info(f"Initialized RerankerManager with default model: {default_model}")
    
    async def get_reranker(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **config
    ) -> BaseReranker:
        """
        Get a reranker model, loading if necessary.
        
        Args:
            model_name: Name of the reranker model
            device: Device to load model on
            **config: Additional model configuration
            
        Returns:
            Loaded reranker instance
        """
        if model_name is None:
            model_name = self.default_model
        
        # Create cache key including configuration
        cache_key = self._create_cache_key(model_name, device, config)
        
        # Try to get from cache first
        reranker = self.cache.get(cache_key)
        if reranker and reranker.is_loaded:
            with self._lock:
                self._usage_stats[cache_key] += 1
            logger.debug(f"Using cached reranker: {cache_key}")
            return reranker
        
        # Load new model
        reranker = await self._load_reranker(model_name, device, **config)
        
        # Cache the loaded model
        self.cache.put(cache_key, reranker)
        
        with self._lock:
            self._usage_stats[cache_key] += 1
        
        return reranker
    
    async def _load_reranker(
        self,
        model_name: str,
        device: Optional[str] = None,
        **config
    ) -> BaseReranker:
        """Load a reranker model."""
        try:
            logger.info(f"Loading reranker model: {model_name}")
            
            # Create reranker instance
            # Currently only supports HuggingFace models, but can be extended
            reranker = HuggingFaceReranker.create_from_model_name(
                model_name, device=device, **config
            )
            
            # Load the model
            success = await reranker.load_model()
            if not success:
                raise RuntimeError(f"Failed to load reranker model: {model_name}")
            
            logger.info(f"Successfully loaded reranker: {model_name}")
            return reranker
            
        except Exception as e:
            logger.error(f"Failed to load reranker {model_name}: {e}")
            raise
    
    def _create_cache_key(
        self,
        model_name: str,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a cache key for a model configuration."""
        key_parts = [model_name]
        
        if device:
            key_parts.append(f"device:{device}")
        
        if config:
            # Sort config items for consistent key generation
            config_parts = [f"{k}:{v}" for k, v in sorted(config.items())]
            key_parts.extend(config_parts)
        
        return "|".join(key_parts)
    
    async def rerank(
        self,
        query: str,
        results: List[Union[SearchResult, Dict[str, Any]]],
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        score_weight: float = 0.5,
        min_rerank_score: Optional[float] = None,
        device: Optional[str] = None,
        **config
    ) -> List[RerankResult]:
        """
        Rerank search results using specified model.
        
        Args:
            query: Search query
            results: Search results to rerank
            model_name: Reranker model to use
            top_k: Maximum results to return
            score_weight: Weight for rerank scores
            min_rerank_score: Minimum rerank score threshold
            device: Device to run model on
            **config: Model configuration
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        # Get reranker model
        reranker = await self.get_reranker(model_name, device, **config)
        
        # Perform reranking
        reranked_results = await reranker.rerank(
            query, results, top_k, score_weight, min_rerank_score
        )
        
        return reranked_results
    
    async def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[Union[SearchResult, Dict[str, Any]]]],
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        score_weight: float = 0.5,
        min_rerank_score: Optional[float] = None,
        device: Optional[str] = None,
        **config
    ) -> List[List[RerankResult]]:
        """Batch rerank multiple result sets."""
        if not queries or not results_list:
            return []
        
        # Get reranker model
        reranker = await self.get_reranker(model_name, device, **config)
        
        # Perform batch reranking
        batch_results = await reranker.batch_rerank(
            queries, results_list, top_k, score_weight, min_rerank_score
        )
        
        return batch_results
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available reranker models."""
        models = []
        
        # Add HuggingFace models
        hf_models = HuggingFaceReranker.get_available_models()
        for model_alias, model_info in hf_models.items():
            models.append({
                'alias': model_alias,
                'full_name': model_info['full_name'],
                'description': model_info['description'],
                'performance_tier': model_info['performance_tier'],
                'provider': 'huggingface',
                'type': 'cross-encoder'
            })
        
        return models
    
    async def health_check(
        self,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform health check on reranker model."""
        try:
            reranker = await self.get_reranker(model_name)
            return await reranker.health_check()
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        with self._lock:
            stats = {
                'default_model': self.default_model,
                'cache_stats': self.cache.get_stats(),
                'usage_stats': dict(self._usage_stats),
                'total_requests': sum(self._usage_stats.values())
            }
        
        return stats
    
    def clear_cache(self):
        """Clear all cached models."""
        self.cache.clear()
        logger.info("Cleared reranker model cache")
    
    async def preload_model(
        self,
        model_name: str,
        device: Optional[str] = None,
        **config
    ) -> bool:
        """
        Preload a model into cache.
        
        Args:
            model_name: Model to preload
            device: Device to load on
            **config: Model configuration
            
        Returns:
            True if preloaded successfully
        """
        try:
            await self.get_reranker(model_name, device, **config)
            logger.info(f"Successfully preloaded reranker: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to preload reranker {model_name}: {e}")
            return False


# Global reranker manager instance
_reranker_manager: Optional[RerankerManager] = None
_manager_lock = threading.RLock()


def get_reranker_manager(**kwargs) -> RerankerManager:
    """
    Get the global reranker manager instance.
    
    Args:
        **kwargs: Configuration for new manager instance
        
    Returns:
        RerankerManager instance
    """
    global _reranker_manager
    
    with _manager_lock:
        if _reranker_manager is None:
            _reranker_manager = RerankerManager(**kwargs)
        return _reranker_manager


def set_reranker_manager(manager: RerankerManager):
    """Set the global reranker manager instance."""
    global _reranker_manager
    
    with _manager_lock:
        _reranker_manager = manager