from typing import Any, Optional, List, Dict, Union
from enum import Enum
from datetime import timedelta
import hashlib
import json
import logging
from .redis_manager import RedisManager

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """
    Enumeration of available caching strategies.
    """
    DISABLED = "disabled"  # No caching
    SIMPLE = "simple"      # Basic key-value caching
    SLIDING = "sliding"    # Sliding window expiration
    TAGGED = "tagged"      # Tag-based cache invalidation
    LAYERED = "layered"    # Multi-level caching

class CacheConfig:
    """Configuration for caching behavior."""
    def __init__(
        self,
        strategy: CacheStrategy = CacheStrategy.SIMPLE,
        ttl: Optional[Union[int, timedelta]] = 3600,
        namespace: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_size: Optional[int] = None
    ):
        self.strategy = strategy
        self.ttl = ttl if isinstance(ttl, timedelta) else timedelta(seconds=ttl) if ttl else None
        self.namespace = namespace
        self.tags = tags or []
        self.max_size = max_size

class CacheStrategyManager:
    """
    Implements various caching strategies for the RAG system.
    Provides flexible caching mechanisms based on different use cases.
    """
    
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self._strategy_handlers = {
            CacheStrategy.DISABLED: self._handle_disabled,
            CacheStrategy.SIMPLE: self._handle_simple,
            CacheStrategy.SLIDING: self._handle_sliding,
            CacheStrategy.TAGGED: self._handle_tagged,
            CacheStrategy.LAYERED: self._handle_layered,
        }

    def _generate_cache_key(self, base_key: str, namespace: Optional[str] = None) -> str:
        """Generate a standardized cache key."""
        if namespace:
            return f"{namespace}:{base_key}"
        return base_key

    def _generate_tag_key(self, tag: str) -> str:
        """Generate a key for storing tag metadata."""
        return f"tag:{tag}:keys"

    def _compute_hash(self, value: Any) -> str:
        """Compute a hash of the value for change detection."""
        serialized = json.dumps(value, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def get(
        self,
        key: str,
        config: CacheConfig,
        fetch_func: Optional[callable] = None
    ) -> Optional[Any]:
        """
        Get a value from cache using the specified strategy.
        
        Args:
            key: Cache key
            config: Cache configuration
            fetch_func: Optional function to fetch value if not cached
            
        Returns:
            Cached value or None
        """
        if config.strategy not in self._strategy_handlers:
            raise ValueError(f"Unsupported cache strategy: {config.strategy}")
            
        handler = self._strategy_handlers[config.strategy]
        return await handler(key, config, fetch_func)

    async def set(
        self,
        key: str,
        value: Any,
        config: CacheConfig
    ) -> bool:
        """Set a value in cache using the specified strategy."""
        if config.strategy == CacheStrategy.DISABLED:
            return True
            
        cache_key = self._generate_cache_key(key, config.namespace)
        
        # Handle tagged caching
        if config.strategy == CacheStrategy.TAGGED and config.tags:
            # Store value
            success = self.redis.set_value(cache_key, value, ttl=config.ttl)
            if not success:
                return False
                
            # Update tag indices
            for tag in config.tags:
                tag_key = self._generate_tag_key(tag)
                self.redis.client.sadd(tag_key, cache_key)
                
            return True
            
        # Handle other strategies
        return self.redis.set_value(cache_key, value, ttl=config.ttl)

    async def invalidate(
        self,
        key: Optional[str] = None,
        namespace: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Invalidate cache entries based on key, namespace, or tags.
        
        Args:
            key: Specific key to invalidate
            namespace: Namespace to clear
            tags: List of tags to invalidate
        """
        try:
            if key and namespace:
                return self.redis.delete_value(key, namespace)
                
            if namespace:
                return self.redis.clear_namespace(namespace)
                
            if tags:
                success = True
                for tag in tags:
                    tag_key = self._generate_tag_key(tag)
                    # Get all keys associated with tag
                    keys = self.redis.client.smembers(tag_key)
                    if keys:
                        # Delete all keys and tag index
                        success &= bool(self.redis.client.delete(*keys))
                        success &= bool(self.redis.client.delete(tag_key))
                return success
                
            return True
            
        except Exception as e:
            logger.error(f"Cache invalidation failed: {str(e)}")
            return False

    async def _handle_disabled(
        self,
        key: str,
        config: CacheConfig,
        fetch_func: Optional[callable]
    ) -> Optional[Any]:
        """Handle disabled caching strategy."""
        return await fetch_func() if fetch_func else None

    async def _handle_simple(
        self,
        key: str,
        config: CacheConfig,
        fetch_func: Optional[callable]
    ) -> Optional[Any]:
        """Handle simple caching strategy."""
        cache_key = self._generate_cache_key(key, config.namespace)
        value = self.redis.get_value(cache_key)
        
        if value is None and fetch_func:
            value = await fetch_func()
            if value is not None:
                self.redis.set_value(cache_key, value, ttl=config.ttl)
                
        return value

    async def _handle_sliding(
        self,
        key: str,
        config: CacheConfig,
        fetch_func: Optional[callable]
    ) -> Optional[Any]:
        """Handle sliding window caching strategy."""
        cache_key = self._generate_cache_key(key, config.namespace)
        value = self.redis.get_value(cache_key)
        
        if value is not None:
            # Reset TTL on access
            self.redis.client.expire(
                cache_key,
                int(config.ttl.total_seconds()) if config.ttl else 3600
            )
        elif fetch_func:
            value = await fetch_func()
            if value is not None:
                self.redis.set_value(cache_key, value, ttl=config.ttl)
                
        return value

    async def _handle_tagged(
        self,
        key: str,
        config: CacheConfig,
        fetch_func: Optional[callable]
    ) -> Optional[Any]:
        """Handle tag-based caching strategy."""
        cache_key = self._generate_cache_key(key, config.namespace)
        value = self.redis.get_value(cache_key)
        
        if value is None and fetch_func:
            value = await fetch_func()
            if value is not None:
                await self.set(key, value, config)
                
        return value

    async def _handle_layered(
        self,
        key: str,
        config: CacheConfig,
        fetch_func: Optional[callable]
    ) -> Optional[Any]:
        """
        Handle layered caching strategy.
        Implements a two-level cache with local memory and Redis.
        """
        cache_key = self._generate_cache_key(key, config.namespace)
        
        # Try local memory cache first (implemented in RedisManager)
        value = self.redis.get_value(cache_key)
        
        if value is None and fetch_func:
            value = await fetch_func()
            if value is not None:
                self.redis.set_value(cache_key, value, ttl=config.ttl)
                
        return value