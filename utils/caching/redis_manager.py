from typing import Any, Optional, Union
import json
from redis import Redis
from datetime import timedelta
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class RedisManager:
    """
    Manages Redis connection and provides caching operations for the RAG system.
    Implements connection pooling and automatic reconnection.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ssl: bool = False,
        default_ttl: int = 3600,  # 1 hour default TTL
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
            "ssl": ssl,
            "decode_responses": True,
        }
        self.default_ttl = default_ttl
        self._redis_client: Optional[Redis] = None

    @property
    def client(self) -> Redis:
        """
        Lazy initialization of Redis client with automatic reconnection.
        """
        if self._redis_client is None:
            try:
                self._redis_client = Redis(**self.connection_params)
                # Test connection
                self._redis_client.ping()
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                raise
        return self._redis_client

    def set_value(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Store a value in Redis with optional TTL and namespace.
        
        Args:
            key: Cache key
            value: Value to store (will be JSON serialized)
            ttl: Time-to-live in seconds or timedelta
            namespace: Optional namespace to prefix the key
            
        Returns:
            bool: Success status
        """
        try:
            if namespace:
                key = f"{namespace}:{key}"
                
            serialized_value = json.dumps(value)
            
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            elif ttl is None:
                ttl = self.default_ttl
                
            return self.client.set(key, serialized_value, ex=ttl)
            
        except Exception as e:
            logger.error(f"Error setting Redis key {key}: {str(e)}")
            return False

    def get_value(
        self,
        key: str,
        namespace: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """
        Retrieve a value from Redis.
        
        Args:
            key: Cache key
            namespace: Optional namespace prefix
            default: Default value if key doesn't exist
            
        Returns:
            Deserialized value or default if not found
        """
        try:
            if namespace:
                key = f"{namespace}:{key}"
                
            value = self.client.get(key)
            if value is None:
                return default
                
            return json.loads(value)
            
        except Exception as e:
            logger.error(f"Error getting Redis key {key}: {str(e)}")
            return default

    def delete_value(self, key: str, namespace: Optional[str] = None) -> bool:
        """Delete a value from Redis."""
        try:
            if namespace:
                key = f"{namespace}:{key}"
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting Redis key {key}: {str(e)}")
            return False

    def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in a namespace."""
        try:
            pattern = f"{namespace}:*"
            keys = self.client.keys(pattern)
            if keys:
                return bool(self.client.delete(*keys))
            return True
        except Exception as e:
            logger.error(f"Error clearing namespace {namespace}: {str(e)}")
            return False

    def cache_decorator(
        self,
        ttl: Optional[Union[int, timedelta]] = None,
        namespace: Optional[str] = None,
        key_prefix: Optional[str] = None
    ):
        """
        Decorator for caching function results.
        
        Usage:
            @redis_manager.cache_decorator(ttl=300, namespace='my_namespace')
            def expensive_operation(arg1, arg2):
                return some_result
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function name, args, and kwargs
                cache_key_parts = [key_prefix or func.__name__]
                if args:
                    cache_key_parts.append(str(args))
                if kwargs:
                    cache_key_parts.append(str(sorted(kwargs.items())))
                    
                cache_key = ":".join(cache_key_parts)
                
                # Try to get cached result
                cached_result = self.get_value(cache_key, namespace=namespace)
                if cached_result is not None:
                    return cached_result
                    
                # Calculate and cache result
                result = func(*args, **kwargs)
                self.set_value(cache_key, result, ttl=ttl, namespace=namespace)
                return result
                
            return wrapper
        return decorator

    def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            return bool(self.client.ping())
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False

    def __del__(self):
        """Cleanup Redis connection on deletion."""
        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception:
                pass