from typing import Any, Optional, Union
import json
from redis.asyncio import Redis
from datetime import timedelta
import logging
import asyncio
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

    async def get_client(self) -> Redis:
        """
        Async initialization of Redis client with automatic reconnection.
        """
        if self._redis_client is None:
            try:
                self._redis_client = Redis(**self.connection_params)
                # Test connection asynchronously
                await self._redis_client.ping()
                logger.info("Redis client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                raise
        return self._redis_client

    async def connect(self):
        """Initialize Redis connection (async interface for consistency)."""
        try:
            # Force initialization of client via async method
            await self.get_client()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Failed to establish Redis connection: {str(e)}")
            raise

    async def disconnect(self):
        """Close Redis connection."""
        try:
            if self._redis_client:
                await self._redis_client.close()
                self._redis_client = None
                logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")

    async def set_value(
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
                
            serialized_value = json.dumps(value, default=str)  # Handle non-serializable objects
            
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            elif ttl is None:
                ttl = self.default_ttl
                
            client = await self.get_client()
            result = await client.set(key, serialized_value, ex=ttl)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting Redis key {key}: {str(e)}")
            return False

    async def get_value(
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
                
            client = await self.get_client()
            value = await client.get(key)
            if value is None:
                return default
                
            return json.loads(value)
            
        except Exception as e:
            logger.error(f"Error getting Redis key {key}: {str(e)}")
            return default

    async def delete_value(self, key: str, namespace: Optional[str] = None) -> bool:
        """Delete a value from Redis."""
        try:
            if namespace:
                key = f"{namespace}:{key}"
            client = await self.get_client()
            result = await client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Error deleting Redis key {key}: {str(e)}")
            return False

    async def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in a namespace."""
        try:
            pattern = f"{namespace}:*"
            client = await self.get_client()
            keys = await client.keys(pattern)
            if keys:
                result = await client.delete(*keys)
                return bool(result)
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
            async def wrapper(*args, **kwargs):
                # Generate cache key from function name, args, and kwargs
                cache_key_parts = [key_prefix or func.__name__]
                if args:
                    cache_key_parts.append(str(args))
                if kwargs:
                    cache_key_parts.append(str(sorted(kwargs.items())))
                    
                cache_key = ":".join(cache_key_parts)
                
                # Try to get cached result
                cached_result = await self.get_value(cache_key, namespace=namespace)
                if cached_result is not None:
                    return cached_result
                    
                # Calculate and cache result
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                await self.set_value(cache_key, result, ttl=ttl, namespace=namespace)
                return result
                
            return wrapper
        return decorator

    async def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            client = await self.get_client()
            result = await client.ping()
            return bool(result)
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False

    def __del__(self):
        """Cleanup Redis connection on deletion."""
        if self._redis_client:
            try:
                # Schedule cleanup for async client
                asyncio.create_task(self._redis_client.close())
            except Exception:
                pass