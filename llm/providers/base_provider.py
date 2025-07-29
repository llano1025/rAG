# llm/providers/base_provider.py

"""
Common utilities and helpers for LLM providers.

Provides shared functionality that multiple providers can use.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..base.interfaces import BaseLLM
from ..base.models import LLMResponse, UsageMetrics
from ..base.exceptions import LLMError, LLMProviderError, LLMRateLimitError

logger = logging.getLogger(__name__)

class ProviderUtils:
    """Utility functions for LLM providers."""
    
    @staticmethod
    def parse_usage_from_dict(usage_dict: Dict[str, Any]) -> UsageMetrics:
        """Parse usage metrics from a dictionary."""
        return UsageMetrics(
            prompt_tokens=usage_dict.get('prompt_tokens', 0),
            completion_tokens=usage_dict.get('completion_tokens', 0),
            total_tokens=usage_dict.get('total_tokens', 0),
            estimated_cost=usage_dict.get('estimated_cost')
        )
    
    @staticmethod
    def create_response(
        text: str,
        model_name: str,
        usage_dict: Optional[Dict[str, Any]] = None,
        finish_reason: str = "stop",
        latency: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Create a standardized LLMResponse."""
        usage = ProviderUtils.parse_usage_from_dict(usage_dict or {})
        return LLMResponse(
            text=text,
            model_name=model_name,
            usage=usage,
            finish_reason=finish_reason,
            latency=latency,
            metadata=metadata or {}
        )
    
    @staticmethod
    def handle_provider_error(
        provider_name: str,
        error: Exception,
        model_name: str = ""
    ) -> LLMProviderError:
        """Convert various provider errors to standardized exceptions."""
        error_msg = str(error)
        error_lower = error_msg.lower()
        
        # Check for rate limiting
        if any(phrase in error_lower for phrase in ['rate limit', 'too many requests', '429']):
            return LLMRateLimitError(provider_name)
        
        # Check for authentication errors
        if any(phrase in error_lower for phrase in ['unauthorized', '401', 'invalid api key', 'authentication']):
            from ..base.exceptions import LLMAuthenticationError
            return LLMAuthenticationError(provider_name)
        
        # Check for quota/billing errors
        if any(phrase in error_lower for phrase in ['quota', 'billing', 'insufficient credits', 'exceeded']):
            from ..base.exceptions import LLMQuotaExceededError
            return LLMQuotaExceededError(provider_name)
        
        # Check for model not found errors
        if any(phrase in error_lower for phrase in ['model not found', 'invalid model', 'unknown model']):
            from ..base.exceptions import LLMModelNotFoundError
            return LLMModelNotFoundError(provider_name, model_name)
        
        # Generic provider error
        return LLMProviderError(provider_name, error_msg, error)
    
    @staticmethod
    async def measure_latency(async_func, *args, **kwargs):
        """Measure the latency of an async function call."""
        start_time = asyncio.get_event_loop().time()
        try:
            result = await async_func(*args, **kwargs)
            end_time = asyncio.get_event_loop().time()
            return result, end_time - start_time
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            raise e
    
    @staticmethod
    def calculate_estimated_cost(
        usage: UsageMetrics,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0
    ) -> float:
        """Calculate estimated cost based on token usage."""
        input_cost = (usage.prompt_tokens / 1000) * cost_per_1k_input
        output_cost = (usage.completion_tokens / 1000) * cost_per_1k_output
        return input_cost + output_cost

class BaseProviderMixin:
    """
    Mixin class that provides common functionality for LLM providers.
    
    This can be mixed in with BaseLLM implementations to get common utilities.
    """
    
    def __init__(self, provider_name: str, **kwargs):
        """Initialize the provider mixin."""
        # Only call super().__init__ if the next class in MRO expects arguments
        next_class = super(BaseProviderMixin, self)
        if hasattr(next_class, '__init__'):
            import inspect
            init_sig = inspect.signature(next_class.__init__)
            if len(init_sig.parameters) > 1:  # More than just 'self'
                super().__init__(provider_name, **kwargs)
            else:
                super().__init__(**kwargs)
        
        self.provider_name = provider_name  # Ensure provider_name is set
        self._health_check_cache: Optional[Dict[str, Any]] = None
        self._health_check_time: Optional[datetime] = None
        self._health_check_ttl = 300  # 5 minutes
    
    async def _cached_health_check(self) -> Dict[str, Any]:
        """Perform health check with caching."""
        now = datetime.utcnow()
        
        # Check if we have a valid cached result
        if (self._health_check_cache and 
            self._health_check_time and 
            (now - self._health_check_time).total_seconds() < self._health_check_ttl):
            return self._health_check_cache
        
        # Perform actual health check
        try:
            result = await self._perform_health_check()
            self._health_check_cache = result
            self._health_check_time = now
            return result
        except Exception as e:
            return {
                'provider': self.provider_name,
                'status': 'unhealthy',
                'message': f'Health check failed: {str(e)}',
                'timestamp': now.isoformat()
            }
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Override this method to implement provider-specific health checks."""
        return {
            'provider': self.provider_name,
            'status': 'healthy',
            'message': 'Provider is operational',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _log_request(self, model_name: str, prompt_length: int):
        """Log request information."""
        logger.debug(f"[{self.provider_name}] Request to {model_name}, prompt length: {prompt_length}")
    
    def _log_response(self, model_name: str, response_length: int, latency: float):
        """Log response information."""
        logger.debug(f"[{self.provider_name}] Response from {model_name}, length: {response_length}, latency: {latency:.2f}s")
    
    def _log_error(self, model_name: str, error: Exception):
        """Log error information."""
        logger.error(f"[{self.provider_name}] Error with {model_name}: {str(error)}")

class AsyncHTTPProviderMixin(BaseProviderMixin):
    """
    Mixin for providers that use HTTP APIs.
    
    Provides session management and common HTTP utilities.
    """
    
    def __init__(self, provider_name: str, base_url: str, **kwargs):
        """Initialize HTTP provider mixin."""
        super().__init__(provider_name, **kwargs)
        self.base_url = base_url.rstrip('/')
        self._session: Optional['aiohttp.ClientSession'] = None
    
    async def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _close_session(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()
    
    def __del__(self):
        """Cleanup on deletion."""
        # Check if _session attribute exists and is valid before accessing
        if hasattr(self, '_session') and self._session and not self._session.closed:
            # Schedule session cleanup
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._close_session())
            except RuntimeError:
                pass  # Event loop not available

def requires_api_key(func):
    """Decorator to check if API key is available before making requests."""
    import functools
    
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'api_key') or not self.api_key:
            from ..base.exceptions import LLMAuthenticationError
            raise LLMAuthenticationError(self.provider_name)
        return await func(self, *args, **kwargs)
    return wrapper

def retry_on_rate_limit(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator to retry on rate limit errors with exponential backoff."""
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(self, *args, **kwargs)
                except LLMRateLimitError as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        if hasattr(e, 'retry_after') and e.retry_after:
                            delay = max(delay, e.retry_after)
                        
                        logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        raise e
                except Exception as e:
                    # Don't retry non-rate-limit errors
                    raise e
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator