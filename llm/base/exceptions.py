# llm/base/exceptions.py

"""
Exception hierarchy for LLM operations.

Provides a structured exception system for different types of LLM failures.
"""

class LLMError(Exception):
    """Base exception for all LLM operations."""
    pass

class LLMProviderError(LLMError):
    """Raised when a specific LLM provider fails."""
    
    def __init__(self, provider: str, message: str, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")

class LLMConfigError(LLMError):
    """Raised when LLM configuration is invalid."""
    
    def __init__(self, config_field: str, message: str):
        self.config_field = config_field
        super().__init__(f"Configuration error in '{config_field}': {message}")

class LLMStreamingError(LLMError):
    """Raised when streaming operations fail."""
    
    def __init__(self, message: str, chunk_index: int = None):
        self.chunk_index = chunk_index
        if chunk_index is not None:
            super().__init__(f"Streaming error at chunk {chunk_index}: {message}")
        else:
            super().__init__(f"Streaming error: {message}")

class LLMRateLimitError(LLMProviderError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, provider: str, retry_after: int = None):
        self.retry_after = retry_after
        message = "Rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after} seconds"
        super().__init__(provider, message)

class LLMAuthenticationError(LLMProviderError):
    """Raised when authentication fails."""
    
    def __init__(self, provider: str):
        super().__init__(provider, "Authentication failed - check API key")

class LLMQuotaExceededError(LLMProviderError):
    """Raised when quota/credits are exceeded."""
    
    def __init__(self, provider: str):
        super().__init__(provider, "Quota or credits exceeded")

class LLMModelNotFoundError(LLMProviderError):
    """Raised when requested model is not available."""
    
    def __init__(self, provider: str, model_name: str):
        self.model_name = model_name
        super().__init__(provider, f"Model '{model_name}' not found or not available")