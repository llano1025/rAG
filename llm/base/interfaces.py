# llm/base/interfaces.py

"""
Core interfaces for LLM providers.

Defines the abstract base classes and protocols that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Union, AsyncGenerator, Optional, Dict, Any
from .models import ModelConfig, LLMResponse, ModelInfo

class BaseLLM(ABC):
    """
    Abstract base class for all LLM implementations.
    
    This interface defines the contract that all LLM providers must implement
    to ensure consistent behavior across different model providers.
    """
    
    def __init__(self, provider_name: str):
        """Initialize the LLM provider."""
        self.provider_name = provider_name
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """
        Generate text from the model.
        
        Args:
            prompt: Input text to generate from
            config: Model configuration parameters
            stream: Whether to stream the response
            
        Returns:
            LLMResponse for non-streaming, AsyncGenerator for streaming
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
            
        Raises:
            LLMError: If embedding generation fails or not supported
        """
        pass
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo object or None if not found
        """
        return None
    
    async def list_available_models(self) -> List[ModelInfo]:
        """
        List all available models for this provider.
        
        Returns:
            List of ModelInfo objects
        """
        return []
    
    async def validate_config(self, config: ModelConfig) -> bool:
        """
        Validate a model configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation is done in ModelConfig.__post_init__
            return True
        except ValueError:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the provider.
        
        Returns:
            Dictionary with health status information
        """
        return {
            'provider': self.provider_name,
            'status': 'unknown',
            'message': 'Health check not implemented'
        }
    
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming responses."""
        return True
    
    def supports_embeddings(self) -> bool:
        """Check if provider supports embedding generation."""
        return True
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.provider_name
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name})"
    
    def __repr__(self) -> str:
        return self.__str__()

class StreamingLLM(BaseLLM):
    """
    Enhanced base class for LLM providers that support streaming.
    
    Provides additional streaming-specific functionality.
    """
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming text from the model.
        
        Args:
            prompt: Input text to generate from
            config: Model configuration parameters
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            LLMStreamingError: If streaming fails
        """
        pass
    
    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """
        Generate text with automatic streaming support.
        
        This implementation automatically routes to streaming or non-streaming
        based on the stream parameter.
        """
        if stream:
            return self.stream_generate(prompt, config)
        else:
            # Collect streaming response into a single response
            chunks = []
            async for chunk in self.stream_generate(prompt, config):
                chunks.append(chunk)
            
            full_text = ''.join(chunks)
            return LLMResponse(
                text=full_text,
                model_name=config.model_name,
                finish_reason="stop"
            )

class EmbeddingLLM(BaseLLM):
    """
    Enhanced base class for LLM providers that specialize in embeddings.
    
    Provides additional embedding-specific functionality.
    """
    
    @abstractmethod
    async def get_embeddings_batch(
        self, 
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
            
        Raises:
            LLMError: If batch embedding fails
        """
        pass
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding dimension or None if unknown
        """
        return None
    
    def supports_embeddings(self) -> bool:
        """Embedding providers always support embeddings."""
        return True