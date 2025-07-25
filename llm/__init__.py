# llm/__init__.py

"""
LLM module providing clean public API exports.

This module exports the main components for LLM integration while maintaining
backward compatibility after the architectural restructuring.
"""

# Core components
from .model_manager import ModelManager
from .factory import (
    create_model_manager_with_defaults,
    create_custom_model_manager,
    create_single_provider_manager,
    get_available_providers,
    check_provider_health
)

# Base interfaces and models (for advanced users)
from .base.interfaces import BaseLLM, StreamingLLM, EmbeddingLLM
from .base.models import ModelConfig, LLMResponse, ModelInfo
from .base.exceptions import LLMError, LLMProviderError, LLMConfigError

# Configuration and registry
from .config.model_configs import ModelConfigTemplates, get_default_config
from .config.provider_registry import get_provider_registry

# Provider implementations (for direct access if needed)
from .providers import OpenAILLM, GeminiLLM, AnthropicLLM, OllamaLLM, LMStudioLLM

# Legacy imports for backward compatibility
from .prompt_templates import PromptTemplate
from .response_handler import ResponseHandler
from .context_optimizer import ContextOptimizer

# Public API
__all__ = [
    # Core classes
    'ModelManager',
    
    # Factory functions (recommended approach)
    'create_model_manager_with_defaults',
    'create_custom_model_manager',
    'create_single_provider_manager',
    'get_available_providers',
    'check_provider_health',
    
    # Base interfaces (for advanced users)
    'BaseLLM',
    'StreamingLLM', 
    'EmbeddingLLM',
    'ModelConfig',
    'LLMResponse',
    'ModelInfo',
    'LLMError',
    'LLMProviderError',
    'LLMConfigError',
    
    # Configuration utilities
    'ModelConfigTemplates',
    'get_default_config',
    'get_provider_registry',
    
    # Provider implementations (for direct access)
    'OpenAILLM',
    'GeminiLLM',
    'AnthropicLLM',
    'OllamaLLM', 
    'LMStudioLLM',
    
    # Legacy components (backward compatibility)
    'PromptTemplate',
    'ResponseHandler',
    'ContextOptimizer'
]