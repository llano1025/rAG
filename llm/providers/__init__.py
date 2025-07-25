# llm/providers/__init__.py

"""
LLM Providers module with auto-discovery.

This module automatically discovers and imports all available LLM providers.
"""

from .base_provider import BaseProviderMixin, ProviderUtils, AsyncHTTPProviderMixin
from .openai_llm import OpenAILLM
from .gemini_llm import GeminiLLM
from .anthropic_llm import AnthropicLLM
from .ollama_llm import OllamaLLM
from .lmstudio_llm import LMStudioLLM

# Auto-register providers with the registry
def _register_providers():
    """Auto-register all providers with the global registry."""
    from ..config.provider_registry import register_provider
    
    # Register all providers
    register_provider(
        "openai",
        OpenAILLM,
        description="OpenAI GPT models with streaming and embeddings",
        supported_models=["gpt-4o-2024-08-06", "gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "o3-2025-04-16"]
    )
    
    register_provider(
        "gemini",
        GeminiLLM,
        description="Google Gemini models with streaming support",
        supported_models=["gemini-2.5-pro", "gemini-2.5-flash"]
    )
    
    register_provider(
        "anthropic",
        AnthropicLLM,
        description="Anthropic Claude models with streaming support",
        supported_models=["claude-sonnet-4-20250514", "claude-3-5-haiku-latest", "claude-opus-4-20250514"]
    )
    
    register_provider(
        "ollama",
        OllamaLLM,
        description="Local Ollama models with streaming and embeddings",
        supported_models=["llama2", "mistral", "codellama"]
    )
    
    register_provider(
        "lmstudio",
        LMStudioLLM,
        description="Local LM Studio models with OpenAI-compatible API",
        supported_models=["local-model"]
    )

# Auto-register on import
_register_providers()

__all__ = [
    'BaseProviderMixin',
    'ProviderUtils', 
    'AsyncHTTPProviderMixin',
    'OpenAILLM',
    'GeminiLLM',
    'AnthropicLLM',
    'OllamaLLM',
    'LMStudioLLM'
]