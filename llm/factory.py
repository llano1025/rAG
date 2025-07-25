# llm/factory.py

"""
LLM factory for clean provider instantiation and model manager setup.

This module provides factory functions for creating model managers with
predefined configurations and automatic provider discovery.
"""

import logging
from typing import Optional, Dict, Any

from .model_manager import ModelManager
from .config.model_configs import ModelConfigTemplates, get_default_config
from .config.provider_registry import get_provider_registry

logger = logging.getLogger(__name__)

def create_model_manager_with_defaults() -> ModelManager:
    """
    Create a ModelManager with default model configurations.
    
    This factory function automatically registers available providers
    based on environment configuration and API key availability.
    
    Returns:
        ModelManager instance with registered models
    """
    from config import get_settings
    
    manager = ModelManager()
    settings = get_settings()
    
    # Register OpenAI models if API key is available
    _register_openai_models(manager, settings)
    
    # Register Gemini models if API key is available
    _register_gemini_models(manager, settings)
    
    # Register Anthropic models if API key is available
    _register_anthropic_models(manager, settings)
    
    # Register Ollama models if available
    _register_ollama_models(manager, settings)
    
    # Register LM Studio models if available
    _register_lmstudio_models(manager, settings)
    
    logger.info(f"Model manager created with {len(manager.list_registered_models())} models")
    return manager

def _register_openai_models(manager: ModelManager, settings):
    """Register OpenAI models if API key is available."""
    if not hasattr(settings, 'OPENAI_API_KEY') or not settings.OPENAI_API_KEY:
        logger.info("OpenAI API key not found, skipping OpenAI model registration")
        return
    
    try:
        # GPT-4o (flagship multimodal model)
        gpt4o_config = get_default_config("openai", "gpt-4o-2024-08-06")
        manager.register_model(
            model_id="openai-gpt4o",
            provider_name="openai",
            config=gpt4o_config,
            provider_kwargs={"api_key": settings.OPENAI_API_KEY},
            fallback_priority=0  # Primary model
        )
        
        # GPT-4.1 (latest standard model)
        gpt41_config = get_default_config("openai", "gpt-4.1-2025-04-14")
        manager.register_model(
            model_id="openai-gpt41",
            provider_name="openai",
            config=gpt41_config,
            provider_kwargs={"api_key": settings.OPENAI_API_KEY},
            fallback_priority=1
        )
        
        # O4-Mini (efficient reasoning model)
        o4_mini_config = get_default_config("openai", "o4-mini-2025-04-16")
        manager.register_model(
            model_id="openai-o4-mini",
            provider_name="openai",
            config=o4_mini_config,
            provider_kwargs={"api_key": settings.OPENAI_API_KEY},
            fallback_priority=2  # Fast option
        )
        
        # O3 (reasoning-focused model)
        o3_config = get_default_config("openai", "o3-2025-04-16")
        manager.register_model(
            model_id="openai-o3",
            provider_name="openai",
            config=o3_config,
            provider_kwargs={"api_key": settings.OPENAI_API_KEY},
            fallback_priority=3
        )
        
        logger.info("Registered OpenAI models successfully")
        
    except Exception as e:
        logger.warning(f"Failed to register OpenAI models: {str(e)}")

def _register_gemini_models(manager: ModelManager, settings):
    """Register Gemini models if API key is available."""
    if not hasattr(settings, 'GEMINI_API_KEY') or not settings.GEMINI_API_KEY:
        logger.info("Gemini API key not found, skipping Gemini model registration")
        return
    
    try:
        # Gemini 2.5 Pro
        gemini_pro_config = get_default_config("gemini", "gemini-2.5-pro")
        manager.register_model(
            model_id="gemini-25-pro",
            provider_name="gemini",
            config=gemini_pro_config,
            provider_kwargs={"api_key": settings.GEMINI_API_KEY},
            fallback_priority=3
        )
        
        # Gemini 2.5 Flash
        gemini_flash_config = get_default_config("gemini", "gemini-2.5-flash")
        manager.register_model(
            model_id="gemini-25-flash",
            provider_name="gemini",
            config=gemini_flash_config,
            provider_kwargs={"api_key": settings.GEMINI_API_KEY},
            fallback_priority=4
        )
        
        logger.info("Registered Gemini models successfully")
        
    except Exception as e:
        logger.warning(f"Failed to register Gemini models: {str(e)}")

def _register_anthropic_models(manager: ModelManager, settings):
    """Register Anthropic models if API key is available."""
    if not hasattr(settings, 'ANTHROPIC_API_KEY') or not settings.ANTHROPIC_API_KEY:
        logger.info("Anthropic API key not found, skipping Anthropic model registration")
        return
    
    try:
        # Claude 4 Sonnet
        claude_4_sonnet_config = get_default_config("anthropic", "claude-sonnet-4-20250514")
        manager.register_model(
            model_id="anthropic-claude-4-sonnet",
            provider_name="anthropic",
            config=claude_4_sonnet_config,
            provider_kwargs={"api_key": settings.ANTHROPIC_API_KEY},
            fallback_priority=3
        )
        
        # Claude 3.5 Haiku
        claude_35_haiku_config = get_default_config("anthropic", "claude-3-5-haiku-latest")
        manager.register_model(
            model_id="anthropic-claude-35-haiku",
            provider_name="anthropic",
            config=claude_35_haiku_config,
            provider_kwargs={"api_key": settings.ANTHROPIC_API_KEY},
            fallback_priority=4
        )
        
        # Claude 4 Opus
        claude_4_opus_config = get_default_config("anthropic", "claude-opus-4-20250514")
        manager.register_model(
            model_id="anthropic-claude-4-opus",
            provider_name="anthropic",
            config=claude_4_opus_config,
            provider_kwargs={"api_key": settings.ANTHROPIC_API_KEY},
            fallback_priority=5
        )
        
        logger.info("Registered Anthropic models successfully")
        
    except Exception as e:
        logger.warning(f"Failed to register Anthropic models: {str(e)}")

def _register_ollama_models(manager: ModelManager, settings):
    """Register Ollama models if available."""
    if not hasattr(settings, 'OLLAMA_BASE_URL'):
        logger.info("Ollama base URL not configured, skipping Ollama model registration")
        return
    
    try:
        # Default Ollama model (llama2)
        ollama_config = get_default_config("ollama", "llama3.2:3b")
        manager.register_model(
            model_id="ollama-llama3.2",
            provider_name="ollama",
            config=ollama_config,
            provider_kwargs={"base_url": settings.OLLAMA_BASE_URL},
            fallback_priority=6
        )
        logger.info("Registered Ollama models successfully")
        
    except Exception as e:
        logger.warning(f"Failed to register Ollama models: {str(e)}")

def _register_lmstudio_models(manager: ModelManager, settings):
    """Register LM Studio models if available."""
    if not hasattr(settings, 'LMSTUDIO_BASE_URL'):
        logger.info("LM Studio base URL not configured, skipping LM Studio model registration")
        return
    
    try:
        # Default LM Studio model
        lmstudio_config = get_default_config("lmstudio", "local-model")
        manager.register_model(
            model_id="lmstudio-local",
            provider_name="lmstudio",
            config=lmstudio_config,
            provider_kwargs={"base_url": settings.LMSTUDIO_BASE_URL},
            fallback_priority=8
        )
        
        logger.info("Registered LM Studio models successfully")
        
    except Exception as e:
        logger.warning(f"Failed to register LM Studio models: {str(e)}")

def create_custom_model_manager(
    models_config: Dict[str, Dict[str, Any]]
) -> ModelManager:
    """
    Create a ModelManager with custom model configurations.
    
    Args:
        models_config: Dictionary mapping model_id to configuration dict
                      Format: {
                          "model_id": {
                              "provider": "openai|gemini|ollama|lmstudio",
                              "model_name": "model-name",
                              "config": {...model config...},
                              "provider_kwargs": {...provider args...},
                              "fallback_priority": int
                          }
                      }
    
    Returns:
        ModelManager instance with custom models
    """
    manager = ModelManager()
    
    for model_id, model_info in models_config.items():
        try:
            # Get or create model config
            if 'config' in model_info:
                config = model_info['config']
                if not isinstance(config, ModelConfig):
                    from .base.models import ModelConfig
                    config = ModelConfig.from_dict(config)
            else:
                config = get_default_config(
                    model_info['provider'],
                    model_info['model_name']
                )
            
            # Register model
            manager.register_model(
                model_id=model_id,
                provider_name=model_info['provider'],
                config=config,
                provider_kwargs=model_info.get('provider_kwargs', {}),
                fallback_priority=model_info.get('fallback_priority')
            )
            
        except Exception as e:
            logger.error(f"Failed to register custom model {model_id}: {str(e)}")
    
    logger.info(f"Custom model manager created with {len(manager.list_registered_models())} models")
    return manager

def create_single_provider_manager(
    provider_name: str,
    provider_kwargs: Dict[str, Any],
    models: Optional[Dict[str, str]] = None
) -> ModelManager:
    """
    Create a ModelManager for a single provider.
    
    Args:
        provider_name: Name of the provider (openai, gemini, ollama, lmstudio)
        provider_kwargs: Keyword arguments for provider initialization
        models: Optional dict mapping model_id to model_name
    
    Returns:
        ModelManager instance with single provider models
    """
    manager = ModelManager()
    
    # Default models for each provider
    default_models = {
        "openai": {
            "gpt4o": "gpt-4o-2024-08-06", 
            "gpt41": "gpt-4.1-2025-04-14", 
            "o4mini": "o4-mini-2025-04-16", 
            "o3": "o3-2025-04-16"
        },
        "gemini": {
            "pro": "gemini-2.5-pro", 
            "flash": "gemini-2.5-flash"
        },
        "anthropic": {
            "sonnet4": "claude-sonnet-4-20250514", 
            "haiku35": "claude-3-5-haiku-latest", 
            "opus4": "claude-opus-4-20250514"
        },
        "ollama": {
            "llama32": "llama3.2:3b", 
        },
        "lmstudio": {
            "local": "local-model"
        }
    }
    
    models_to_register = models or default_models.get(provider_name, {"default": "default-model"})
    
    for i, (model_id, model_name) in enumerate(models_to_register.items()):
        try:
            full_model_id = f"{provider_name}-{model_id}"
            config = get_default_config(provider_name, model_name)
            
            manager.register_model(
                model_id=full_model_id,
                provider_name=provider_name,
                config=config,
                provider_kwargs=provider_kwargs,
                fallback_priority=i
            )
            
        except Exception as e:
            logger.error(f"Failed to register {provider_name} model {model_id}: {str(e)}")
    
    logger.info(f"Single provider manager created for {provider_name} with {len(manager.list_registered_models())} models")
    return manager

def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available providers.
    
    Returns:
        Dictionary with provider information
    """
    registry = get_provider_registry()
    return registry.list_provider_info()

def check_provider_health(provider_name: str, **provider_kwargs) -> Dict[str, Any]:
    """
    Check the health of a specific provider.
    
    Args:
        provider_name: Name of the provider to check
        **provider_kwargs: Arguments for provider initialization
    
    Returns:
        Health check results
    """
    try:
        registry = get_provider_registry()
        provider = registry.create_provider(provider_name, **provider_kwargs)
        
        if provider:
            import asyncio
            return asyncio.run(provider.health_check())
        else:
            return {
                'provider': provider_name,
                'status': 'unavailable',
                'message': 'Provider could not be instantiated'
            }
            
    except Exception as e:
        return {
            'provider': provider_name,
            'status': 'error',
            'message': str(e)
        }