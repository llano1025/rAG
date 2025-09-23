# llm/config/provider_registry.py

"""
Provider registry and discovery system.

Manages registration and discovery of LLM providers with support for
plugin-like architecture and dynamic provider loading.
"""

import importlib
import inspect
import logging
from typing import Dict, List, Type, Optional, Any, Callable
from ..base.interfaces import BaseLLM
from ..base.models import ModelInfo

logger = logging.getLogger(__name__)

class ProviderRegistry:
    """Registry for LLM providers with automatic discovery."""
    
    def __init__(self):
        self._providers: Dict[str, Type[BaseLLM]] = {}
        self._provider_info: Dict[str, Dict[str, Any]] = {}
        self._initialization_callbacks: Dict[str, Callable] = {}
    
    def register_provider(
        self,
        provider_name: str,
        provider_class: Type[BaseLLM],
        description: str = "",
        supported_models: Optional[List[str]] = None,
        initialization_callback: Optional[Callable] = None
    ):
        """
        Register a new LLM provider.
        
        Args:
            provider_name: Unique name for the provider
            provider_class: Provider class that implements BaseLLM
            description: Human-readable description
            supported_models: List of supported model names
            initialization_callback: Optional callback for provider setup
        """
        if not issubclass(provider_class, BaseLLM):
            raise ValueError(f"Provider class must inherit from BaseLLM")
        
        self._providers[provider_name] = provider_class
        self._provider_info[provider_name] = {
            'class': provider_class,
            'description': description,
            'supported_models': supported_models or [],
            'module': provider_class.__module__,
            'available': True
        }
        
        if initialization_callback:
            self._initialization_callbacks[provider_name] = initialization_callback
        
        logger.debug(f"Registered LLM provider: {provider_name}")
    
    def unregister_provider(self, provider_name: str):
        """Unregister a provider."""
        if provider_name in self._providers:
            del self._providers[provider_name]
            del self._provider_info[provider_name]
            if provider_name in self._initialization_callbacks:
                del self._initialization_callbacks[provider_name]
            logger.debug(f"Unregistered LLM provider: {provider_name}")
    
    def get_provider_class(self, provider_name: str) -> Optional[Type[BaseLLM]]:
        """Get provider class by name."""
        return self._providers.get(provider_name)
    
    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._providers.keys())
    
    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider."""
        return self._provider_info.get(provider_name)
    
    def list_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all providers."""
        return self._provider_info.copy()
    
    def create_provider(
        self,
        provider_name: str,
        *args,
        **kwargs
    ) -> Optional[BaseLLM]:
        """
        Create an instance of a provider.
        
        Args:
            provider_name: Name of the provider to create
            *args: Arguments for provider constructor
            **kwargs: Keyword arguments for provider constructor
            
        Returns:
            Provider instance or None if not found
        """
        provider_class = self.get_provider_class(provider_name)
        if not provider_class:
            logger.error(f"Provider not found: {provider_name}")
            return None
        
        try:
            # Run initialization callback if present
            if provider_name in self._initialization_callbacks:
                callback = self._initialization_callbacks[provider_name]
                callback()
            
            # Create provider instance
            logger.debug(f"Creating provider instance: {provider_name}")
            provider = provider_class(*args, **kwargs)
            
            # Verify provider is properly initialized
            if not hasattr(provider, 'provider_name'):
                logger.warning(f"Provider {provider_name} missing provider_name attribute")
            
            logger.debug(f"Successfully created provider instance: {provider_name}")
            return provider
            
        except Exception as e:
            logger.error(f"Failed to create provider {provider_name}: {str(e)}", exc_info=True)
            logger.error(f"Provider creation failed with args: {args}, kwargs: {kwargs}")
            # Mark provider as unavailable
            if provider_name in self._provider_info:
                self._provider_info[provider_name]['available'] = False
            return None
    
    def discover_providers(self, base_module: str = "llm.providers"):
        """
        Automatically discover and register providers.
        
        Args:
            base_module: Base module to search for providers
        """
        try:
            # Import the providers module
            providers_module = importlib.import_module(base_module)
            
            # Get all modules in the providers package
            import pkgutil
            for importer, modname, ispkg in pkgutil.iter_modules(providers_module.__path__):
                if not ispkg:  # Only process modules, not packages
                    try:
                        full_module_name = f"{base_module}.{modname}"
                        module = importlib.import_module(full_module_name)
                        
                        # Find classes that inherit from BaseLLM
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, BaseLLM) and 
                                obj != BaseLLM):
                                
                                # Extract provider name from class name
                                provider_name = modname.replace('_llm', '').replace('_', '-')
                                
                                # Get description from docstring
                                description = obj.__doc__ or f"{name} provider"
                                description = description.split('\n')[0].strip()
                                
                                # Register the provider
                                self.register_provider(
                                    provider_name=provider_name,
                                    provider_class=obj,
                                    description=description
                                )
                                
                    except Exception as e:
                        logger.warning(f"Failed to load provider module {modname}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Failed to discover providers in {base_module}: {str(e)}")
    
    def check_provider_availability(self, provider_name: str) -> bool:
        """
        Check if a provider is available and can be instantiated.
        
        Args:
            provider_name: Name of the provider to check
            
        Returns:
            True if provider is available, False otherwise
        """
        if provider_name not in self._providers:
            return False
        
        # Check if provider info indicates availability
        info = self._provider_info.get(provider_name, {})
        if not info.get('available', True):
            return False
        
        # Try to create a test instance (without API keys)
        try:
            provider_class = self._providers[provider_name]
            # This might fail for providers that require API keys
            # But it will catch import errors and other issues
            return True
        except Exception:
            return False
    
    def get_available_providers(self) -> List[str]:
        """Get list of currently available providers."""
        available = []
        for provider_name in self._providers:
            if self.check_provider_availability(provider_name):
                available.append(provider_name)
        return available
    
    def update_provider_models(
        self,
        provider_name: str,
        models: List[str]
    ):
        """Update the list of supported models for a provider."""
        if provider_name in self._provider_info:
            self._provider_info[provider_name]['supported_models'] = models

# Global registry instance
_registry_instance: Optional[ProviderRegistry] = None

def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ProviderRegistry()
        # Auto-discover providers on first access
        _registry_instance.discover_providers()
    return _registry_instance

def register_provider(
    provider_name: str,
    provider_class: Type[BaseLLM],
    **kwargs
):
    """Convenience function to register a provider globally."""
    registry = get_provider_registry()
    registry.register_provider(provider_name, provider_class, **kwargs)

def create_provider(provider_name: str, *args, **kwargs) -> Optional[BaseLLM]:
    """Convenience function to create a provider instance."""
    registry = get_provider_registry()
    return registry.create_provider(provider_name, *args, **kwargs)