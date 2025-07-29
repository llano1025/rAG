# llm/model_manager.py

"""
Lightweight LLM model manager for orchestration and fallback logic.

This module provides a simplified model manager that focuses on:
- Model registration and lifecycle management
- Fallback logic between providers
- Provider orchestration and coordination
"""

from typing import Dict, List, Optional, Union, AsyncGenerator
import asyncio
import logging

from .base.interfaces import BaseLLM
from .base.models import ModelConfig, LLMResponse, ModelInfo
from .base.exceptions import LLMError
from .config.provider_registry import get_provider_registry, create_provider

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Lightweight model manager for LLM orchestration.
    
    Focuses on model registration, fallback logic, and provider coordination
    rather than containing provider implementations.
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self.registered_models: Dict[str, Dict[str, any]] = {}
        self.provider_instances: Dict[str, BaseLLM] = {}
        self.fallback_chain: List[str] = []
        self.provider_registry = get_provider_registry()
    
    def register_model(
        self,
        model_id: str,
        provider_name: str,
        config: ModelConfig,
        provider_kwargs: Optional[Dict] = None,
        fallback_priority: Optional[int] = None
    ):
        """
        Register a new LLM model.
        
        Args:
            model_id: Unique identifier for the model
            provider_name: Name of the provider (openai, gemini, ollama, lmstudio)
            config: Model configuration
            provider_kwargs: Additional arguments for provider initialization
            fallback_priority: Priority in fallback chain (lower = higher priority)
        """
        self.registered_models[model_id] = {
            'provider_name': provider_name,
            'config': config,
            'provider_kwargs': provider_kwargs or {},
            'available': True
        }
        
        # Add to fallback chain
        if fallback_priority is not None:
            if fallback_priority < len(self.fallback_chain):
                self.fallback_chain.insert(fallback_priority, model_id)
            else:
                self.fallback_chain.append(model_id)
        else:
            self.fallback_chain.append(model_id)
        
        logger.info(f"Registered model: {model_id} (provider: {provider_name})")
    
    def unregister_model(self, model_id: str):
        """Unregister a model."""
        if model_id in self.registered_models:
            del self.registered_models[model_id]
        
        if model_id in self.provider_instances:
            del self.provider_instances[model_id]
        
        if model_id in self.fallback_chain:
            self.fallback_chain.remove(model_id)
        
        logger.info(f"Unregistered model: {model_id}")
    
    def list_registered_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self.registered_models.keys())
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        model_info = self.registered_models.get(model_id)
        return model_info['config'] if model_info else None
    
    async def _get_provider_instance(self, model_id: str) -> Optional[BaseLLM]:
        """Get or create a provider instance for a model."""
        logger.debug(f"[MODEL_MANAGER] Getting provider instance for {model_id}")
        
        if model_id in self.provider_instances:
            logger.debug(f"[MODEL_MANAGER] Using cached provider instance for {model_id}")
            return self.provider_instances[model_id]
        
        model_info = self.registered_models.get(model_id)
        if not model_info:
            logger.error(f"[MODEL_MANAGER] Model {model_id} not registered")
            return None
        
        # Create provider instance
        provider_name = model_info['provider_name']
        provider_kwargs = model_info['provider_kwargs']
        
        logger.info(f"[MODEL_MANAGER] Creating provider instance - Model: {model_id}, Provider: {provider_name}")
        logger.debug(f"[MODEL_MANAGER] Provider kwargs: {provider_kwargs}")
        
        try:
            provider_instance = create_provider(provider_name, **provider_kwargs)
            if provider_instance:
                self.provider_instances[model_id] = provider_instance
                logger.info(f"[MODEL_MANAGER] Provider instance created successfully for {model_id}")
                return provider_instance
            else:
                logger.error(f"[MODEL_MANAGER] Failed to create provider {provider_name} for model {model_id} - create_provider returned None")
                # Mark model as unavailable
                model_info['available'] = False
                return None
                
        except Exception as e:
            logger.error(f"[MODEL_MANAGER] Error creating provider {provider_name} for model {model_id}: {str(e)}", exc_info=True)
            model_info['available'] = False
            return None
    
    async def generate_with_fallback(
        self,
        prompt: str,
        primary_model_id: str,
        stream: bool = False,
        max_fallback_attempts: int = 3
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """
        Generate text with fallback to other models if primary fails.
        
        Args:
            prompt: Input text prompt
            primary_model_id: Primary model to try first
            stream: Whether to stream the response
            max_fallback_attempts: Maximum number of fallback attempts
            
        Returns:
            LLMResponse or AsyncGenerator depending on stream parameter
        """
        logger.info(f"[MODEL_MANAGER] Starting generation with fallback - Primary: {primary_model_id}, Stream: {stream}, Prompt length: {len(prompt)}")
        logger.debug(f"[MODEL_MANAGER] Fallback chain: {self.fallback_chain}")
        logger.debug(f"[MODEL_MANAGER] Registered models: {list(self.registered_models.keys())}")
        
        # Try primary model first
        logger.info(f"[MODEL_MANAGER] Attempting primary model: {primary_model_id}")
        
        try:
            # Check if model is registered
            if primary_model_id not in self.registered_models:
                logger.error(f"[MODEL_MANAGER] Primary model {primary_model_id} not registered")
                raise LLMError(f"Model {primary_model_id} not registered")
            
            # Check if model is available
            model_info = self.registered_models[primary_model_id]
            if not model_info.get('available', True):
                logger.warning(f"[MODEL_MANAGER] Primary model {primary_model_id} marked as unavailable")
                raise LLMError(f"Model {primary_model_id} is unavailable")
            
            logger.debug(f"[MODEL_MANAGER] Primary model info - Provider: {model_info['provider_name']}, Available: {model_info.get('available', True)}")
            
            provider = await self._get_provider_instance(primary_model_id)
            if provider:
                config = self.get_model_config(primary_model_id)
                logger.info(f"[MODEL_MANAGER] Provider instance obtained for {primary_model_id}, generating response")
                
                # Handle streaming vs non-streaming differently
                if stream:
                    # For streaming, await the coroutine to get the async generator
                    logger.info(f"[MODEL_MANAGER] Returning streaming generator from {primary_model_id}")
                    return await provider.generate(prompt, config, stream)
                else:
                    # For non-streaming, await the coroutine
                    return await provider.generate(prompt, config, stream)
            else:
                logger.error(f"[MODEL_MANAGER] Failed to get provider instance for {primary_model_id}")
                raise LLMError(f"Failed to create provider for {primary_model_id}")
                
        except Exception as primary_error:
            logger.error(f"[MODEL_MANAGER] Primary model {primary_model_id} failed: {str(primary_error)}", exc_info=True)
        
        # Try fallback models
        logger.info(f"[MODEL_MANAGER] Starting fallback chain - Max attempts: {max_fallback_attempts}")
        attempted_models = {primary_model_id}
        attempts = 0
        
        for fallback_id in self.fallback_chain:
            if fallback_id in attempted_models or attempts >= max_fallback_attempts:
                logger.debug(f"[MODEL_MANAGER] Skipping {fallback_id} - Already attempted: {fallback_id in attempted_models}, Max attempts reached: {attempts >= max_fallback_attempts}")
                continue
            
            model_info = self.registered_models.get(fallback_id, {})
            if not model_info.get('available', False):
                logger.debug(f"[MODEL_MANAGER] Skipping {fallback_id} - Model not available")
                continue
            
            try:
                logger.info(f"[MODEL_MANAGER] Attempting fallback to model {fallback_id} (attempt {attempts + 1}/{max_fallback_attempts})")
                
                provider = await self._get_provider_instance(fallback_id)
                if provider:
                    config = self.get_model_config(fallback_id)
                    logger.info(f"[MODEL_MANAGER] Fallback provider instance obtained for {fallback_id}")
                    
                    # Handle streaming vs non-streaming differently for fallback too
                    if stream:
                        # For streaming, await the coroutine to get the async generator
                        logger.info(f"[MODEL_MANAGER] Returning streaming generator from fallback {fallback_id}")
                        return await provider.generate(prompt, config, stream)
                    else:
                        # For non-streaming, await the coroutine
                        return await provider.generate(prompt, config, stream)
                else:
                    logger.warning(f"[MODEL_MANAGER] Failed to get provider instance for fallback {fallback_id}")
                    
            except Exception as fallback_error:
                logger.error(f"[MODEL_MANAGER] Fallback model {fallback_id} failed: {str(fallback_error)}", exc_info=True)
                attempted_models.add(fallback_id)
                attempts += 1
                continue
        
        logger.error(f"[MODEL_MANAGER] All models in fallback chain failed - Attempted: {list(attempted_models)}")
        raise LLMError("All models in fallback chain failed")
    
    async def get_embedding_with_fallback(
        self,
        text: str,
        primary_model_id: str,
        max_fallback_attempts: int = 3
    ) -> List[float]:
        """
        Get embeddings with fallback to other models if primary fails.
        
        Args:
            text: Text to embed
            primary_model_id: Primary model to try first
            max_fallback_attempts: Maximum number of fallback attempts
            
        Returns:
            List of embedding values
        """
        # Try primary model first
        try:
            provider = await self._get_provider_instance(primary_model_id)
            if provider and provider.supports_embeddings():
                return await provider.get_embedding(text)
        except Exception as primary_error:
            logger.warning(f"Primary embedding model {primary_model_id} failed: {str(primary_error)}")
        
        # Try fallback models that support embeddings
        attempted_models = {primary_model_id}
        attempts = 0
        
        for fallback_id in self.fallback_chain:
            if fallback_id in attempted_models or attempts >= max_fallback_attempts:
                continue
            
            if not self.registered_models.get(fallback_id, {}).get('available', False):
                continue
            
            try:
                provider = await self._get_provider_instance(fallback_id)
                if provider and provider.supports_embeddings():
                    logger.info(f"Attempting embedding fallback to model {fallback_id}")
                    return await provider.get_embedding(text)
                    
            except Exception as fallback_error:
                logger.warning(f"Embedding fallback model {fallback_id} failed: {str(fallback_error)}")
                attempted_models.add(fallback_id)
                attempts += 1
                continue
        
        raise LLMError("All embedding models in fallback chain failed")
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get information about all available models."""
        all_models = []
        
        for model_id, model_info in self.registered_models.items():
            if not model_info.get('available', False):
                continue
            
            try:
                provider = await self._get_provider_instance(model_id)
                if provider:
                    # Try to get model info from provider
                    config = model_info['config']
                    provider_model_info = await provider.get_model_info(config.model_name)
                    
                    if provider_model_info:
                        all_models.append(provider_model_info)
                    else:
                        # Create basic model info
                        all_models.append(ModelInfo(
                            model_id=model_id,
                            model_name=config.model_name,
                            provider=model_info['provider_name'],
                            display_name=config.model_name,
                            description=f"{model_info['provider_name']} model: {config.model_name}",
                            max_tokens=config.max_tokens,
                            supports_streaming=provider.supports_streaming(),
                            supports_embeddings=provider.supports_embeddings()
                        ))
                        
            except Exception as e:
                logger.warning(f"Failed to get info for model {model_id}: {str(e)}")
        
        return all_models
    
    async def health_check_all_models(self) -> Dict[str, Dict[str, any]]:
        """Perform health check on all registered models."""
        health_results = {}
        
        for model_id, model_info in self.registered_models.items():
            try:
                provider = await self._get_provider_instance(model_id)
                if provider:
                    health_result = await provider.health_check()
                    health_results[model_id] = {
                        'provider': model_info['provider_name'],
                        'model_name': model_info['config'].model_name,
                        'health': health_result,
                        'available': health_result.get('status') == 'healthy'
                    }
                    
                    # Update availability based on health check
                    model_info['available'] = health_result.get('status') == 'healthy'
                else:
                    health_results[model_id] = {
                        'provider': model_info['provider_name'],
                        'model_name': model_info['config'].model_name,
                        'health': {'status': 'unavailable', 'message': 'Provider instance not available'},
                        'available': False
                    }
                    model_info['available'] = False
                    
            except Exception as e:
                health_results[model_id] = {
                    'provider': model_info['provider_name'],
                    'model_name': model_info['config'].model_name,
                    'health': {'status': 'error', 'message': str(e)},
                    'available': False
                }
                model_info['available'] = False
        
        return health_results
    
    def get_provider_registry(self):
        """Get the provider registry instance."""
        return self.provider_registry
    
    def get_fallback_chain(self) -> List[str]:
        """Get the current fallback chain."""
        return self.fallback_chain.copy()
    
    def set_fallback_chain(self, chain: List[str]):
        """Set the fallback chain order."""
        # Validate that all models in chain are registered
        valid_models = [model_id for model_id in chain if model_id in self.registered_models]
        self.fallback_chain = valid_models
        logger.info(f"Updated fallback chain: {valid_models}")
    
    async def cleanup(self):
        """Clean up provider instances and resources."""
        for provider in self.provider_instances.values():
            try:
                # Close any open connections (like HTTP sessions)
                if hasattr(provider, '_close_session'):
                    await provider._close_session()
            except Exception as e:
                logger.warning(f"Error cleaning up provider: {str(e)}")
        
        self.provider_instances.clear()
        logger.info("Model manager cleanup completed")