"""
Model Discovery Service

Provides functionality to discover available models from different LLM providers.
Supports OpenAI, Gemini, Anthropic, Ollama, and LMStudio.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
import json
import time
from functools import wraps

from llm.config.provider_registry import get_provider_registry
from llm.config.model_configs import ModelConfigTemplates
from llm.base.models import ModelInfo

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry async functions on failure with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {str(e)}")
            
            raise last_exception
        return wrapper
    return decorator

@dataclass
class DiscoveredModel:
    """Information about a discovered model."""
    id: str
    name: str
    provider: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    supports_streaming: bool = True
    supports_embeddings: bool = False
    is_available: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.name
        if self.metadata is None:
            self.metadata = {}

class ModelDiscoveryService:
    """Service for discovering available models from different providers."""
    
    def __init__(self):
        self.provider_registry = get_provider_registry()
        self._session = None
        self._model_cache = {}  # Cache for discovered models
        self._cache_ttl = 3600  # Cache TTL in seconds (1 hour)
        self._cache_timestamps = {}  # Track when models were cached
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for external API calls."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def cleanup(self):
        """Clean up HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _get_cache_key(self, provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
        """Generate cache key for provider/credentials combination."""
        # Hash sensitive data (API key) to avoid storing it in plaintext
        key_hash = str(hash(api_key)) if api_key else "no_key"
        base_url_part = base_url or "default_url"
        return f"{provider}_{key_hash}_{base_url_part}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        
        cached_time = self._cache_timestamps[cache_key]
        return (time.time() - cached_time) < self._cache_ttl
    
    def _cache_models(self, cache_key: str, models: List[DiscoveredModel]):
        """Cache discovered models."""
        self._model_cache[cache_key] = models
        self._cache_timestamps[cache_key] = time.time()
        logger.debug(f"Cached {len(models)} models for key: {cache_key}")
    
    def _get_cached_models(self, cache_key: str) -> Optional[List[DiscoveredModel]]:
        """Get cached models if valid."""
        if self._is_cache_valid(cache_key):
            models = self._model_cache.get(cache_key)
            if models:
                logger.debug(f"Using cached models for key: {cache_key}")
                return models
        return None
    
    def clear_cache(self, provider: Optional[str] = None):
        """Clear model cache for a specific provider or all providers."""
        if provider:
            # Clear cache entries for specific provider
            keys_to_remove = [key for key in self._model_cache.keys() if key.startswith(f"{provider}_")]
            for key in keys_to_remove:
                del self._model_cache[key]
                del self._cache_timestamps[key]
            logger.info(f"Cleared cache for provider: {provider}")
        else:
            # Clear all cache
            self._model_cache.clear()
            self._cache_timestamps.clear()
            logger.info("Cleared all model cache")
    
    async def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of all available providers with their information."""
        providers = []
        
        # Get registered providers from the registry
        provider_info = self.provider_registry.list_provider_info()
        
        for provider_name, info in provider_info.items():
            provider_data = {
                'name': provider_name,
                'display_name': provider_name.replace('-', ' ').title(),
                'description': info.get('description', f'{provider_name} LLM provider'),
                'available': info.get('available', True),
                'supports_discovery': self._supports_model_discovery(provider_name),
                'requires_api_key': self._requires_api_key(provider_name),
                'supported_models': info.get('supported_models', [])
            }
            providers.append(provider_data)
        
        return providers
    
    def _supports_model_discovery(self, provider: str) -> bool:
        """Check if provider supports automatic model discovery."""
        discovery_support = {
            'openai': True,
            'ollama': True,
            'lmstudio': True,
            'gemini': True,  # Now supports live API discovery
            'anthropic': True  # Now supports live API discovery
        }
        return discovery_support.get(provider.lower(), False)
    
    def _requires_api_key(self, provider: str) -> bool:
        """Check if provider requires API key for model discovery."""
        api_key_required = {
            'openai': True,
            'gemini': True,
            'anthropic': True,
            'ollama': False,
            'lmstudio': False
        }
        return api_key_required.get(provider.lower(), True)
    
    async def discover_models(
        self, 
        provider: str, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_cache: bool = True
    ) -> List[DiscoveredModel]:
        """
        Discover available models for a specific provider.
        
        Args:
            provider: Provider name (openai, gemini, anthropic, ollama, lmstudio)
            api_key: API key for providers that require it
            base_url: Custom base URL for local providers
            use_cache: Whether to use cached results if available
            
        Returns:
            List of discovered models
        """
        provider_lower = provider.lower()
        
        # Check cache first if enabled
        cache_key = self._get_cache_key(provider_lower, api_key, base_url)
        if use_cache:
            cached_models = self._get_cached_models(cache_key)
            if cached_models:
                return cached_models
        
        try:
            models = []
            if provider_lower == 'openai':
                models = await self._discover_openai_models(api_key)
            elif provider_lower == 'ollama':
                models = await self._discover_ollama_models(base_url or "http://localhost:11434")
            elif provider_lower == 'lmstudio':
                models = await self._discover_lmstudio_models(base_url or "http://localhost:1234")
            elif provider_lower == 'gemini':
                models = await self._discover_gemini_models(api_key)
            elif provider_lower == 'anthropic':
                models = await self._discover_anthropic_models(api_key)
            else:
                logger.warning(f"Unknown provider: {provider}")
                return []
            
            # Cache the results
            if models and use_cache:
                self._cache_models(cache_key, models)
            
            return models
                
        except Exception as e:
            logger.error(f"Error discovering models for {provider}: {str(e)}")
            # Try to return cached models as fallback
            if use_cache:
                cached_models = self._get_cached_models(cache_key)
                if cached_models:
                    logger.info(f"Returning cached models for {provider} due to API error")
                    return cached_models
            return []
    
    @retry_on_failure(max_retries=2, delay=1.0)
    async def _discover_openai_models(self, api_key: Optional[str]) -> List[DiscoveredModel]:
        """Discover OpenAI models using their API."""
        if not api_key:
            # Return known models without API call
            return self._get_openai_models()
        
        try:
            session = await self._get_session()
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            async with session.get('https://api.openai.com/v1/models', headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get('data', []):
                        model_id = model_data.get('id', '')
                        owned_by = model_data.get('owned_by', '')
                        
                        # Filter to only include relevant OpenAI models
                        if any(keyword in model_id.lower() for keyword in ['gpt', 'text-embedding', 'whisper', 'dall-e']):
                            # Categorize model type
                            model_type = self._categorize_openai_model(model_id)
                            supports_embeddings = 'embedding' in model_id.lower()
                            
                            # Estimate context window and max tokens
                            context_window = self._estimate_openai_context_window(model_id)
                            max_tokens = self._estimate_openai_max_tokens(model_id)
                            
                            # Create better display name
                            display_name = self._format_openai_display_name(model_id)
                            
                            models.append(DiscoveredModel(
                                id=model_id,
                                name=model_id,
                                provider='openai',
                                display_name=display_name,
                                description=f"OpenAI {model_type} model",
                                context_window=context_window,
                                max_tokens=max_tokens,
                                supports_streaming=not supports_embeddings,  # Embeddings don't support streaming
                                supports_embeddings=supports_embeddings,
                                metadata={
                                    'created': model_data.get('created'),
                                    'owned_by': owned_by,
                                    'object': model_data.get('object'),
                                    'model_type': model_type,
                                    'api_discovered': True
                                }
                            ))
                    
                    return models
                else:
                    logger.warning(f"OpenAI API returned status {response.status}")
                    return self._get_openai_models()
                    
        except Exception as e:
            logger.error(f"Error fetching OpenAI models: {str(e)}")
            return self._get_openai_models()
    
    def _categorize_openai_model(self, model_id: str) -> str:
        """Categorize OpenAI model by type."""
        model_id_lower = model_id.lower()
        
        if 'embedding' in model_id_lower:
            return 'Embedding'
        elif 'whisper' in model_id_lower:
            return 'Audio'
        elif 'dall-e' in model_id_lower or 'dalle' in model_id_lower:
            return 'Image Generation'
        elif any(gpt in model_id_lower for gpt in ['gpt-4', 'gpt-3.5', 'gpt-3']):
            return 'Chat Completion'
        else:
            return 'Language Model'
    
    def _estimate_openai_context_window(self, model_id: str) -> int:
        """Estimate context window based on OpenAI model ID."""
        model_id_lower = model_id.lower()
        
        # GPT-4 models
        if 'gpt-4o' in model_id_lower:
            return 128000  # 128k context
        elif 'gpt-4' in model_id_lower and 'turbo' in model_id_lower:
            return 128000  # 128k context
        elif 'gpt-4' in model_id_lower:
            return 128000  # 128k context for most GPT-4 models
        # GPT-3.5 models
        elif 'gpt-3.5-turbo' in model_id_lower:
            return 16385   # 16k context
        elif 'gpt-3.5' in model_id_lower:
            return 4096    # 4k context
        # Embedding models
        elif 'embedding' in model_id_lower:
            return 8191    # Text embedding context
        # Other models
        else:
            return 4096    # Default context
    
    def _estimate_openai_max_tokens(self, model_id: str) -> int:
        """Estimate max output tokens based on OpenAI model ID."""
        model_id_lower = model_id.lower()
        
        # GPT-4 models typically support 4096 max tokens
        if 'gpt-4' in model_id_lower:
            return 4096
        # GPT-3.5 models
        elif 'gpt-3.5' in model_id_lower:
            return 4096
        # Embedding models don't have output tokens (they return vectors)
        elif 'embedding' in model_id_lower:
            return 0
        else:
            return 4096
    
    def _format_openai_display_name(self, model_id: str) -> str:
        """Format OpenAI model ID into a readable display name."""
        # Remove common suffixes and format nicely
        display_name = model_id.replace('-', ' ').title()
        
        # Handle specific model formats
        if 'Gpt 4O' in display_name:
            display_name = display_name.replace('Gpt 4O', 'GPT-4o')
        elif 'Gpt 4' in display_name:
            display_name = display_name.replace('Gpt 4', 'GPT-4')
        elif 'Gpt 3.5' in display_name or 'Gpt 3 5' in display_name:
            display_name = display_name.replace('Gpt 3.5', 'GPT-3.5').replace('Gpt 3 5', 'GPT-3.5')
        elif 'Text Embedding' in display_name:
            display_name = display_name.replace('Text Embedding', 'Text Embedding')
        elif 'Dall E' in display_name:
            display_name = display_name.replace('Dall E', 'DALL-E')
        
        return display_name
    
    def _get_openai_models(self) -> List[DiscoveredModel]:
        """Get known OpenAI models from config templates (fallback)."""
        templates = ModelConfigTemplates.get_provider_templates('openai')
        models = []
        
        for template_name, config in templates.items():
            model_name = config.model_name
            models.append(DiscoveredModel(
                id=model_name,
                name=model_name,
                provider='openai',
                display_name=model_name,
                context_window=config.context_window,
                max_tokens=config.max_tokens,
                supports_embeddings='embedding' in model_name.lower(),
                metadata={'template': template_name, 'api_discovered': False}
            ))
        
        return models
    
    async def _discover_ollama_models(self, base_url: str) -> List[DiscoveredModel]:
        """Discover Ollama models by calling the local API."""
        try:
            session = await self._get_session()
            
            # Check if Ollama server is running
            async with session.get(f"{base_url}/api/version") as response:
                if response.status != 200:
                    logger.warning(f"Ollama server not responding at {base_url}")
                    return []
            
            # Get list of installed models
            async with session.get(f"{base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get('models', []):
                        model_name = model_data.get('name', '')
                        model_size = model_data.get('size', 0)
                        modified_at = model_data.get('modified_at', '')
                        
                        models.append(DiscoveredModel(
                            id=model_name,
                            name=model_name,
                            provider='ollama',
                            display_name=model_name,
                            description=f"Local Ollama model ({self._format_size(model_size)})",
                            metadata={
                                'size': model_size,
                                'modified_at': modified_at,
                                'digest': model_data.get('digest', ''),
                                'base_url': base_url
                            }
                        ))
                    
                    return models
                else:
                    logger.warning(f"Failed to get Ollama models: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error discovering Ollama models: {str(e)}")
            return []
    
    async def _discover_lmstudio_models(self, base_url: str) -> List[DiscoveredModel]:
        """Discover LMStudio models by calling the local API."""
        try:
            session = await self._get_session()
            
            # Check if LMStudio server is running
            async with session.get(f"{base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get('data', []):
                        model_id = model_data.get('id', '')
                        
                        models.append(DiscoveredModel(
                            id=model_id,
                            name=model_id,
                            provider='lmstudio',
                            display_name=model_id,
                            description=f"Local LMStudio model",
                            metadata={
                                'object': model_data.get('object', ''),
                                'owned_by': model_data.get('owned_by', ''),
                                'base_url': base_url
                            }
                        ))
                    
                    return models
                else:
                    logger.warning(f"LMStudio server not responding at {base_url}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error discovering LMStudio models: {str(e)}")
            return []
    
    @retry_on_failure(max_retries=2, delay=1.0)
    async def _discover_gemini_models(self, api_key: Optional[str]) -> List[DiscoveredModel]:
        """Discover Gemini models using their API."""
        if not api_key:
            # Return known models without API call
            return self._get_gemini_models()
        
        try:
            session = await self._get_session()
            
            # Gemini API endpoint with API key as query parameter
            url = 'https://generativelanguage.googleapis.com/v1beta/models'
            params = {'key': api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get('models', []):
                        model_name = model_data.get('name', '')
                        display_name = model_data.get('displayName', model_name)
                        description = model_data.get('description', '')
                        
                        # Extract model ID from full name (e.g., "models/gemini-pro" -> "gemini-pro")
                        model_id = model_name.split('/')[-1] if '/' in model_name else model_name
                        
                        # Filter for generation models (not embedding models)
                        supported_methods = model_data.get('supportedGenerationMethods', [])
                        if 'generateContent' in supported_methods:
                            # Estimate context window and max tokens based on model name
                            context_window = self._estimate_gemini_context_window(model_id)
                            max_tokens = self._estimate_gemini_max_tokens(model_id)
                            
                            models.append(DiscoveredModel(
                                id=model_id,
                                name=model_id,
                                provider='gemini',
                                display_name=display_name,
                                description=description,
                                context_window=context_window,
                                max_tokens=max_tokens,
                                supports_streaming=True,
                                supports_embeddings='embedding' in model_id.lower(),
                                metadata={
                                    'full_name': model_name,
                                    'version': model_data.get('version', ''),
                                    'supported_methods': supported_methods,
                                    'api_discovered': True
                                }
                            ))
                    
                    logger.info(f"Discovered {len(models)} Gemini models via API")
                    return models if models else self._get_gemini_models()
                    
                else:
                    logger.warning(f"Gemini API returned status {response.status}")
                    return self._get_gemini_models()
                    
        except Exception as e:
            logger.error(f"Error fetching Gemini models: {str(e)}")
            return self._get_gemini_models()
    
    def _estimate_gemini_context_window(self, model_id: str) -> int:
        """Estimate context window based on Gemini model ID."""
        model_id_lower = model_id.lower()
        
        # Gemini 2.5 models have 1M+ context
        if 'gemini-2.5' in model_id_lower or 'gemini-2-5' in model_id_lower:
            return 1048576  # 1M tokens
        # Gemini 2.0 models
        elif 'gemini-2.0' in model_id_lower or 'gemini-2-0' in model_id_lower:
            return 1048576  # 1M tokens  
        # Gemini 1.5 models have larger context
        elif 'gemini-1.5' in model_id_lower or 'gemini-1-5' in model_id_lower:
            return 1048576  # 1M tokens
        # Pro models typically have larger context
        elif 'pro' in model_id_lower:
            return 1048576  # 1M tokens
        # Flash models are optimized for speed
        elif 'flash' in model_id_lower:
            return 1048576  # 1M tokens
        # Default for other models
        else:
            return 30720  # 30k tokens
    
    def _estimate_gemini_max_tokens(self, model_id: str) -> int:
        """Estimate max output tokens based on Gemini model ID."""
        # Gemini models typically support up to 65536 output tokens
        return 65536
    
    def _get_gemini_models(self) -> List[DiscoveredModel]:
        """Get known Gemini models from config templates (fallback)."""
        templates = ModelConfigTemplates.get_provider_templates('gemini')
        models = []
        
        for template_name, config in templates.items():
            model_name = config.model_name
            models.append(DiscoveredModel(
                id=model_name,
                name=model_name,
                provider='gemini',
                display_name=model_name,
                context_window=config.context_window,
                max_tokens=config.max_tokens,
                metadata={'template': template_name, 'api_discovered': False}
            ))
        
        return models
    
    @retry_on_failure(max_retries=2, delay=1.0)
    async def _discover_anthropic_models(self, api_key: Optional[str]) -> List[DiscoveredModel]:
        """Discover Anthropic models using their API."""
        if not api_key:
            # Return known models without API call
            return self._get_anthropic_models()
        
        try:
            session = await self._get_session()
            headers = {
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            }
            
            # Try to get models with default pagination
            url = 'https://api.anthropic.com/v1/models'
            params = {'limit': 100}  # Get up to 100 models
            
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get('data', []):
                        model_id = model_data.get('id', '')
                        display_name = model_data.get('display_name', model_id)
                        created_at = model_data.get('created_at', '')
                        
                        # Filter to only include Claude models
                        if 'claude' in model_id.lower():
                            # Estimate context window and max tokens based on model name
                            context_window = self._estimate_anthropic_context_window(model_id)
                            max_tokens = self._estimate_anthropic_max_tokens(model_id)
                            
                            models.append(DiscoveredModel(
                                id=model_id,
                                name=model_id,
                                provider='anthropic',
                                display_name=display_name,
                                description=f"Claude model - {display_name}",
                                context_window=context_window,
                                max_tokens=max_tokens,
                                supports_streaming=True,
                                supports_embeddings=False,
                                metadata={
                                    'created_at': created_at,
                                    'type': model_data.get('type', 'model'),
                                    'api_discovered': True
                                }
                            ))
                    
                    logger.info(f"Discovered {len(models)} Anthropic models via API")
                    return models if models else self._get_anthropic_models()
                    
                else:
                    logger.warning(f"Anthropic API returned status {response.status}")
                    return self._get_anthropic_models()
                    
        except Exception as e:
            logger.error(f"Error fetching Anthropic models: {str(e)}")
            return self._get_anthropic_models()
    
    def _estimate_anthropic_context_window(self, model_id: str) -> int:
        """Estimate context window based on Anthropic model ID."""
        model_id_lower = model_id.lower()
        
        # Claude 4 models typically have 200k context
        if 'claude-sonnet-4' in model_id_lower or 'claude-4' in model_id_lower:
            return 200000
        # Claude 3.5 models typically have 200k context  
        elif 'claude-3-5' in model_id_lower or 'claude-3.5' in model_id_lower:
            return 200000
        # Claude 3 models typically have 200k context
        elif 'claude-3' in model_id_lower:
            return 200000
        # Older models or unknown - use conservative estimate
        else:
            return 100000
    
    def _estimate_anthropic_max_tokens(self, model_id: str) -> int:
        """Estimate max output tokens based on Anthropic model ID."""
        # Anthropic models typically support up to 8192 output tokens
        return 8192
    
    def _get_anthropic_models(self) -> List[DiscoveredModel]:
        """Get known Anthropic models from config templates (fallback)."""
        templates = ModelConfigTemplates.get_provider_templates('anthropic')
        models = []
        
        for template_name, config in templates.items():
            model_name = config.model_name
            models.append(DiscoveredModel(
                id=model_name,
                name=model_name,
                provider='anthropic',
                display_name=model_name,
                context_window=config.context_window,
                max_tokens=config.max_tokens,
                metadata={'template': template_name, 'api_discovered': False}
            ))
        
        return models
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable format."""
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    async def get_model_templates(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get model configuration templates for a provider or all providers."""
        if provider:
            templates = ModelConfigTemplates.get_provider_templates(provider)
            return {
                provider: {
                    name: config.to_dict() 
                    for name, config in templates.items()
                }
            }
        else:
            all_templates = ModelConfigTemplates.get_all_templates()
            organized = {}
            
            for template_name, config in all_templates.items():
                # Extract provider from template name
                provider_name = template_name.split('_')[0]
                if provider_name not in organized:
                    organized[provider_name] = {}
                organized[provider_name][template_name] = config.to_dict()
            
            return organized
    
    async def test_provider_connectivity(
        self, 
        provider: str, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Test connectivity to a provider."""
        provider_lower = provider.lower()
        
        try:
            if provider_lower == 'openai' and api_key:
                session = await self._get_session()
                headers = {'Authorization': f'Bearer {api_key}'}
                async with session.get('https://api.openai.com/v1/models', headers=headers) as response:
                    return {
                        'provider': provider,
                        'connected': response.status == 200,
                        'status_code': response.status,
                        'message': 'Connected successfully' if response.status == 200 else f'HTTP {response.status}'
                    }
            
            elif provider_lower == 'anthropic' and api_key:
                session = await self._get_session()
                headers = {
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01'
                }
                async with session.get('https://api.anthropic.com/v1/models', headers=headers) as response:
                    return {
                        'provider': provider,
                        'connected': response.status == 200,
                        'status_code': response.status,
                        'message': 'Connected successfully' if response.status == 200 else f'HTTP {response.status}'
                    }
            
            elif provider_lower == 'gemini' and api_key:
                session = await self._get_session()
                params = {'key': api_key}
                async with session.get('https://generativelanguage.googleapis.com/v1beta/models', params=params) as response:
                    return {
                        'provider': provider,
                        'connected': response.status == 200,
                        'status_code': response.status,
                        'message': 'Connected successfully' if response.status == 200 else f'HTTP {response.status}'
                    }
            
            elif provider_lower == 'ollama':
                url = base_url or "http://localhost:11434"
                session = await self._get_session()
                async with session.get(f"{url}/api/version") as response:
                    return {
                        'provider': provider,
                        'connected': response.status == 200,
                        'status_code': response.status,
                        'base_url': url,
                        'message': 'Connected successfully' if response.status == 200 else f'HTTP {response.status}'
                    }
            
            elif provider_lower == 'lmstudio':
                url = base_url or "http://localhost:1234"
                session = await self._get_session()
                async with session.get(f"{url}/v1/models") as response:
                    return {
                        'provider': provider,
                        'connected': response.status == 200,
                        'status_code': response.status,
                        'base_url': url,
                        'message': 'Connected successfully' if response.status == 200 else f'HTTP {response.status}'
                    }
            
            else:
                return {
                    'provider': provider,
                    'connected': False,
                    'message': f'Provider {provider} does not support connectivity testing'
                }
                
        except Exception as e:
            return {
                'provider': provider,
                'connected': False,
                'error': str(e),
                'message': f'Connection failed: {str(e)}'
            }

# Global instance
_discovery_service = None

def get_discovery_service() -> ModelDiscoveryService:
    """Get global model discovery service instance."""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = ModelDiscoveryService()
    return _discovery_service