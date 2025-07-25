# llm/providers/openai_llm.py

"""
OpenAI LLM provider implementation.

Provides integration with OpenAI's GPT models including GPT-3.5, GPT-4, and embeddings.
"""

import asyncio
import logging
from typing import List, Union, AsyncGenerator, Optional, Dict, Any

from ..base.interfaces import BaseLLM, StreamingLLM
from ..base.models import ModelConfig, LLMResponse, UsageMetrics, ModelInfo
from ..base.exceptions import LLMError, LLMProviderError
from .base_provider import BaseProviderMixin, ProviderUtils, requires_api_key, retry_on_rate_limit

logger = logging.getLogger(__name__)

class OpenAILLM(StreamingLLM, BaseProviderMixin):
    """OpenAI-specific LLM implementation with streaming support."""
    
    def __init__(self, api_key: str):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
        """
        super().__init__("openai")
        self.api_key = api_key
        self._client = None
        
        # OpenAI model pricing (per 1K tokens)
        self._model_pricing = {
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004},
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-32k': {'input': 0.06, 'output': 0.12},
            'gpt-4-turbo-preview': {'input': 0.01, 'output': 0.03},
            'text-embedding-ada-002': {'input': 0.0001, 'output': 0.0},
            'text-embedding-3-small': {'input': 0.00002, 'output': 0.0},
            'text-embedding-3-large': {'input': 0.00013, 'output': 0.0}
        }
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise LLMError("OpenAI library not installed. Install with: pip install openai")
        return self._client
    
    @requires_api_key
    @retry_on_rate_limit(max_retries=3)
    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate text using OpenAI API."""
        client = self._get_client()
        
        self._log_request(config.model_name, len(prompt))
        
        try:
            # Prepare request parameters
            request_params = {
                "model": config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "presence_penalty": config.presence_penalty,
                "frequency_penalty": config.frequency_penalty,
                "stream": stream
            }
            
            # Add stop sequences if specified
            if config.stop_sequences:
                request_params["stop"] = config.stop_sequences
            
            if stream:
                return self._stream_response(client, request_params, config)
            else:
                return await self._complete_response(client, request_params, config)
                
        except Exception as e:
            self._log_error(config.model_name, e)
            raise ProviderUtils.handle_provider_error("openai", e, config.model_name)
    
    async def _complete_response(
        self,
        client,
        request_params: Dict[str, Any],
        config: ModelConfig
    ) -> LLMResponse:
        """Handle complete (non-streaming) response."""
        response, latency = await ProviderUtils.measure_latency(
            client.chat.completions.create,
            **request_params
        )
        
        # Extract response data
        choice = response.choices[0]
        text = choice.message.content
        finish_reason = choice.finish_reason
        
        # Parse usage
        usage_dict = {}
        if hasattr(response, 'usage') and response.usage:
            usage_dict = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            # Calculate estimated cost
            pricing = self._model_pricing.get(config.model_name, {'input': 0, 'output': 0})
            estimated_cost = ProviderUtils.calculate_estimated_cost(
                ProviderUtils.parse_usage_from_dict(usage_dict),
                pricing['input'],
                pricing['output']
            )
            usage_dict['estimated_cost'] = estimated_cost
        
        self._log_response(config.model_name, len(text), latency)
        
        return ProviderUtils.create_response(
            text=text,
            model_name=config.model_name,
            usage_dict=usage_dict,
            finish_reason=finish_reason,
            latency=latency,
            metadata={'provider': 'openai'}
        )
    
    async def _stream_response(
        self,
        client,
        request_params: Dict[str, Any],
        config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        """Handle streaming response."""
        response = await client.chat.completions.create(**request_params)
        
        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
    
    async def stream_generate(
        self,
        prompt: str,
        config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text from the model."""
        async for chunk in await self.generate(prompt, config, stream=True):
            yield chunk
    
    @requires_api_key
    async def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using OpenAI API."""
        client = self._get_client()
        
        try:
            response = await client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            self._log_error("text-embedding-ada-002", e)
            raise ProviderUtils.handle_provider_error("openai", e, "text-embedding-ada-002")
    
    async def get_embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002",
        batch_size: int = 32
    ) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently."""
        client = self._get_client()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = await client.embeddings.create(
                    model=model,
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay between batches to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                raise ProviderUtils.handle_provider_error("openai", e, model)
        
        return all_embeddings
    
    async def list_available_models(self) -> List[ModelInfo]:
        """List available OpenAI models."""
        client = self._get_client()
        
        try:
            models = await client.models.list()
            
            model_infos = []
            for model in models.data:
                # Filter to relevant models
                if any(prefix in model.id for prefix in ['gpt-', 'text-embedding-', 'babbage', 'davinci']):
                    pricing = self._model_pricing.get(model.id)
                    
                    model_infos.append(ModelInfo(
                        model_id=model.id,
                        model_name=model.id,
                        provider="openai",
                        display_name=model.id,
                        cost_per_1k_tokens=pricing['input'] if pricing else None,
                        supports_streaming=model.id.startswith('gpt-'),
                        supports_embeddings=model.id.startswith('text-embedding-'),
                        capabilities=['text-generation'] if model.id.startswith('gpt-') else ['embeddings']
                    ))
            
            return model_infos
            
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {str(e)}")
            return []
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific OpenAI model."""
        available_models = await self.list_available_models()
        
        for model in available_models:
            if model.model_name == model_name:
                return model
        
        return None
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform OpenAI-specific health check."""
        try:
            # Try to list models as a health check
            client = self._get_client()
            models = await client.models.list()
            
            return {
                'provider': 'openai',
                'status': 'healthy',
                'message': f'API accessible, {len(models.data)} models available',
                'available_models': len(models.data),
                'timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            return {
                'provider': 'openai',
                'status': 'unhealthy',
                'message': f'API health check failed: {str(e)}',
                'timestamp': asyncio.get_event_loop().time()
            }
    
    def supports_streaming(self) -> bool:
        """OpenAI supports streaming for chat completions."""
        return True
    
    def supports_embeddings(self) -> bool:
        """OpenAI supports embeddings."""
        return True