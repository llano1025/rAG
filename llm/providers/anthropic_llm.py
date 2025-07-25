# llm/providers/anthropic_llm.py

"""
Anthropic Claude LLM provider implementation.

Provides integration with Anthropic's Claude models including streaming support.
"""

import asyncio
import json
import logging
from typing import List, Union, AsyncGenerator, Optional, Dict, Any

from ..base.interfaces import BaseLLM, StreamingLLM
from ..base.models import ModelConfig, LLMResponse, ModelInfo
from ..base.exceptions import LLMError, LLMProviderError
from .base_provider import AsyncHTTPProviderMixin, ProviderUtils

logger = logging.getLogger(__name__)

class AnthropicLLM(StreamingLLM, AsyncHTTPProviderMixin):
    """Anthropic Claude-specific LLM implementation."""
    
    def __init__(self, api_key: str):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
        """
        super().__init__("anthropic", "https://api.anthropic.com")
        self.api_key = api_key
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    
    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate text using Anthropic API."""
        self._log_request(config.model_name, len(prompt))
        
        try:
            # Prepare request payload
            payload = {
                "model": config.model_name,
                "max_tokens": config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature,
                "top_p": config.top_p,
                "stream": stream
            }
            
            if stream:
                return self._stream_response(payload, config)
            else:
                return await self._complete_response(payload, config)
                
        except Exception as e:
            self._log_error(config.model_name, e)
            raise ProviderUtils.handle_provider_error("anthropic", e, config.model_name)
    
    async def _complete_response(
        self,
        payload: Dict[str, Any],
        config: ModelConfig
    ) -> LLMResponse:
        """Handle complete (non-streaming) response."""
        session = await self._get_session()
        
        response, latency = await ProviderUtils.measure_latency(
            session.post,
            f"{self.base_url}/v1/messages",
            json=payload,
            headers=self.headers
        )
        
        result = await response.json()
        
        # Extract response text from Claude format
        text = result["content"][0]["text"] if result.get("content") else ""
        finish_reason = result.get("stop_reason", "stop_sequence")
        
        # Extract usage information
        usage_dict = result.get("usage", {
            "input_tokens": 0,
            "output_tokens": 0
        })
        
        # Convert to standard format
        usage_dict["prompt_tokens"] = usage_dict.get("input_tokens", 0)
        usage_dict["completion_tokens"] = usage_dict.get("output_tokens", 0)
        usage_dict["total_tokens"] = usage_dict["prompt_tokens"] + usage_dict["completion_tokens"]
        
        self._log_response(config.model_name, len(text), latency)
        
        return ProviderUtils.create_response(
            text=text,
            model_name=config.model_name,
            usage_dict=usage_dict,
            finish_reason=finish_reason,
            latency=latency,
            metadata={'provider': 'anthropic'}
        )
    
    async def _stream_response(
        self,
        payload: Dict[str, Any],
        config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        """Handle streaming response."""
        session = await self._get_session()
        
        async with session.post(
            f"{self.base_url}/v1/messages",
            json=payload,
            headers=self.headers
        ) as response:
            async for line in response.content:
                try:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            if data_str == '[DONE]':
                                break
                            
                            chunk = json.loads(data_str)
                            if chunk.get("type") == "content_block_delta":
                                delta = chunk.get("delta", {})
                                if delta.get("text"):
                                    yield delta["text"]
                except (json.JSONDecodeError, KeyError):
                    continue
    
    async def stream_generate(
        self,
        prompt: str,
        config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text from the model."""
        async for chunk in await self.generate(prompt, config, stream=True):
            yield chunk
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for text.
        
        Note: Anthropic doesn't provide embeddings API, so this will raise an error.
        """
        raise LLMProviderError(
            "Anthropic Claude models do not support embeddings. "
            "Use a different provider like OpenAI for embedding generation."
        )
    
    async def list_available_models(self) -> List[ModelInfo]:
        """List available Anthropic models."""
        # Anthropic doesn't have a models endpoint, so we return known models
        known_models = [
            {
                "id": "claude-sonnet-4-20250514",
                "name": "Claude Sonnet 4",
                "description": "Latest flagship model with enhanced reasoning and multimodal capabilities"
            },
            {
                "id": "claude-3-5-haiku-latest", 
                "name": "Claude 3.5 Haiku",
                "description": "Fastest model, optimized for speed and efficiency"
            },
            {
                "id": "claude-opus-4-20250514",
                "name": "Claude Opus 4",
                "description": "Most powerful model for highly complex reasoning tasks"
            }
        ]
        
        model_infos = []
        for model in known_models:
            model_infos.append(ModelInfo(
                model_id=model["id"],
                model_name=model["id"],
                provider="anthropic",
                display_name=model["name"],
                description=model["description"],
                max_tokens=8192,
                supports_streaming=True,
                supports_embeddings=False,
                capabilities=['text-generation', 'reasoning', 'analysis']
            ))
        
        return model_infos
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific Anthropic model."""
        available_models = await self.list_available_models()
        
        for model in available_models:
            if model.model_name == model_name:
                return model
        
        return None
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Anthropic-specific health check."""
        try:
            # Try a simple completion request
            test_payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Say 'OK' if you can hear me."}]
            }
            
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/v1/messages",
                json=test_payload,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    return {
                        'provider': 'anthropic',
                        'status': 'healthy',
                        'message': 'Anthropic API is accessible and responding',
                        'timestamp': asyncio.get_event_loop().time()
                    }
                else:
                    error_text = await response.text()
                    return {
                        'provider': 'anthropic',
                        'status': 'unhealthy',
                        'message': f'Anthropic API returned status {response.status}: {error_text}',
                        'timestamp': asyncio.get_event_loop().time()
                    }
                    
        except Exception as e:
            return {
                'provider': 'anthropic',
                'status': 'unhealthy',
                'message': f'Cannot connect to Anthropic API: {str(e)}',
                'timestamp': asyncio.get_event_loop().time()
            }
    
    def supports_streaming(self) -> bool:
        """Anthropic supports streaming."""
        return True
    
    def supports_embeddings(self) -> bool:
        """Anthropic does not support embeddings."""
        return False