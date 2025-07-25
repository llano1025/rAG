# llm/providers/lmstudio_llm.py

"""
LM Studio LLM provider implementation.

Provides integration with LM Studio local models using OpenAI-compatible API.
"""

import asyncio
import json
import logging
import numpy as np
from typing import List, Union, AsyncGenerator, Optional, Dict, Any

from ..base.interfaces import BaseLLM, StreamingLLM
from ..base.models import ModelConfig, LLMResponse, ModelInfo
from ..base.exceptions import LLMError
from .base_provider import AsyncHTTPProviderMixin, ProviderUtils

logger = logging.getLogger(__name__)

class LMStudioLLM(StreamingLLM, AsyncHTTPProviderMixin):
    """LM Studio-specific LLM implementation using OpenAI-compatible API."""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        """
        Initialize LM Studio provider.
        
        Args:
            base_url: Base URL for LM Studio server
        """
        super().__init__("lmstudio", base_url)
    
    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate text using LM Studio API."""
        self._log_request(config.model_name, len(prompt))
        
        try:
            # Prepare request payload (OpenAI-compatible)
            payload = {
                "model": config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "stream": stream,
                "stop": config.stop_sequences,
            }
            
            if stream:
                return self._stream_response(payload, config)
            else:
                return await self._complete_response(payload, config)
                
        except Exception as e:
            self._log_error(config.model_name, e)
            raise ProviderUtils.handle_provider_error("lmstudio", e, config.model_name)
    
    async def _complete_response(
        self,
        payload: Dict[str, Any],
        config: ModelConfig
    ) -> LLMResponse:
        """Handle complete (non-streaming) response."""
        session = await self._get_session()
        
        response, latency = await ProviderUtils.measure_latency(
            session.post,
            f"{self.base_url}/v1/chat/completions",
            json=payload
        )
        
        result = await response.json()
        
        # Extract response text
        text = result["choices"][0]["message"]["content"]
        finish_reason = result["choices"][0].get("finish_reason", "stop")
        
        # Extract usage information
        usage_dict = result.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })
        
        self._log_response(config.model_name, len(text), latency)
        
        return ProviderUtils.create_response(
            text=text,
            model_name=config.model_name,
            usage_dict=usage_dict,
            finish_reason=finish_reason,
            latency=latency,
            metadata={'provider': 'lmstudio'}
        )
    
    async def _stream_response(
        self,
        payload: Dict[str, Any],
        config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        """Handle streaming response."""
        session = await self._get_session()
        
        async with session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
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
                            if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                                yield chunk["choices"][0]["delta"]["content"]
                except (json.JSONDecodeError, IndexError, KeyError):
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
        
        Note: LM Studio doesn't support embeddings directly, so this provides
        a simple fallback method using a bag-of-words approach.
        """
        try:
            # LM Studio doesn't support embeddings directly
            # Using a simple fallback method
            words = text.lower().split()
            # Create a simple bag-of-words embedding
            embedding = np.zeros(512)  # Using a fixed size
            for i, word in enumerate(words[:512]):
                embedding[i] = hash(word) % 10000 / 10000.0
            return embedding.tolist()
        except Exception as e:
            self._log_error("embedding", e)
            raise ProviderUtils.handle_provider_error("lmstudio", e, "embedding")
    
    async def list_available_models(self) -> List[ModelInfo]:
        """List available LM Studio models."""
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    result = await response.json()
                    
                    model_infos = []
                    for model in result.get("data", []):
                        model_infos.append(ModelInfo(
                            model_id=model["id"],
                            model_name=model["id"],
                            provider="lmstudio",
                            display_name=model.get("name", model["id"]),
                            description=f"Local LM Studio model: {model['id']}",
                            supports_streaming=True,
                            supports_embeddings=False,  # LM Studio doesn't support embeddings
                            capabilities=['text-generation']
                        ))
                    
                    return model_infos
                else:
                    logger.error(f"Failed to list LM Studio models: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to list LM Studio models: {str(e)}")
            return []
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific LM Studio model."""
        available_models = await self.list_available_models()
        
        for model in available_models:
            if model.model_name == model_name:
                return model
        
        return None
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform LM Studio-specific health check."""
        try:
            session = await self._get_session()
            
            # Check if LM Studio server is accessible
            async with session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    result = await response.json()
                    model_count = len(result.get("data", []))
                    
                    return {
                        'provider': 'lmstudio',
                        'status': 'healthy',
                        'message': f'LM Studio server accessible, {model_count} models available',
                        'available_models': model_count,
                        'server_url': self.base_url,
                        'timestamp': asyncio.get_event_loop().time()
                    }
                else:
                    return {
                        'provider': 'lmstudio',
                        'status': 'unhealthy',
                        'message': f'LM Studio server returned status {response.status}',
                        'server_url': self.base_url,
                        'timestamp': asyncio.get_event_loop().time()
                    }
                    
        except Exception as e:
            return {
                'provider': 'lmstudio',
                'status': 'unhealthy',
                'message': f'Cannot connect to LM Studio server: {str(e)}',
                'server_url': self.base_url,
                'timestamp': asyncio.get_event_loop().time()
            }
    
    def supports_streaming(self) -> bool:
        """LM Studio supports streaming."""
        return True
    
    def supports_embeddings(self) -> bool:
        """LM Studio does not support embeddings directly."""
        return False