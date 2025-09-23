# llm/providers/ollama_llm.py

"""
Ollama LLM provider implementation.

Provides integration with Ollama local models including streaming support and embeddings.
"""

import asyncio
import json
import logging
from typing import List, Union, AsyncGenerator, Optional, Dict, Any

from ..base.interfaces import BaseLLM, StreamingLLM
from ..base.models import ModelConfig, LLMResponse, ModelInfo
from ..base.exceptions import LLMError
from .base_provider import AsyncHTTPProviderMixin, ProviderUtils

logger = logging.getLogger(__name__)

class OllamaLLM(StreamingLLM, AsyncHTTPProviderMixin):
    """Ollama-specific LLM implementation for local models."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.
        
        Args:
            base_url: Base URL for Ollama server
        """
        # Set attributes directly to avoid MRO issues
        self.provider_name = "ollama"
        self.base_url = base_url.rstrip('/')
        self._session = None
        self._health_check_cache = None
        self._health_check_time = None
        self._health_check_ttl = 300
    
    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate text using Ollama API."""
        logger.debug(f"[OLLAMA] Starting generation - Model: {config.model_name}, Stream: {stream}, Prompt length: {len(prompt)}")
        self._log_request(config.model_name, len(prompt))
        
        try:
            # Prepare request payload
            payload = {
                "model": config.model_name,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k or 40,
                    "repeat_penalty": config.repeat_penalty or 1.1,
                    "stop": config.stop_sequences or [],
                    "num_predict": config.max_tokens
                }
            }
            
            logger.debug(f"[OLLAMA] Request payload prepared - Model: {config.model_name}, Options: {payload['options']}")
            
            if stream:
                logger.debug(f"[OLLAMA] Initiating streaming response")
                return self.stream_generate(prompt, config)
            else:
                logger.debug(f"[OLLAMA] Initiating complete response")
                return await self._complete_response(payload, config)
                
        except Exception as e:
            logger.error(f"[OLLAMA] Generation failed - Model: {config.model_name}, Error: {str(e)}", exc_info=True)
            self._log_error(config.model_name, e)
            raise ProviderUtils.handle_provider_error("ollama", e, config.model_name)
    
    async def _complete_response(
        self,
        payload: Dict[str, Any],
        config: ModelConfig
    ) -> LLMResponse:
        """Handle complete (non-streaming) response."""
        session = await self._get_session()
        
        response, latency = await ProviderUtils.measure_latency(
            session.post,
            f"{self.base_url}/api/generate",
            json=payload
        )
        
        result = await response.json()
        
        # Extract usage information
        usage_dict = {
            "prompt_tokens": result.get("prompt_eval_count", 0),
            "completion_tokens": result.get("eval_count", 0),
            "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
        }
        
        self._log_response(config.model_name, len(result["response"]), latency)
        
        return ProviderUtils.create_response(
            text=result["response"],
            model_name=config.model_name,
            usage_dict=usage_dict,
            finish_reason="stop",
            latency=latency,
            metadata={
                'provider': 'ollama',
                'eval_duration': result.get('eval_duration'),
                'load_duration': result.get('load_duration')
            }
        )
    
    async def _stream_response(
        self,
        payload: Dict[str, Any],
        config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        """Handle streaming response."""
        logger.debug(f"[OLLAMA] Starting stream response - Model: {config.model_name}, URL: {self.base_url}")
        
        try:
            session = await self._get_session()
            logger.debug(f"[OLLAMA] HTTP session obtained")
            
            chunk_count = 0
            total_content = ""
            
            logger.debug(f"[OLLAMA] Making POST request to {self.base_url}/api/generate")
            
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                logger.debug(f"[OLLAMA] Response received - Status: {response.status}, Content-Type: {response.content_type}")
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"[OLLAMA] HTTP error - Status: {response.status}, Response: {error_text}")
                    raise Exception(f"Ollama request failed with status {response.status}: {error_text}")
                
                logger.debug(f"[OLLAMA] Starting to read streaming content")
                
                async for line in response.content:
                    chunk_count += 1
                    
                    try:
                        # Decode the line
                        line_str = line.decode('utf-8').strip() if isinstance(line, bytes) else line.strip()
                        
                        if not line_str:
                            logger.debug(f"[OLLAMA] Empty line received (chunk {chunk_count})")
                            continue
                        
                        logger.debug(f"[OLLAMA] Raw line received (chunk {chunk_count}): {line_str[:100]}{'...' if len(line_str) > 100 else ''}")
                        
                        # Parse JSON
                        chunk = json.loads(line_str)
                        logger.debug(f"[OLLAMA] Parsed chunk {chunk_count}: {list(chunk.keys())}")
                        
                        # Check for response content
                        if chunk.get("response"):
                            content = chunk["response"]
                            total_content += content
                            logger.debug(f"[OLLAMA] Yielding content (chunk {chunk_count}): '{content}' (length: {len(content)})")
                            yield content
                        
                        # Check for completion
                        if chunk.get("done", False):
                            logger.debug(f"[OLLAMA] Stream completed - Total chunks: {chunk_count}, Total content length: {len(total_content)}")
                            
                            # Log final statistics
                            if "eval_count" in chunk:
                                logger.debug(f"[OLLAMA] Final stats - Eval count: {chunk.get('eval_count')}, Prompt eval count: {chunk.get('prompt_eval_count')}")
                            if "eval_duration" in chunk:
                                eval_duration_ms = chunk.get('eval_duration', 0) / 1000000  # Convert nanoseconds to milliseconds
                                logger.debug(f"[OLLAMA] Eval duration: {eval_duration_ms:.2f}ms")
                            
                            break
                            
                        # Check for errors in the chunk
                        if "error" in chunk:
                            error_msg = chunk["error"]
                            logger.error(f"[OLLAMA] Error in response chunk: {error_msg}")
                            raise Exception(f"Ollama returned error: {error_msg}")
                            
                    except json.JSONDecodeError as json_err:
                        logger.warning(f"[OLLAMA] JSON decode error (chunk {chunk_count}): {str(json_err)}, Line: {line_str[:200]}")
                        continue
                    except Exception as chunk_err:
                        logger.error(f"[OLLAMA] Error processing chunk {chunk_count}: {str(chunk_err)}")
                        continue
                
                logger.debug(f"[OLLAMA] Stream processing completed - Total chunks processed: {chunk_count}")
                
        except Exception as stream_err:
            logger.error(f"[OLLAMA] Stream response failed: {str(stream_err)}", exc_info=True)
            raise
    
    async def stream_generate(
        self,
        prompt: str,
        config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text from the model."""
        # Prepare request payload (same as in generate method)
        payload = {
            "model": config.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k or 40,
                "repeat_penalty": config.repeat_penalty or 1.1,
                "stop": config.stop_sequences or [],
                "num_predict": config.max_tokens
            }
        }
        
        # Delegate to the actual streaming implementation
        async for chunk in self._stream_response(payload, config):
            yield chunk
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using Ollama."""
        session = await self._get_session()
        
        try:
            async with session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": "llama2", "prompt": text}  # Using llama2 as default embedding model
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["embedding"]
                else:
                    error_text = await response.text()
                    raise LLMError(f"Ollama embedding failed: {response.status} - {error_text}")
                    
        except Exception as e:
            self._log_error("llama2", e)
            raise ProviderUtils.handle_provider_error("ollama", e, "llama2")
    
    async def list_available_models(self) -> List[ModelInfo]:
        """List available Ollama models."""
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    result = await response.json()
                    
                    model_infos = []
                    for model in result.get("models", []):
                        model_infos.append(ModelInfo(
                            model_id=model["name"],
                            model_name=model["name"],
                            provider="ollama",
                            display_name=model["name"],
                            description=f"Local Ollama model: {model['name']}",
                            supports_streaming=True,
                            supports_embeddings=True,
                            capabilities=['text-generation', 'embeddings']
                        ))
                    
                    return model_infos
                else:
                    logger.error(f"Failed to list Ollama models: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {str(e)}")
            return []
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific Ollama model."""
        available_models = await self.list_available_models()
        
        for model in available_models:
            if model.model_name == model_name:
                return model
        
        return None
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Ollama-specific health check."""
        logger.debug(f"[OLLAMA] Starting health check - URL: {self.base_url}")
        
        try:
            session = await self._get_session()
            logger.debug(f"[OLLAMA] Health check session obtained")
            
            # Check if Ollama server is accessible
            logger.debug(f"[OLLAMA] Checking server accessibility at {self.base_url}/api/tags")
            
            async with session.get(f"{self.base_url}/api/tags") as response:
                logger.debug(f"[OLLAMA] Health check response - Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    model_count = len(result.get("models", []))
                    models = result.get("models", [])
                    
                    logger.debug(f"[OLLAMA] Health check successful - {model_count} models available")
                    if models:
                        model_names = [model.get("name", "unknown") for model in models]
                        logger.debug(f"[OLLAMA] Available models: {model_names}")
                    
                    return {
                        'provider': 'ollama',
                        'status': 'healthy',
                        'message': f'Ollama server accessible, {model_count} models available',
                        'available_models': model_count,
                        'server_url': self.base_url,
                        'timestamp': asyncio.get_event_loop().time(),
                        'models': model_names if models else []
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"[OLLAMA] Health check failed - Status: {response.status}, Response: {error_text}")
                    
                    return {
                        'provider': 'ollama',
                        'status': 'unhealthy',
                        'message': f'Ollama server returned status {response.status}: {error_text}',
                        'server_url': self.base_url,
                        'timestamp': asyncio.get_event_loop().time()
                    }
                    
        except Exception as e:
            logger.error(f"[OLLAMA] Health check error: {str(e)}", exc_info=True)
            return {
                'provider': 'ollama',
                'status': 'unhealthy',
                'message': f'Cannot connect to Ollama server: {str(e)}',
                'server_url': self.base_url,
                'timestamp': asyncio.get_event_loop().time()
            }
    
    def supports_streaming(self) -> bool:
        """Ollama supports streaming."""
        return True
    
    def supports_embeddings(self) -> bool:
        """Ollama supports embeddings."""
        return True