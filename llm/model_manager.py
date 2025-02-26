# llm/model_manager.py

from typing import Dict, List, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod
import asyncio
import logging
from dataclasses import dataclass
import json
import aiohttp
import numpy as np

# Custom exceptions
class LLMError(Exception):
    """Raised when LLM operations fail."""
    pass

@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    model_name: str
    max_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    # Additional params for specific models
    stop_sequences: Optional[List[str]] = None
    repeat_penalty: Optional[float] = None
    top_k: Optional[int] = None
    context_window: Optional[int] = None

class LLMResponse:
    """Structured response from LLM models."""
    def __init__(
        self,
        text: str,
        model_name: str,
        usage: Dict[str, int],
        finish_reason: str,
        latency: float
    ):
        self.text = text
        self.model_name = model_name
        self.usage = usage
        self.finish_reason = finish_reason
        self.latency = latency

class BaseLLM(ABC):
    """Base class for LLM implementations."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate text from the model."""
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for text."""
        pass

class OpenAILLM(BaseLLM):
    """OpenAI-specific LLM implementation."""
    
    def __init__(self, api_key: str):
        import openai
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        start_time = asyncio.get_event_loop().time()
        
        try:
            response = await self.client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
                stream=stream
            )
            
            if stream:
                async def response_generator():
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return response_generator()
            
            else:
                end_time = asyncio.get_event_loop().time()
                return LLMResponse(
                    text=response.choices[0].message.content,
                    model_name=config.model_name,
                    usage=response.usage.model_dump(),
                    finish_reason=response.choices[0].finish_reason,
                    latency=end_time - start_time
                )
                
        except Exception as e:
            logging.error(f"OpenAI API error: {str(e)}")
            raise LLMError(f"Failed to generate response: {str(e)}")

    async def get_embedding(self, text: str) -> List[float]:
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"OpenAI embedding error: {str(e)}")
            raise LLMError(f"Failed to generate embedding: {str(e)}")

class OllamaLLM(BaseLLM):
    """Ollama-specific LLM implementation."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.session = aiohttp.ClientSession()

    async def __del__(self):
        await self.session.close()

    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        start_time = asyncio.get_event_loop().time()
        
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

            if stream:
                async def stream_generator():
                    async with self.session.post(
                        f"{self.base_url}/api/generate",
                        json=payload
                    ) as response:
                        async for line in response.content:
                            try:
                                chunk = json.loads(line)
                                if chunk.get("response"):
                                    yield chunk["response"]
                            except json.JSONDecodeError:
                                continue
                return stream_generator()
            else:
                async with self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    end_time = asyncio.get_event_loop().time()
                    result = await response.json()
                    
                    return LLMResponse(
                        text=result["response"],
                        model_name=config.model_name,
                        usage={
                            "prompt_tokens": result.get("prompt_eval_count", 0),
                            "completion_tokens": result.get("eval_count", 0),
                            "total_tokens": result.get("total_eval_count", 0)
                        },
                        finish_reason="stop",
                        latency=end_time - start_time
                    )

        except Exception as e:
            logging.error(f"Ollama API error: {str(e)}")
            raise LLMError(f"Failed to generate response: {str(e)}")

    async def get_embedding(self, text: str) -> List[float]:
        try:
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": "llama2", "prompt": text}  # Using llama2 as default embedding model
            ) as response:
                result = await response.json()
                return result["embedding"]
        except Exception as e:
            logging.error(f"Ollama embedding error: {str(e)}")
            raise LLMError(f"Failed to generate embedding: {str(e)}")

class LMStudioLLM(BaseLLM):
    """LM Studio-specific LLM implementation."""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url.rstrip('/')
        self.session = aiohttp.ClientSession()

    async def __del__(self):
        await self.session.close()

    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        start_time = asyncio.get_event_loop().time()
        
        try:
            # LM Studio uses OpenAI-compatible API
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
                async def stream_generator():
                    async with self.session.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload
                    ) as response:
                        async for line in response.content:
                            try:
                                if line:
                                    chunk = json.loads(line.decode('utf-8').split('data: ')[1])
                                    if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                                        yield chunk["choices"][0]["delta"]["content"]
                            except (json.JSONDecodeError, IndexError, KeyError):
                                continue
                return stream_generator()
            else:
                async with self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload
                ) as response:
                    end_time = asyncio.get_event_loop().time()
                    result = await response.json()
                    
                    return LLMResponse(
                        text=result["choices"][0]["message"]["content"],
                        model_name=config.model_name,
                        usage=result.get("usage", {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }),
                        finish_reason=result["choices"][0].get("finish_reason", "stop"),
                        latency=end_time - start_time
                    )

        except Exception as e:
            logging.error(f"LM Studio API error: {str(e)}")
            raise LLMError(f"Failed to generate response: {str(e)}")

    async def get_embedding(self, text: str) -> List[float]:
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
            logging.error(f"LM Studio embedding error: {str(e)}")
            raise LLMError(f"Failed to generate embedding: {str(e)}")

class ModelManager:
    """Manages multiple LLM models and handles fallback logic."""
    
    def __init__(self):
        self.models: Dict[str, BaseLLM] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.fallback_chain: List[str] = []

    def register_model(
        self,
        model_id: str,
        model: BaseLLM,
        config: ModelConfig,
        fallback_priority: Optional[int] = None
    ):
        """Register a new LLM model."""
        self.models[model_id] = model
        self.configs[model_id] = config
        
        if fallback_priority is not None:
            if fallback_priority < len(self.fallback_chain):
                self.fallback_chain.insert(fallback_priority, model_id)
            else:
                self.fallback_chain.append(model_id)

    async def generate_with_fallback(
        self,
        prompt: str,
        primary_model_id: str,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate text with fallback to other models if primary fails."""
        try:
            return await self.models[primary_model_id].generate(
                prompt,
                self.configs[primary_model_id],
                stream
            )
        except Exception as primary_error:
            logging.warning(f"Primary model {primary_model_id} failed: {str(primary_error)}")
            
            for fallback_id in self.fallback_chain:
                if fallback_id != primary_model_id:
                    try:
                        logging.info(f"Attempting fallback to model {fallback_id}")
                        return await self.models[fallback_id].generate(
                            prompt,
                            self.configs[fallback_id],
                            stream
                        )
                    except Exception as fallback_error:
                        logging.warning(f"Fallback model {fallback_id} failed: {str(fallback_error)}")
                        continue
            
            raise LLMError("All models in fallback chain failed")