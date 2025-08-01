# llm/providers/gemini_llm.py

"""
Google Gemini LLM provider implementation.

Provides integration with Google's Gemini models including streaming support and embeddings.
"""

from typing import Dict, List, Optional, Union, AsyncGenerator, Any
import asyncio
import logging
import json

from ..base.interfaces import BaseLLM, StreamingLLM
from ..base.models import ModelConfig, LLMResponse, ModelInfo
from ..base.exceptions import LLMError, LLMProviderError
from .base_provider import BaseProviderMixin, ProviderUtils, requires_api_key

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    logging.warning("Google GenerativeAI library not available. Install with: pip install google-generativeai")

class GeminiLLM(StreamingLLM, BaseProviderMixin):
    """Google Gemini-specific LLM implementation."""
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google API key for Gemini
        """
        if not GOOGLE_GENAI_AVAILABLE:
            raise LLMError("Google GenerativeAI library not installed. Install with: pip install google-generativeai")
        
        super().__init__("gemini")
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.models = {}  # Cache for model instances
        
        # Safety settings for content generation
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
    def _get_model(self, model_name: str):
        """Get or create a model instance."""
        if model_name not in self.models:
            self.models[model_name] = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=self.safety_settings
            )
        return self.models[model_name]
        
    def _convert_config_to_generation_config(self, config: ModelConfig):
        """Convert ModelConfig to Gemini GenerationConfig."""
        generation_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_output_tokens": config.max_tokens,
        }
        
        # Add top_k if specified
        if config.top_k:
            generation_config["top_k"] = config.top_k
            
        # Add stop sequences if specified
        if config.stop_sequences:
            generation_config["stop_sequences"] = config.stop_sequences
            
        return generation_config

    @requires_api_key
    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate text using Gemini API."""
        self._log_request(config.model_name, len(prompt))
        
        try:
            model = self._get_model(config.model_name)
            generation_config = self._convert_config_to_generation_config(config)
            
            if stream:
                return self._stream_response(model, prompt, generation_config, config)
            else:
                return await self._complete_response(model, prompt, generation_config, config)
                
        except Exception as e:
            self._log_error(config.model_name, e)
            
            # Handle specific Gemini API errors before generic handling
            error_str = str(e)
            if "User location is not supported" in error_str:
                raise LLMError(
                    "Gemini API is not available in your geographic location. "
                    "Please configure an alternative LLM provider (OpenAI, Ollama, etc.) "
                    "or use a VPN to access from a supported region."
                )
            elif "API key" in error_str.lower():
                raise LLMError("Invalid or missing Gemini API key. Please check your configuration.")
            elif "quota" in error_str.lower() or "rate limit" in error_str.lower():
                raise LLMError("Gemini API quota or rate limit exceeded. Please try again later.")
            elif "connection" in error_str.lower():
                raise LLMError("Unable to connect to Gemini API. Please check your internet connection.")
            else:
                raise ProviderUtils.handle_provider_error("gemini", e, config.model_name)
    
    async def _complete_response(
        self,
        model,
        prompt: str,
        generation_config: Dict,
        config: ModelConfig
    ) -> LLMResponse:
        """Handle complete (non-streaming) response."""
        response, latency = await ProviderUtils.measure_latency(
            asyncio.to_thread,
            model.generate_content,
            prompt,
            generation_config=generation_config
        )
        
        # Extract usage information
        usage_metadata = getattr(response, 'usage_metadata', None)
        usage_dict = {
            "prompt_tokens": getattr(usage_metadata, 'prompt_token_count', 0) if usage_metadata else 0,
            "completion_tokens": getattr(usage_metadata, 'candidates_token_count', 0) if usage_metadata else 0,
            "total_tokens": getattr(usage_metadata, 'total_token_count', 0) if usage_metadata else 0
        }
        
        # Determine finish reason
        finish_reason = "stop"
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                finish_reason = str(candidate.finish_reason).lower()
        
        self._log_response(config.model_name, len(response.text), latency)
        
        return ProviderUtils.create_response(
            text=response.text,
            model_name=config.model_name,
            usage_dict=usage_dict,
            finish_reason=finish_reason,
            latency=latency,
            metadata={'provider': 'gemini'}
        )
    
    async def _stream_response(
        self,
        model,
        prompt: str,
        generation_config: Dict,
        config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        """Handle streaming response."""
        try:
            # Gemini streaming response
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            error_str = str(e)
            logging.error(f"Gemini streaming error: {error_str}")
            
            # Handle specific Gemini API errors
            if "User location is not supported" in error_str:
                raise LLMError(
                    "Gemini API is not available in your geographic location. "
                    "Please configure an alternative LLM provider (OpenAI, Ollama, etc.) "
                    "or use a VPN to access from a supported region."
                )
            elif "API key" in error_str.lower():
                raise LLMError("Invalid or missing Gemini API key. Please check your configuration.")
            elif "quota" in error_str.lower() or "rate limit" in error_str.lower():
                raise LLMError("Gemini API quota or rate limit exceeded. Please try again later.")
            elif "connection" in error_str.lower():
                raise LLMError("Unable to connect to Gemini API. Please check your internet connection.")
            else:
                raise LLMError(f"Gemini API error: {error_str}")
    
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
        """Get embeddings using Gemini embedding models."""
        try:
            # Use Gemini's embedding model
            result = await asyncio.to_thread(
                genai.embed_content,
                model="models/embedding-001",  # Gemini's embedding model
                content=text,
                task_type="retrieval_document"
            )
            
            return result['embedding']
            
        except Exception as e:
            self._log_error("embedding-001", e)
            raise ProviderUtils.handle_provider_error("gemini", e, "embedding-001")
    
    async def list_available_models(self) -> List[ModelInfo]:
        """List available Gemini models."""
        try:
            models = await asyncio.to_thread(genai.list_models)
            
            model_infos = []
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    model_infos.append(ModelInfo(
                        model_id=model.name,
                        model_name=model.name,
                        provider="gemini",
                        display_name=model.display_name,
                        description=model.description,
                        context_window=getattr(model, 'input_token_limit', None),
                        max_tokens=getattr(model, 'output_token_limit', None),
                        supports_streaming=True,
                        supports_embeddings=False,
                        capabilities=['text-generation']
                    ))
            
            return model_infos
            
        except Exception as e:
            logger.error(f"Failed to list Gemini models: {str(e)}")
            return []
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific Gemini model."""
        try:
            model_info = await asyncio.to_thread(genai.get_model, model_name)
            return ModelInfo(
                model_id=model_info.name,
                model_name=model_info.name,
                provider="gemini",
                display_name=model_info.display_name,
                description=model_info.description,
                context_window=model_info.input_token_limit,
                max_tokens=model_info.output_token_limit,
                supports_streaming=True,
                supports_embeddings=False,
                capabilities=['text-generation']
            )
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {str(e)}")
            return None
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Gemini-specific health check."""
        try:
            # Try to list models as a health check
            models = await asyncio.to_thread(genai.list_models)
            model_count = len(list(models))
            
            return {
                'provider': 'gemini',
                'status': 'healthy',
                'message': f'API accessible, {model_count} models available',
                'available_models': model_count,
                'timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            return {
                'provider': 'gemini',
                'status': 'unhealthy',
                'message': f'API health check failed: {str(e)}',
                'timestamp': asyncio.get_event_loop().time()
            }
    
    def supports_streaming(self) -> bool:
        """Gemini supports streaming."""
        return True
    
    def supports_embeddings(self) -> bool:
        """Gemini supports embeddings."""
        return True