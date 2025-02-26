# llm/response_handler.py

from typing import Dict, List, Optional, Union, AsyncGenerator
import json
import logging
from datetime import datetime

from .model_manager import LLMResponse

class ResponseError(Exception):
    """Raised when response processing fails."""
    pass

class ResponseHandler:
    """Handles processing and validation of LLM responses."""
    
    def __init__(self, schema_validator: Optional[callable] = None):
        self.schema_validator = schema_validator
        self.response_history: List[Dict] = []

    async def process_streaming_response(
        self,
        response_stream: AsyncGenerator[str, None],
        callback: Optional[callable] = None
    ) -> str:
        """Process streaming response with optional callback."""
        full_response = []
        
        try:
            async for chunk in response_stream:
                full_response.append(chunk)
                if callback:
                    await callback(chunk)
                    
            complete_response = "".join(full_response)
            self._log_response(complete_response)
            return complete_response
            
        except Exception as e:
            logging.error(f"Error processing stream: {str(e)}")
            raise ResponseError(f"Stream processing failed: {str(e)}")

    def process_response(self, response: LLMResponse) -> Dict:
        """Process and validate non-streaming response."""
        try:
            if self.schema_validator:
                validated_response = self.schema_validator(response.text)
            else:
                validated_response = response.text
                
            self._log_response(response)
            return {
                "content": validated_response,
                "metadata": {
                    "model": response.model_name,
                    "usage": response.usage,
                    "latency": response.latency,
                    "finish_reason": response.finish_reason
                }
            }
            
        except Exception as e:
            logging.error(f"Error processing response: {str(e)}")
            raise ResponseError(f"Response processing failed: {str(e)}")

    def _log_response(self, response: Union[str, LLMResponse]):
        """Log response for history tracking."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "response": response.text if isinstance(response, LLMResponse) else response
        }
        
        if isinstance(response, LLMResponse):
            entry.update({
                "model": response.model_name,
                "usage": response.usage,
                "latency": response.latency
            })
            
        self.response_history.append(entry)