# llm/base/models.py

"""
Data models for LLM operations.

Provides structured data classes for LLM configuration, responses, and metadata.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    model_name: str
    max_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    
    # Optional provider-specific settings
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    
    # Additional parameters for specific models
    stop_sequences: Optional[List[str]] = None
    repeat_penalty: Optional[float] = None
    top_k: Optional[int] = None
    context_window: Optional[int] = None
    
    # Provider-specific extensions
    provider_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.presence_penalty < -2.0 or self.presence_penalty > 2.0:
            raise ValueError("presence_penalty must be between -2.0 and 2.0")
        if self.frequency_penalty < -2.0 or self.frequency_penalty > 2.0:
            raise ValueError("frequency_penalty must be between -2.0 and 2.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_name': self.model_name,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'presence_penalty': self.presence_penalty,
            'frequency_penalty': self.frequency_penalty,
            'api_base': self.api_base,
            'stop_sequences': self.stop_sequences,
            'repeat_penalty': self.repeat_penalty,
            'top_k': self.top_k,
            'context_window': self.context_window,
            'provider_config': self.provider_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if v is not None})

@dataclass
class UsageMetrics:
    """Token usage and cost metrics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: Optional[float] = None
    
    def __add__(self, other: 'UsageMetrics') -> 'UsageMetrics':
        """Add two usage metrics together."""
        return UsageMetrics(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost=(
                (self.estimated_cost or 0) + (other.estimated_cost or 0)
                if self.estimated_cost is not None or other.estimated_cost is not None
                else None
            )
        )

class LLMResponse:
    """Structured response from LLM models."""
    
    def __init__(
        self,
        text: str,
        model_name: str,
        usage: Optional[UsageMetrics] = None,
        finish_reason: str = "stop",
        latency: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.text = text
        self.model_name = model_name
        self.usage = usage or UsageMetrics()
        self.finish_reason = finish_reason
        self.latency = latency
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            'text': self.text,
            'model_name': self.model_name,
            'usage': {
                'prompt_tokens': self.usage.prompt_tokens,
                'completion_tokens': self.usage.completion_tokens,
                'total_tokens': self.usage.total_tokens,
                'estimated_cost': self.usage.estimated_cost
            },
            'finish_reason': self.finish_reason,
            'latency': self.latency,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMResponse':
        """Create response from dictionary."""
        usage_data = data.get('usage', {})
        usage = UsageMetrics(
            prompt_tokens=usage_data.get('prompt_tokens', 0),
            completion_tokens=usage_data.get('completion_tokens', 0),
            total_tokens=usage_data.get('total_tokens', 0),
            estimated_cost=usage_data.get('estimated_cost')
        )
        
        timestamp = None
        if 'timestamp' in data:
            timestamp = datetime.fromisoformat(data['timestamp'])
        
        return cls(
            text=data['text'],
            model_name=data['model_name'],
            usage=usage,
            finish_reason=data.get('finish_reason', 'stop'),
            latency=data.get('latency', 0.0),
            metadata=data.get('metadata', {}),
            timestamp=timestamp
        )
    
    def __str__(self) -> str:
        return self.text
    
    def __len__(self) -> int:
        return len(self.text)

@dataclass
class ModelInfo:
    """Information about an LLM model."""
    model_id: str
    model_name: str
    provider: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    cost_per_1k_tokens: Optional[float] = None
    supports_streaming: bool = True
    supports_embeddings: bool = False
    capabilities: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.model_name
        if self.capabilities is None:
            self.capabilities = []