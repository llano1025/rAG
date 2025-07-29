# llm/config/model_configs.py

"""
Default model configurations and templates.

Provides standard configurations for different LLM providers and models.
"""

from typing import Dict, Optional
from ..base.models import ModelConfig

class ModelConfigTemplates:
    """Predefined model configuration templates."""
    
    # OpenAI Models
    OPENAI_GPT4O = ModelConfig(
        model_name="gpt-4o-2024-08-06",  # latest multimodal flagship model
        max_tokens=4096,      # per response limit (context is much larger)
        temperature=0.7,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        context_window=128000  # confirmed 128k context
    )

    OPENAI_GPT41 = ModelConfig(
        model_name="gpt-4.1-2025-04-14",
        max_tokens=4096,           # per response limit
        temperature=0.7,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        context_window=128000     # 128k context supported
    )

    OPENAI_O4_MINI = ModelConfig(
        model_name="o4-mini-2025-04-16",  # latest smaller o4 model, successor to o1-mini
        max_tokens=4096,           # per response token limit
        temperature=0.7,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        context_window=128000     # also supports 128k context
    )

    OPENAI_O3 = ModelConfig(
        model_name="o3-2025-04-16",  # o3 model
        max_tokens=4096,
        temperature=0.7,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        context_window=16384       # 16k context
    )
    
    # Google Gemini Models
    GEMINI_25_PRO = ModelConfig(
        model_name="gemini-2.5-pro",
        max_tokens=65536,
        temperature=0.7,
        top_p=0.95,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        context_window=1048576, # 1 million tokens [25, 35]
        provider_config={
            "safety_settings": "default"
        }
    )
    
    GEMINI_25_FLASH = ModelConfig(
        model_name="gemini-2.5-flash",
        max_tokens=65536,
        temperature=0.7,
        top_p=0.95,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        context_window=1048576, # 1 million tokens [25]
        provider_config={
            "safety_settings": "default"
        }
    )
    
    # Anthropic Claude Models
    ANTHROPIC_CLAUDE_4_SONNET = ModelConfig(
        model_name="claude-sonnet-4-20250514",
        max_tokens=8192,
        temperature=0.7,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        context_window=200000  # 200k tokens
    )
    
    ANTHROPIC_CLAUDE_35_HAIKU = ModelConfig(
        model_name="claude-3-5-haiku-latest",
        max_tokens=8192,
        temperature=0.7,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        context_window=200000  # 200k tokens
    )
    
    ANTHROPIC_CLAUDE_4_OPUS = ModelConfig(
        model_name="claude-opus-4-20250514",
        max_tokens=4096,
        temperature=0.7,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        context_window=200000  # 200k tokens
    )
    
    # Ollama Models (Local)
    OLLAMA_LLAMA32_3B = ModelConfig(
        model_name="llama3.2:latest",
        max_tokens=8192,
        temperature=0.7,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        top_k=40,
        repeat_penalty=1.1,
        context_window=8192,
        api_base="http://localhost:11434"
    )
    
    # LM Studio Models (Local)
    LMSTUDIO_LOCAL = ModelConfig(
        model_name="local-model",
        max_tokens=2048,
        temperature=0.7,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        context_window=4096,
        api_base="http://localhost:1234"
    )
    
    @classmethod
    def get_all_templates(cls) -> Dict[str, ModelConfig]:
        """Get all predefined configuration templates."""
        templates = {}
        
        # Get all class attributes that are ModelConfig instances
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name.isupper():
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, ModelConfig):
                    templates[attr_name.lower()] = attr_value
        
        return templates
    
    @classmethod
    def get_template(cls, template_name: str) -> Optional[ModelConfig]:
        """Get a specific configuration template."""
        templates = cls.get_all_templates()
        return templates.get(template_name.lower())
    
    @classmethod
    def get_provider_templates(cls, provider: str) -> Dict[str, ModelConfig]:
        """Get all templates for a specific provider."""
        all_templates = cls.get_all_templates()
        provider_lower = provider.lower()
        
        return {
            name: config 
            for name, config in all_templates.items() 
            if name.startswith(provider_lower)
        }

def get_default_config(provider: str = "openai", model: str = "gpt-3.5-turbo") -> ModelConfig:
    """
    Get a default configuration for a provider and model.
    
    Args:
        provider: Provider name (openai, gemini, ollama, lmstudio)
        model: Model name
        
    Returns:
        ModelConfig with default settings
    """
    provider_lower = provider.lower()
    model_lower = model.lower().replace("-", "_").replace(".", "_")
    
    # Try to find exact template match
    template_name = f"{provider_lower}_{model_lower}"
    template = ModelConfigTemplates.get_template(template_name)
    
    if template:
        return template
    
    # Fallback to provider defaults
    if provider_lower == "openai":
        return ModelConfigTemplates.OPENAI_GPT4O
    elif provider_lower == "gemini":
        return ModelConfigTemplates.GEMINI_25_FLASH
    elif provider_lower == "anthropic":
        return ModelConfigTemplates.ANTHROPIC_CLAUDE_4_SONNET
    elif provider_lower == "ollama":
        return ModelConfigTemplates.OLLAMA_LLAMA32_3B
    elif provider_lower == "lmstudio":
        return ModelConfigTemplates.LMSTUDIO_LOCAL
    else:
        # Ultimate fallback
        return ModelConfig(
            model_name=model,
            max_tokens=2048,
            temperature=0.7,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )

def create_custom_config(
    model_name: str,
    provider: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    **kwargs
) -> ModelConfig:
    """
    Create a custom model configuration.
    
    Args:
        model_name: Name of the model
        provider: Provider name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        **kwargs: Additional configuration parameters
        
    Returns:
        Custom ModelConfig instance
    """
    # Start with provider defaults
    base_config = get_default_config(provider, model_name)
    
    # Override with provided parameters
    config_dict = base_config.to_dict()
    config_dict.update({
        'model_name': model_name,
        'max_tokens': max_tokens,
        'temperature': temperature,
        **kwargs
    })
    
    return ModelConfig.from_dict(config_dict)