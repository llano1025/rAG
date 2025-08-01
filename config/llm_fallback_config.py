"""
LLM Provider Fallback Configuration

This module provides configuration for LLM provider fallback strategies,
allowing the system to gracefully degrade when primary providers fail.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

class FallbackStrategy(Enum):
    """Fallback strategies for LLM provider failures."""
    NONE = "none"           # No fallback, fail immediately
    SIMPLE = "simple"       # Try next provider in list
    WEIGHTED = "weighted"   # Try providers based on weights/priorities
    ERROR_AWARE = "error_aware"  # Choose fallback based on error type

@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    provider_name: str
    model_name: str
    priority: int = 1  # Higher priority = tried first
    weight: float = 1.0  # Used for weighted selection
    enabled: bool = True
    api_key_env_var: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 30
    
    # Error-specific configuration
    handle_geographic_errors: bool = True
    handle_quota_errors: bool = True
    handle_auth_errors: bool = False  # Usually don't fallback on auth errors

@dataclass
class FallbackConfig:
    """Main fallback configuration."""
    strategy: FallbackStrategy = FallbackStrategy.ERROR_AWARE
    max_fallback_attempts: int = 3
    fallback_timeout_seconds: int = 60
    
    # Provider configurations in order of preference
    providers: List[ProviderConfig] = field(default_factory=list)
    
    # Error-specific fallback rules
    geographic_error_fallbacks: List[str] = field(default_factory=lambda: ["openai", "ollama"])
    quota_error_fallbacks: List[str] = field(default_factory=lambda: ["ollama", "openai"]) 
    auth_error_fallbacks: List[str] = field(default_factory=list)  # Usually no fallback
    connection_error_fallbacks: List[str] = field(default_factory=lambda: ["ollama"])
    
    def get_fallback_providers(self, error_type: str, current_provider: str) -> List[str]:
        """Get list of fallback providers for a specific error type."""
        fallback_map = {
            "geographic_restriction": self.geographic_error_fallbacks,
            "quota_exceeded": self.quota_error_fallbacks,
            "auth_error": self.auth_error_fallbacks,
            "connection_error": self.connection_error_fallbacks,
            "timeout_error": self.connection_error_fallbacks,
        }
        
        fallbacks = fallback_map.get(error_type, [])
        # Remove current provider from fallback list
        return [p for p in fallbacks if p != current_provider]
    
    def get_enabled_providers(self) -> List[ProviderConfig]:
        """Get list of enabled providers sorted by priority."""
        enabled = [p for p in self.providers if p.enabled]
        return sorted(enabled, key=lambda x: x.priority, reverse=True)

# Default configuration for common setups
def create_default_fallback_config() -> FallbackConfig:
    """Create a sensible default fallback configuration."""
    providers = [
        ProviderConfig(
            provider_name="gemini",
            model_name="gemini-1.5-flash",
            priority=3,
            api_key_env_var="GEMINI_API_KEY",
            handle_geographic_errors=True,
            handle_quota_errors=True
        ),
        ProviderConfig(
            provider_name="openai",
            model_name="gpt-3.5-turbo",
            priority=2,
            api_key_env_var="OPENAI_API_KEY",
            handle_geographic_errors=False,  # Usually available globally
            handle_quota_errors=True
        ),
        ProviderConfig(
            provider_name="ollama",
            model_name="llama3.2",
            priority=1,
            base_url="http://localhost:11434",
            handle_geographic_errors=False,  # Local provider
            handle_quota_errors=False       # No quota limits
        )
    ]
    
    return FallbackConfig(
        strategy=FallbackStrategy.ERROR_AWARE,
        providers=providers,
        max_fallback_attempts=2,
        geographic_error_fallbacks=["openai", "ollama"],
        quota_error_fallbacks=["ollama", "openai"],
        connection_error_fallbacks=["ollama"]
    )

def create_production_fallback_config() -> FallbackConfig:
    """Create a production-optimized fallback configuration."""
    providers = [
        ProviderConfig(
            provider_name="openai",
            model_name="gpt-4o-mini",
            priority=3,
            api_key_env_var="OPENAI_API_KEY",
            max_retries=2,
            timeout_seconds=20
        ),
        ProviderConfig(
            provider_name="gemini",
            model_name="gemini-1.5-flash",
            priority=2,
            api_key_env_var="GEMINI_API_KEY",
            handle_geographic_errors=True,
            max_retries=1,  # Quick fallback for geographic issues
            timeout_seconds=15
        ),
        # Ollama as final fallback for high availability
        ProviderConfig(
            provider_name="ollama",
            model_name="llama3.2",
            priority=1,
            base_url="http://localhost:11434",
            enabled=False  # Enable only if Ollama is set up
        )
    ]
    
    return FallbackConfig(
        strategy=FallbackStrategy.ERROR_AWARE,
        providers=providers,
        max_fallback_attempts=2,
        fallback_timeout_seconds=45,
        geographic_error_fallbacks=["openai"],
        quota_error_fallbacks=["openai"],  # Assume OpenAI has higher limits
    )

def create_local_development_config() -> FallbackConfig:
    """Create a configuration optimized for local development."""
    providers = [
        ProviderConfig(
            provider_name="ollama",
            model_name="llama3.2",
            priority=3,
            base_url="http://localhost:11434",
            timeout_seconds=60  # Local models can be slower
        ),
        ProviderConfig(
            provider_name="gemini",
            model_name="gemini-1.5-flash",
            priority=2,
            api_key_env_var="GEMINI_API_KEY",
            enabled=False  # Disabled by default for dev
        ),
        ProviderConfig(
            provider_name="openai",
            model_name="gpt-3.5-turbo",
            priority=1,
            api_key_env_var="OPENAI_API_KEY",
            enabled=False  # Disabled by default for dev
        )
    ]
    
    return FallbackConfig(
        strategy=FallbackStrategy.SIMPLE,
        providers=providers,
        max_fallback_attempts=1,
        geographic_error_fallbacks=[],  # Use only local providers
        quota_error_fallbacks=[]
    )

# Global configuration instance
_fallback_config: Optional[FallbackConfig] = None

def get_fallback_config() -> FallbackConfig:
    """Get the global fallback configuration."""
    global _fallback_config
    if _fallback_config is None:
        _fallback_config = create_default_fallback_config()
    return _fallback_config

def set_fallback_config(config: FallbackConfig):
    """Set the global fallback configuration."""
    global _fallback_config
    _fallback_config = config

def load_fallback_config_from_env() -> FallbackConfig:
    """Load fallback configuration from environment variables."""
    import os
    
    # Determine environment
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return create_production_fallback_config()
    elif env == "development":
        return create_local_development_config()
    else:
        return create_default_fallback_config()

# Configuration presets for easy selection
CONFIG_PRESETS = {
    "default": create_default_fallback_config,
    "production": create_production_fallback_config,
    "development": create_local_development_config,
}

def get_config_preset(preset_name: str) -> FallbackConfig:
    """Get a predefined configuration preset."""
    if preset_name in CONFIG_PRESETS:
        return CONFIG_PRESETS[preset_name]()
    else:
        raise ValueError(f"Unknown config preset: {preset_name}. Available: {list(CONFIG_PRESETS.keys())}")