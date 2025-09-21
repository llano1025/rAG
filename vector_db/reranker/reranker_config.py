"""
Configuration management for reranker models and settings.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


class PerformanceTier(Enum):
    """Performance tiers for reranker models."""
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


@dataclass
class RerankerModelConfig:
    """Configuration for a specific reranker model."""
    
    name: str
    full_name: str
    description: str
    performance_tier: PerformanceTier
    max_length: int = 512
    batch_size: int = 32
    device: Optional[str] = None
    enabled: bool = True
    
    # Model-specific settings
    use_sentence_transformers: bool = True
    cache_enabled: bool = True
    preload: bool = False
    
    # Performance settings
    default_score_weight: float = 0.5
    min_rerank_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'full_name': self.full_name,
            'description': self.description,
            'performance_tier': self.performance_tier.value,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'device': self.device,
            'enabled': self.enabled,
            'use_sentence_transformers': self.use_sentence_transformers,
            'cache_enabled': self.cache_enabled,
            'preload': self.preload,
            'default_score_weight': self.default_score_weight,
            'min_rerank_score': self.min_rerank_score
        }


@dataclass
class RerankerConfig:
    """Global reranker configuration."""
    
    # Default model settings
    default_model: str = 'ms-marco-MiniLM-L-6-v2'
    enabled: bool = False
    
    # Cache settings
    cache_size: int = 3
    cache_ttl_hours: int = 24
    
    # Performance settings
    default_top_k: Optional[int] = None
    default_score_weight: float = 0.5
    min_rerank_score: Optional[float] = None
    
    # Device settings
    device: Optional[str] = None  # None for auto-detection
    force_cpu: bool = False
    
    # Search integration settings
    enable_for_semantic_search: bool = True
    enable_for_hybrid_search: bool = True
    enable_for_contextual_search: bool = True
    enable_for_chat: bool = True
    
    # Performance thresholds
    max_results_to_rerank: int = 100  # Only rerank top N initial results
    min_results_for_reranking: int = 5  # Don't rerank if fewer than N results
    
    # Model configurations
    models: Dict[str, RerankerModelConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default model configurations."""
        if not self.models:
            self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default model configurations."""
        default_models = [
            RerankerModelConfig(
                name='ms-marco-MiniLM-L-6-v2',
                full_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                description='Efficient MS MARCO trained cross-encoder',
                performance_tier=PerformanceTier.FAST,
                max_length=512,
                batch_size=64,
                default_score_weight=0.6,
                preload=True  # Preload the default fast model
            ),
            RerankerModelConfig(
                name='bge-reranker-base',
                full_name='BAAI/bge-reranker-base',
                description='BGE base reranker model',
                performance_tier=PerformanceTier.BALANCED,
                max_length=512,
                batch_size=32,
                default_score_weight=0.5
            ),
            RerankerModelConfig(
                name='bge-reranker-large',
                full_name='BAAI/bge-reranker-large',
                description='BGE large reranker model (highest quality)',
                performance_tier=PerformanceTier.ACCURATE,
                max_length=512,
                batch_size=16,
                default_score_weight=0.7
            ),
            RerankerModelConfig(
                name='jina-reranker-v1-base-en',
                full_name='jinaai/jina-reranker-v1-base-en',
                description='Jina AI English reranker',
                performance_tier=PerformanceTier.BALANCED,
                max_length=512,
                batch_size=32,
                default_score_weight=0.5
            )
        ]
        
        for model_config in default_models:
            self.models[model_config.name] = model_config
    
    def get_model_config(self, model_name: str) -> Optional[RerankerModelConfig]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def get_enabled_models(self) -> List[RerankerModelConfig]:
        """Get list of enabled model configurations."""
        return [config for config in self.models.values() if config.enabled]
    
    def get_models_by_tier(self, tier: PerformanceTier) -> List[RerankerModelConfig]:
        """Get models by performance tier."""
        return [
            config for config in self.models.values()
            if config.performance_tier == tier and config.enabled
        ]
    
    def add_model(self, config: RerankerModelConfig):
        """Add a model configuration."""
        self.models[config.name] = config
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model configuration."""
        if model_name in self.models:
            del self.models[model_name]
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'default_model': self.default_model,
            'enabled': self.enabled,
            'cache_size': self.cache_size,
            'cache_ttl_hours': self.cache_ttl_hours,
            'default_top_k': self.default_top_k,
            'default_score_weight': self.default_score_weight,
            'min_rerank_score': self.min_rerank_score,
            'device': self.device,
            'force_cpu': self.force_cpu,
            'enable_for_semantic_search': self.enable_for_semantic_search,
            'enable_for_hybrid_search': self.enable_for_hybrid_search,
            'enable_for_contextual_search': self.enable_for_contextual_search,
            'enable_for_chat': self.enable_for_chat,
            'max_results_to_rerank': self.max_results_to_rerank,
            'min_results_for_reranking': self.min_results_for_reranking,
            'models': {name: config.to_dict() for name, config in self.models.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RerankerConfig':
        """Create configuration from dictionary."""
        # Extract model configurations
        models_data = data.pop('models', {})
        models = {}
        
        for name, model_data in models_data.items():
            # Convert performance_tier string to enum
            tier_str = model_data.get('performance_tier', 'balanced')
            try:
                tier = PerformanceTier(tier_str)
            except ValueError:
                tier = PerformanceTier.BALANCED
                logger.warning(f"Unknown performance tier '{tier_str}', using BALANCED")
            
            model_data['performance_tier'] = tier
            models[name] = RerankerModelConfig(**model_data)
        
        # Create config
        config = cls(**data)
        config.models = models
        return config
    
    def save_to_file(self, file_path: str):
        """Save configuration to YAML file."""
        try:
            config_dict = self.to_dict()
            
            with open(file_path, 'w') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved reranker configuration to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save reranker configuration: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'RerankerConfig':
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Reranker config file not found: {file_path}, using defaults")
                return cls()
            
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data:
                logger.warning("Empty reranker config file, using defaults")
                return cls()
            
            config = cls.from_dict(data)
            logger.info(f"Loaded reranker configuration from {file_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load reranker configuration from {file_path}: {e}")
            logger.warning("Using default reranker configuration")
            return cls()


# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    'reranker_config.yaml'
)

# Global configuration instance
_reranker_config: Optional[RerankerConfig] = None


def get_reranker_config(config_path: Optional[str] = None) -> RerankerConfig:
    """
    Get the global reranker configuration.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        RerankerConfig instance
    """
    global _reranker_config
    
    if _reranker_config is None:
        config_file = config_path or DEFAULT_CONFIG_PATH
        _reranker_config = RerankerConfig.load_from_file(config_file)
    
    return _reranker_config


def set_reranker_config(config: RerankerConfig):
    """Set the global reranker configuration."""
    global _reranker_config
    _reranker_config = config


def reload_reranker_config(config_path: Optional[str] = None):
    """Reload the reranker configuration from file."""
    global _reranker_config
    
    config_file = config_path or DEFAULT_CONFIG_PATH
    _reranker_config = RerankerConfig.load_from_file(config_file)
    logger.info("Reranker configuration reloaded")


def create_default_config_file(config_path: Optional[str] = None):
    """Create a default configuration file."""
    config_file = config_path or DEFAULT_CONFIG_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    # Create default configuration
    config = RerankerConfig()
    config.save_to_file(config_file)
    
    logger.info(f"Created default reranker configuration file: {config_file}")


# Environment variable overrides
def apply_env_overrides(config: RerankerConfig) -> RerankerConfig:
    """Apply environment variable overrides to configuration."""
    
    # Global settings
    if os.getenv('RERANKER_ENABLED'):
        config.enabled = os.getenv('RERANKER_ENABLED').lower() == 'true'
    
    if os.getenv('RERANKER_DEFAULT_MODEL'):
        config.default_model = os.getenv('RERANKER_DEFAULT_MODEL')
    
    if os.getenv('RERANKER_DEVICE'):
        config.device = os.getenv('RERANKER_DEVICE')
    
    if os.getenv('RERANKER_FORCE_CPU'):
        config.force_cpu = os.getenv('RERANKER_FORCE_CPU').lower() == 'true'
    
    if os.getenv('RERANKER_CACHE_SIZE'):
        try:
            config.cache_size = int(os.getenv('RERANKER_CACHE_SIZE'))
        except ValueError:
            logger.warning("Invalid RERANKER_CACHE_SIZE value, using default")
    
    if os.getenv('RERANKER_SCORE_WEIGHT'):
        try:
            config.default_score_weight = float(os.getenv('RERANKER_SCORE_WEIGHT'))
        except ValueError:
            logger.warning("Invalid RERANKER_SCORE_WEIGHT value, using default")
    
    return config