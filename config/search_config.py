"""
Professional search configuration for RAG system.
Centralizes all search-related parameters and strategies.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

class SearchMode(Enum):
    """Available search modes."""
    TEXT = "text"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"

class FusionMethod(Enum):
    """Available result fusion methods."""
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"
    COMB_SUM = "comb_sum"
    SIMPLE = "simple"

@dataclass
class QueryProcessingConfig:
    """Configuration for query processing."""
    # Stop words to filter out
    stop_words: List[str] = field(default_factory=lambda: [
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
    ])
    
    # Minimum word length for concepts
    min_concept_length: int = 2
    
    # Maximum number of terms to extract
    max_primary_terms: int = 10
    max_secondary_terms: int = 5
    
    # Enable query expansion
    enable_expansion: bool = True
    expansion_threshold: float = 0.8  # Expand if confidence < threshold
    
    # Concept indicators (business document terms)
    concept_indicators: List[str] = field(default_factory=lambda: [
        'letter', 'document', 'report', 'memo', 'circular', 'notice', 'policy',
        'procedure', 'guideline', 'manual', 'contract', 'agreement', 'proposal',
        'invoice', 'receipt', 'statement', 'summary', 'analysis', 'review'
    ])

@dataclass 
class TextSearchConfig:
    """Configuration for text search."""
    # Search field weights
    title_weight: float = 0.3
    filename_weight: float = 0.25
    description_weight: float = 0.2
    content_weight: float = 0.15
    tag_weight: float = 0.1
    
    # Search strategy weights
    primary_term_weight: float = 0.8
    secondary_term_weight: float = 0.3
    phrase_weight: float = 0.9
    
    # Scoring parameters
    min_score: float = 0.1
    max_score: float = 1.0
    multiple_term_bonus: float = 0.1
    concept_metadata_bonus: float = 0.05

@dataclass
class SemanticSearchConfig:
    """Configuration for semantic search."""
    # Embedding strategy weights
    full_query_weight: float = 0.4
    concepts_weight: float = 0.6
    expanded_weight: float = 0.3
    
    # Search parameters
    default_limit: int = 10
    max_limit: int = 100
    min_similarity_threshold: float = 0.1
    confidence_threshold: float = 0.8  # For expansion
    
    # Multi-embedding search
    enable_multi_embedding: bool = True
    embedding_fusion_limit_multiplier: int = 2  # Get 2x results for fusion

@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    # Default fusion method
    default_fusion_method: FusionMethod = FusionMethod.RRF
    
    # RRF parameters
    rrf_k_parameter: int = 60
    rrf_text_weight: float = 0.6
    rrf_semantic_weight: float = 0.4
    
    # Weighted fusion parameters
    weighted_text_weight: float = 0.6
    weighted_semantic_weight: float = 0.4
    
    # Query-type specific strategies
    fusion_strategies: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "exact_phrase": {
            "method": "rrf",
            "text_weight": 0.8,
            "semantic_weight": 0.2,
            "k_parameter": 30
        },
        "multi_concept": {
            "method": "rrf", 
            "text_weight": 0.5,
            "semantic_weight": 0.5,
            "k_parameter": 60
        },
        "single_concept": {
            "method": "rrf",
            "text_weight": 0.3,
            "semantic_weight": 0.7,
            "k_parameter": 45
        },
        "document_lookup": {
            "method": "weighted",
            "text_weight": 0.9,
            "semantic_weight": 0.1
        },
        "temporal_concept": {
            "method": "rrf",
            "text_weight": 0.7,
            "semantic_weight": 0.3,
            "k_parameter": 50
        }
    })

@dataclass
class CachingConfig:
    """Configuration for search result caching."""
    enable_caching: bool = True
    cache_duration_hours: int = 24
    max_cache_entries: int = 10000
    cache_key_components: List[str] = field(default_factory=lambda: [
        'query', 'user_id', 'filters', 'search_type'
    ])

@dataclass
class PerformanceConfig:
    """Configuration for search performance."""
    # Timeout settings (milliseconds)
    text_search_timeout: int = 5000
    semantic_search_timeout: int = 10000
    hybrid_search_timeout: int = 15000
    
    # Parallel processing
    enable_parallel_search: bool = True
    max_concurrent_searches: int = 10
    
    # Result limits
    default_page_size: int = 20
    max_page_size: int = 100
    max_total_results: int = 1000

@dataclass
class SearchConfig:
    """Master search configuration."""
    # Sub-configurations
    query_processing: QueryProcessingConfig = field(default_factory=QueryProcessingConfig)
    text_search: TextSearchConfig = field(default_factory=TextSearchConfig)
    semantic_search: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)
    hybrid_search: HybridSearchConfig = field(default_factory=HybridSearchConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Global settings
    default_search_mode: SearchMode = SearchMode.HYBRID
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Feature flags
    enable_query_suggestions: bool = True
    enable_search_analytics: bool = True
    enable_result_highlighting: bool = True
    enable_query_expansion: bool = True
    
    def get_fusion_strategy(self, query_type: str) -> Dict[str, Any]:
        """Get fusion strategy for a specific query type."""
        return self.hybrid_search.fusion_strategies.get(
            query_type,
            {
                "method": "rrf",
                "text_weight": 0.6,
                "semantic_weight": 0.4,
                "k_parameter": 60
            }
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Validate weights sum to reasonable values
        text_weights = (
            self.text_search.title_weight +
            self.text_search.filename_weight +
            self.text_search.description_weight +
            self.text_search.content_weight +
            self.text_search.tag_weight
        )
        
        if text_weights < 0.8 or text_weights > 1.2:
            issues.append(f"Text search weights sum to {text_weights:.2f}, should be close to 1.0")
        
        # Validate RRF weights
        rrf_weights = (
            self.hybrid_search.rrf_text_weight +
            self.hybrid_search.rrf_semantic_weight
        )
        
        if abs(rrf_weights - 1.0) > 0.1:
            issues.append(f"RRF weights sum to {rrf_weights:.2f}, should sum to 1.0")
        
        # Validate thresholds
        if self.text_search.min_score >= self.text_search.max_score:
            issues.append("Text search min_score must be less than max_score")
        
        if self.semantic_search.min_similarity_threshold < 0 or self.semantic_search.min_similarity_threshold > 1:
            issues.append("Semantic similarity threshold must be between 0 and 1")
        
        # Validate performance limits
        if self.performance.default_page_size > self.performance.max_page_size:
            issues.append("Default page size cannot exceed max page size")
        
        return issues

# Global search configuration instance
_search_config: Optional[SearchConfig] = None

def get_search_config() -> SearchConfig:
    """Get the global search configuration instance."""
    global _search_config
    if _search_config is None:
        _search_config = SearchConfig()
        
        # Validate configuration on first load
        issues = _search_config.validate()
        if issues:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Search configuration issues: {issues}")
    
    return _search_config

def load_search_config_from_dict(config_dict: Dict[str, Any]) -> SearchConfig:
    """Load search configuration from dictionary."""
    # This would be implemented to load from environment variables,
    # config files, or database settings in a full implementation
    config = SearchConfig()
    
    # Example of loading specific values
    if 'default_search_mode' in config_dict:
        config.default_search_mode = SearchMode(config_dict['default_search_mode'])
    
    if 'hybrid_search' in config_dict:
        hybrid_config = config_dict['hybrid_search']
        if 'rrf_text_weight' in hybrid_config:
            config.hybrid_search.rrf_text_weight = hybrid_config['rrf_text_weight']
        if 'rrf_semantic_weight' in hybrid_config:
            config.hybrid_search.rrf_semantic_weight = hybrid_config['rrf_semantic_weight']
    
    return config

# Environment-specific configurations
DEVELOPMENT_CONFIG = {
    "enable_logging": True,
    "log_level": "DEBUG",
    "caching": {"enable_caching": False},
    "performance": {
        "text_search_timeout": 10000,
        "semantic_search_timeout": 20000
    }
}

PRODUCTION_CONFIG = {
    "enable_logging": True,
    "log_level": "INFO", 
    "caching": {"enable_caching": True, "cache_duration_hours": 12},
    "performance": {
        "text_search_timeout": 3000,
        "semantic_search_timeout": 5000,
        "enable_parallel_search": True
    }
}

TESTING_CONFIG = {
    "enable_logging": False,
    "caching": {"enable_caching": False},
    "performance": {
        "default_page_size": 5,
        "max_page_size": 10
    }
}