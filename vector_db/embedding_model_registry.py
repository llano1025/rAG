# vector_db/embedding_model_registry.py

from typing import Dict, List, Any, Optional, Union
import logging
import json
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    OPENAI = "openai"

@dataclass
class ModelMetadata:
    """Metadata for embedding models."""
    model_id: str
    provider: EmbeddingProvider
    model_name: str
    display_name: str
    description: str
    embedding_dimension: int
    max_input_length: int
    performance_tier: str  # "fast", "balanced", "quality"
    use_cases: List[str]
    language_support: List[str]
    model_size_mb: Optional[int] = None
    inference_speed_ms: Optional[float] = None
    quality_score: Optional[float] = None
    memory_requirements_mb: Optional[int] = None
    gpu_required: bool = False
    api_cost_per_1k_tokens: Optional[float] = None
    license: Optional[str] = None
    created_at: datetime = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for embedding models."""
    model_id: str
    total_embeddings_generated: int = 0
    total_processing_time_ms: float = 0.0
    average_latency_ms: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    last_used: Optional[datetime] = None
    peak_memory_usage_mb: Optional[float] = None
    
    def update_metrics(self, processing_time_ms: float, success: bool = True):
        """Update performance metrics with new data."""
        if success:
            self.total_embeddings_generated += 1
            self.total_processing_time_ms += processing_time_ms
            self.average_latency_ms = self.total_processing_time_ms / self.total_embeddings_generated
        else:
            self.error_count += 1
        
        total_attempts = self.total_embeddings_generated + self.error_count
        self.success_rate = self.total_embeddings_generated / total_attempts if total_attempts > 0 else 1.0
        self.last_used = datetime.utcnow()

class EmbeddingModelRegistry:
    """Registry for managing embedding models and their metadata."""
    
    def __init__(self, registry_file: str = "runtime/storage/embedding_models.json"):
        self.registry_file = Path(registry_file)
        self.models: Dict[str, ModelMetadata] = {}
        self.performance_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self._ensure_directory()
        self._load_default_models()
        self._load_registry()

    def _ensure_directory(self):
        """Ensure registry directory exists."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_default_models(self):
        """Load default model configurations."""
        default_models = [
            # HuggingFace Sentence Transformers Models
            ModelMetadata(
                model_id="hf-mpnet-base-v2",
                provider=EmbeddingProvider.HUGGINGFACE,
                model_name="sentence-transformers/all-mpnet-base-v2",
                display_name="MPNet Base v2",
                description="High-quality general-purpose embedding model with good performance",
                embedding_dimension=768,
                max_input_length=512,
                performance_tier="balanced",
                use_cases=["general", "semantic_search", "question_answering"],
                language_support=["en"],
                model_size_mb=420,
                quality_score=0.85,
                memory_requirements_mb=2048,
                license="Apache 2.0"
            ),
            ModelMetadata(
                model_id="hf-minilm-l6-v2",
                provider=EmbeddingProvider.HUGGINGFACE,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                display_name="MiniLM L6 v2",
                description="Fast and lightweight embedding model for quick processing",
                embedding_dimension=384,
                max_input_length=512,
                performance_tier="fast",
                use_cases=["general", "semantic_search", "clustering"],
                language_support=["en"],
                model_size_mb=90,
                quality_score=0.75,
                memory_requirements_mb=1024,
                license="Apache 2.0"
            ),
            ModelMetadata(
                model_id="hf-e5-large-instruct",
                provider=EmbeddingProvider.HUGGINGFACE,
                model_name="intfloat/multilingual-e5-large-instruct",
                display_name="Multilingual E5 Large Instruct",
                description="Instruction-tuned multilingual embedding model with enhanced performance across 100 languages",
                embedding_dimension=1024,
                max_input_length=512,
                performance_tier="quality",
                use_cases=["multilingual", "instruction_following", "semantic_search", "retrieval"],
                language_support=["en", "zh", "es", "fr", "de", "it", "ja", "ko", "ru", "ar", "multilingual"],
                model_size_mb=560,
                quality_score=0.93,
                memory_requirements_mb=3072,
                license="MIT"
            ),
            ModelMetadata(
                model_id="hf-qwen3-embedding",
                provider=EmbeddingProvider.HUGGINGFACE,
                model_name="Qwen/Qwen3-Embedding-0.6B",
                display_name="Qwen3 Embedding 0.6B",
                description="Efficient multilingual embedding model with instruction-aware capabilities and flexible dimensions",
                embedding_dimension=1024,
                max_input_length=8192,
                performance_tier="balanced",
                use_cases=["multilingual", "text_retrieval", "code_retrieval", "classification", "clustering"],
                language_support=["en", "zh", "es", "fr", "de", "it", "ja", "ko", "ru", "ar", "multilingual"],
                model_size_mb=600,
                quality_score=0.92,
                memory_requirements_mb=2048,
                license="Apache 2.0"
            ),
            ModelMetadata(
                model_id="hf-jina-v3",
                provider=EmbeddingProvider.HUGGINGFACE,
                model_name="jinaai/jina-embeddings-v3",
                display_name="Jina Embeddings v3",
                description="Frontier multilingual embedding model with task-specific LoRA adapters and flexible dimensions",
                embedding_dimension=1024,
                max_input_length=8192,
                performance_tier="quality",
                use_cases=["multilingual", "query_document_retrieval", "clustering", "classification", "text_matching"],
                language_support=["en", "zh", "es", "fr", "de", "it", "ja", "ko", "ru", "ar", "multilingual"],
                model_size_mb=570,
                quality_score=0.94,
                memory_requirements_mb=3072,
                license="Apache 2.0"
            ),
            ModelMetadata(
                model_id="hf-bge-m3",
                provider=EmbeddingProvider.HUGGINGFACE,
                model_name="BAAI/bge-m3",
                display_name="BGE-M3",
                description="Multi-functional, multilingual, multi-granularity embedding model supporting 100+ languages",
                embedding_dimension=1024,
                max_input_length=8192,
                performance_tier="quality",
                use_cases=["multilingual", "dense_retrieval", "multi_vector_retrieval", "sparse_retrieval", "long_documents"],
                language_support=["en", "zh", "es", "fr", "de", "it", "ja", "ko", "ru", "ar", "multilingual"],
                model_size_mb=1340,
                quality_score=0.96,
                memory_requirements_mb=4096,
                license="MIT"
            ),
            
            # Ollama Models
            ModelMetadata(
                model_id="ollama-nomic-embed",
                provider=EmbeddingProvider.OLLAMA,
                model_name="nomic-embed-text",
                display_name="Nomic Embed Text",
                description="High-quality local embedding model via Ollama",
                embedding_dimension=768,
                max_input_length=2048,
                performance_tier="balanced",
                use_cases=["general", "local_deployment", "privacy"],
                language_support=["en"],
                model_size_mb=550,
                quality_score=0.82,
                memory_requirements_mb=2048,
                license="Apache 2.0"
            ),
            ModelMetadata(
                model_id="ollama-mxbai-embed",
                provider=EmbeddingProvider.OLLAMA,
                model_name="mxbai-embed-large",
                display_name="MxBai Embed Large",
                description="Large embedding model optimized for accuracy via Ollama",
                embedding_dimension=1024,
                max_input_length=512,
                performance_tier="quality",
                use_cases=["high_accuracy", "local_deployment", "enterprise"],
                language_support=["en"],
                model_size_mb=970,
                quality_score=0.88,
                memory_requirements_mb=3072,
                license="Apache 2.0"
            ),
            
            # OpenAI Models
            ModelMetadata(
                model_id="openai-ada-002",
                provider=EmbeddingProvider.OPENAI,
                model_name="text-embedding-ada-002",
                display_name="Ada 002",
                description="OpenAI's standard embedding model with good performance",
                embedding_dimension=1536,
                max_input_length=8191,
                performance_tier="balanced",
                use_cases=["general", "cloud", "production"],
                language_support=["en", "multilingual"],
                api_cost_per_1k_tokens=0.0001,
                quality_score=0.87,
                license="Commercial"
            ),
            ModelMetadata(
                model_id="openai-3-small",
                provider=EmbeddingProvider.OPENAI,
                model_name="text-embedding-3-small",
                display_name="Embedding 3 Small",
                description="OpenAI's latest small embedding model with improved efficiency",
                embedding_dimension=1536,
                max_input_length=8191,
                performance_tier="fast",
                use_cases=["cost_effective", "cloud", "high_volume"],
                language_support=["en", "multilingual"],
                api_cost_per_1k_tokens=0.00002,
                quality_score=0.85,
                license="Commercial"
            ),
            ModelMetadata(
                model_id="openai-3-large",
                provider=EmbeddingProvider.OPENAI,
                model_name="text-embedding-3-large",
                display_name="Embedding 3 Large",
                description="OpenAI's highest quality embedding model",
                embedding_dimension=3072,
                max_input_length=8191,
                performance_tier="quality",
                use_cases=["highest_quality", "cloud", "enterprise"],
                language_support=["en", "multilingual"],
                api_cost_per_1k_tokens=0.00013,
                quality_score=0.95,
                license="Commercial"
            )
        ]
        
        for model in default_models:
            self.models[model.model_id] = model
            self.performance_metrics[model.model_id] = ModelPerformanceMetrics(model.model_id)

    def _load_registry(self):
        """Load registry from file."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                # Load models
                for model_data in data.get("models", []):
                    model_data["provider"] = EmbeddingProvider(model_data["provider"])
                    model_data["created_at"] = datetime.fromisoformat(model_data["created_at"])
                    model_data["last_updated"] = datetime.fromisoformat(model_data["last_updated"])
                    model = ModelMetadata(**model_data)
                    self.models[model.model_id] = model
                
                # Load performance metrics
                for metrics_data in data.get("performance_metrics", []):
                    if metrics_data.get("last_used"):
                        metrics_data["last_used"] = datetime.fromisoformat(metrics_data["last_used"])
                    metrics = ModelPerformanceMetrics(**metrics_data)
                    self.performance_metrics[metrics.model_id] = metrics
                    
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")

    def _save_registry(self):
        """Save registry to file."""
        try:
            data = {
                "models": [],
                "performance_metrics": []
            }
            
            # Serialize models
            for model in self.models.values():
                model_dict = asdict(model)
                model_dict["provider"] = model.provider.value
                model_dict["created_at"] = model.created_at.isoformat()
                model_dict["last_updated"] = model.last_updated.isoformat()
                data["models"].append(model_dict)
            
            # Serialize performance metrics
            for metrics in self.performance_metrics.values():
                metrics_dict = asdict(metrics)
                if metrics.last_used:
                    metrics_dict["last_used"] = metrics.last_used.isoformat()
                data["performance_metrics"].append(metrics_dict)
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")

    def register_model(self, model: ModelMetadata) -> bool:
        """Register a new model."""
        try:
            self.models[model.model_id] = model
            if model.model_id not in self.performance_metrics:
                self.performance_metrics[model.model_id] = ModelPerformanceMetrics(model.model_id)
            self._save_registry()
            logger.info(f"Registered model: {model.model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register model {model.model_id}: {e}")
            return False

    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model."""
        try:
            if model_id in self.models:
                del self.models[model_id]
            if model_id in self.performance_metrics:
                del self.performance_metrics[model_id]
            self._save_registry()
            logger.info(f"Unregistered model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister model {model_id}: {e}")
            return False

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        return self.models.get(model_id)

    def list_models(
        self,
        provider: Optional[EmbeddingProvider] = None,
        performance_tier: Optional[str] = None,
        use_case: Optional[str] = None
    ) -> List[ModelMetadata]:
        """List models with optional filtering."""
        models = list(self.models.values())
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        if performance_tier:
            models = [m for m in models if m.performance_tier == performance_tier]
        
        if use_case:
            models = [m for m in models if use_case in m.use_cases]
        
        return models

    def get_recommended_models(self, use_case: str, max_models: int = 3) -> List[ModelMetadata]:
        """Get recommended models for a specific use case."""
        matching_models = [m for m in self.models.values() if use_case in m.use_cases]
        
        # Sort by quality score and performance
        def score_model(model):
            metrics = self.performance_metrics.get(model.model_id)
            quality_score = model.quality_score or 0.5
            success_rate = metrics.success_rate if metrics else 1.0
            recency_score = 1.0
            
            if metrics and metrics.last_used:
                days_since_used = (datetime.utcnow() - metrics.last_used).days
                recency_score = max(0.1, 1.0 - (days_since_used / 30))  # Decay over 30 days
            
            return quality_score * 0.5 + success_rate * 0.3 + recency_score * 0.2
        
        matching_models.sort(key=score_model, reverse=True)
        return matching_models[:max_models]

    def update_performance_metrics(
        self,
        model_id: str,
        processing_time_ms: float,
        success: bool = True,
        memory_usage_mb: Optional[float] = None
    ):
        """Update performance metrics for a model."""
        if model_id not in self.performance_metrics:
            self.performance_metrics[model_id] = ModelPerformanceMetrics(model_id)
        
        metrics = self.performance_metrics[model_id]
        metrics.update_metrics(processing_time_ms, success)
        
        if memory_usage_mb:
            if metrics.peak_memory_usage_mb is None:
                metrics.peak_memory_usage_mb = memory_usage_mb
            else:
                metrics.peak_memory_usage_mb = max(metrics.peak_memory_usage_mb, memory_usage_mb)
        
        # Save periodically (every 10 updates)
        if metrics.total_embeddings_generated % 10 == 0:
            self._save_registry()

    def get_performance_metrics(self, model_id: str) -> Optional[ModelPerformanceMetrics]:
        """Get performance metrics for a model."""
        return self.performance_metrics.get(model_id)

    def get_model_comparison(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models."""
        comparison = {
            "models": {},
            "summary": {}
        }
        
        for model_id in model_ids:
            if model_id in self.models:
                model = self.models[model_id]
                metrics = self.performance_metrics.get(model_id)
                
                comparison["models"][model_id] = {
                    "metadata": asdict(model),
                    "performance": asdict(metrics) if metrics else None
                }
        
        # Add summary statistics
        embedding_dims = [self.models[mid].embedding_dimension for mid in model_ids if mid in self.models]
        quality_scores = [self.models[mid].quality_score for mid in model_ids if mid in self.models and self.models[mid].quality_score]
        
        comparison["summary"] = {
            "total_models": len(model_ids),
            "available_models": len([mid for mid in model_ids if mid in self.models]),
            "dimension_range": [min(embedding_dims), max(embedding_dims)] if embedding_dims else None,
            "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else None
        }
        
        return comparison

    def cleanup_old_metrics(self, days_threshold: int = 90):
        """Clean up old performance metrics."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
        models_to_clean = []
        
        for model_id, metrics in self.performance_metrics.items():
            if metrics.last_used and metrics.last_used < cutoff_date:
                models_to_clean.append(model_id)
        
        for model_id in models_to_clean:
            if model_id in self.performance_metrics:
                # Reset metrics but keep the model registration
                self.performance_metrics[model_id] = ModelPerformanceMetrics(model_id)
        
        if models_to_clean:
            self._save_registry()
            logger.info(f"Cleaned up metrics for {len(models_to_clean)} models")

    async def health_check_models(self) -> Dict[str, Any]:
        """Perform health check on registered models."""
        from .embedding_manager import EnhancedEmbeddingManager
        
        health_status = {
            "healthy_models": [],
            "unhealthy_models": [],
            "total_models": len(self.models),
            "check_timestamp": datetime.utcnow().isoformat()
        }
        
        for model_id, model in self.models.items():
            try:
                # Create appropriate manager for health check
                if model.provider == EmbeddingProvider.HUGGINGFACE:
                    manager = EnhancedEmbeddingManager.create_huggingface_manager(model.model_name)
                elif model.provider == EmbeddingProvider.OLLAMA:
                    manager = EnhancedEmbeddingManager.create_ollama_manager(model_name=model.model_name)
                elif model.provider == EmbeddingProvider.OPENAI:
                    # Skip OpenAI health check if no API key available
                    health_status["healthy_models"].append({
                        "model_id": model_id,
                        "status": "skipped",
                        "reason": "api_key_required"
                    })
                    continue
                
                # Perform health check
                health_result = await manager.health_check()
                
                if health_result["status"] == "healthy":
                    health_status["healthy_models"].append({
                        "model_id": model_id,
                        "model_name": model.model_name,
                        "provider": model.provider.value,
                        "status": "healthy"
                    })
                else:
                    health_status["unhealthy_models"].append({
                        "model_id": model_id,
                        "model_name": model.model_name,
                        "provider": model.provider.value,
                        "status": "unhealthy",
                        "error": health_result.get("error", "Unknown error")
                    })
                    
            except Exception as e:
                health_status["unhealthy_models"].append({
                    "model_id": model_id,
                    "model_name": model.model_name,
                    "provider": model.provider.value,
                    "status": "error",
                    "error": str(e)
                })
        
        return health_status

# Global registry instance
_registry_instance = None

def get_embedding_model_registry() -> EmbeddingModelRegistry:
    """Get the global embedding model registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = EmbeddingModelRegistry()
    return _registry_instance