# vector_db/model_preloader.py

"""
Model Preloader for startup checks and automatic model downloading.

This module provides functionality to:
- Check registered models availability at startup
- Download missing models automatically
- Validate model integrity and compatibility
- Update model registry with availability status
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .embedding_model_registry import get_embedding_model_registry, EmbeddingModelRegistry, ModelMetadata
from .model_storage_manager import get_model_storage_manager, ModelStorageManager

logger = logging.getLogger(__name__)

class PreloadPolicy(Enum):
    """Model preload policies."""
    NONE = "none"  # No automatic downloading
    ESSENTIAL = "essential"  # Only download essential models
    ALL = "all"  # Download all registered models
    SELECTIVE = "selective"  # Download based on use case priority

@dataclass
class PreloadResult:
    """Result of model preloading operation."""
    total_models: int
    already_available: int
    successfully_downloaded: int
    failed_downloads: int
    skipped: int
    download_time_seconds: float
    errors: List[str]
    downloaded_models: List[str]
    failed_models: List[str]

class ModelPreloader:
    """Handles model preloading and startup checks."""

    def __init__(
        self,
        model_registry: Optional[EmbeddingModelRegistry] = None,
        storage_manager: Optional[ModelStorageManager] = None,
        preload_policy: PreloadPolicy = PreloadPolicy.ESSENTIAL,
        max_concurrent_downloads: int = 2,
        download_timeout_minutes: int = 30,
        essential_use_cases: Optional[List[str]] = None
    ):
        """
        Initialize model preloader.

        Args:
            model_registry: Model registry instance
            storage_manager: Storage manager instance
            preload_policy: Policy for automatic downloading
            max_concurrent_downloads: Maximum concurrent downloads
            download_timeout_minutes: Timeout for individual downloads
            essential_use_cases: Use cases considered essential
        """
        self.model_registry = model_registry or get_embedding_model_registry()
        self.storage_manager = storage_manager or get_model_storage_manager()
        self.preload_policy = preload_policy
        self.max_concurrent_downloads = max_concurrent_downloads
        self.download_timeout_minutes = download_timeout_minutes

        # Default essential use cases
        self.essential_use_cases = essential_use_cases or [
            "general", "semantic_search", "question_answering"
        ]

        # Preloading state
        self._preload_in_progress = False
        self._preload_results: Optional[PreloadResult] = None


    def get_models_to_preload(self) -> List[ModelMetadata]:
        """Get list of models to preload based on policy."""
        all_models = self.model_registry.list_models()

        if self.preload_policy == PreloadPolicy.NONE:
            return []

        elif self.preload_policy == PreloadPolicy.ALL:
            return all_models

        elif self.preload_policy == PreloadPolicy.ESSENTIAL:
            essential_models = []
            for model in all_models:
                # Check if model has any essential use case
                if any(use_case in self.essential_use_cases for use_case in model.use_cases):
                    essential_models.append(model)
            return essential_models

        elif self.preload_policy == PreloadPolicy.SELECTIVE:
            # Get recommended models for essential use cases
            recommended_models = set()
            for use_case in self.essential_use_cases:
                models = self.model_registry.get_recommended_models(use_case, max_models=1)
                recommended_models.update(models)
            return list(recommended_models)

        else:
            logger.warning(f"Unknown preload policy: {self.preload_policy}")
            return []

    async def check_model_availability(self) -> Dict[str, Dict[str, Any]]:
        """Check availability of all registered models."""
        all_models = self.model_registry.list_models()
        availability_status = {}

        for model in all_models:
            model_id = model.model_id

            # Check if stored locally
            is_stored = self.storage_manager.is_model_stored(model_id)
            stored_info = self.storage_manager.get_stored_model_info(model_id)

            # Get performance metrics
            metrics = self.model_registry.get_performance_metrics(model_id)

            availability_status[model_id] = {
                "model_name": model.model_name,
                "provider": model.provider.value,
                "is_stored": is_stored,
                "status": stored_info.status if stored_info else "not_downloaded",
                "storage_path": stored_info.storage_path if stored_info else None,
                "file_size_mb": stored_info.file_size_mb if stored_info else None,
                "last_validated": stored_info.last_validated if stored_info else None,
                "download_recommended": model_id in [m.model_id for m in self.get_models_to_preload()],
                "performance_metrics": {
                    "total_embeddings": metrics.total_embeddings_generated if metrics else 0,
                    "success_rate": metrics.success_rate if metrics else None,
                    "last_used": metrics.last_used if metrics else None
                }
            }
            
        return availability_status

    async def validate_stored_models(self) -> Dict[str, bool]:
        """Validate all stored models."""
        logger.info("Validating stored models...")

        stored_models = self.storage_manager.list_stored_models()
        validation_results = {}

        for stored_model in stored_models:
            model_id = stored_model.model_id

            # Skip if recently validated (within last 24 hours)
            if (stored_model.last_validated and
                stored_model.last_validated > datetime.utcnow() - timedelta(hours=24)):
                validation_results[model_id] = True
                continue

            try:
                logger.info(f"Validating model {model_id}")
                is_valid = await self.storage_manager.validate_model(model_id)
                validation_results[model_id] = is_valid

                if not is_valid:
                    logger.warning(f"Model {model_id} validation failed")

            except Exception as e:
                logger.error(f"Error validating model {model_id}: {e}")
                validation_results[model_id] = False

        valid_count = sum(validation_results.values())
        total_count = len(validation_results)
        logger.info(f"Model validation completed: {valid_count}/{total_count} models valid")

        return validation_results

    async def preload_models(
        self,
        force_redownload: bool = False,
        specific_models: Optional[List[str]] = None
    ) -> PreloadResult:
        """
        Preload models based on the configured policy.

        Args:
            force_redownload: Force redownload even if models exist
            specific_models: Specific model IDs to preload (overrides policy)

        Returns:
            PreloadResult with details of the operation
        """
        if self._preload_in_progress:
            logger.warning("Model preloading already in progress")
            return self._preload_results or PreloadResult(0, 0, 0, 0, 0, 0.0, [], [], [])

        self._preload_in_progress = True
        start_time = datetime.utcnow()

        try:
            # Determine which models to preload
            if specific_models:
                models_to_check = []
                for model_id in specific_models:
                    model = self.model_registry.get_model(model_id)
                    if model:
                        models_to_check.append(model)
                    else:
                        logger.warning(f"Model {model_id} not found in registry")
            else:
                models_to_check = self.get_models_to_preload()

            logger.info(f"Starting preload for {len(models_to_check)} models")

            # Initialize result tracking
            total_models = len(models_to_check)
            already_available = 0
            successfully_downloaded = 0
            failed_downloads = 0
            skipped = 0
            errors = []
            downloaded_models = []
            failed_models = []

            # Check which models need downloading
            models_to_download = []
            for model in models_to_check:
                if force_redownload or not self.storage_manager.is_model_stored(model.model_id):
                    models_to_download.append(model)
                else:
                    already_available += 1

            logger.info(f"Models needing download: {len(models_to_download)}")

            # Download models with concurrency control
            if models_to_download:
                semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
                download_tasks = []

                for model in models_to_download:
                    task = asyncio.create_task(
                        self._download_with_timeout(model, semaphore, force_redownload)
                    )
                    download_tasks.append((model.model_id, task))

                # Wait for all downloads to complete
                for model_id, task in download_tasks:
                    try:
                        success = await task
                        if success:
                            successfully_downloaded += 1
                            downloaded_models.append(model_id)
                            logger.info(f"Successfully preloaded model {model_id}")
                        else:
                            failed_downloads += 1
                            failed_models.append(model_id)
                            errors.append(f"Download failed for model {model_id}")
                            logger.error(f"Failed to preload model {model_id}")

                    except asyncio.TimeoutError:
                        failed_downloads += 1
                        failed_models.append(model_id)
                        errors.append(f"Download timeout for model {model_id}")
                        logger.error(f"Download timeout for model {model_id}")

                    except Exception as e:
                        failed_downloads += 1
                        failed_models.append(model_id)
                        error_msg = f"Download error for model {model_id}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)

            end_time = datetime.utcnow()
            download_time = (end_time - start_time).total_seconds()

            # Create result
            self._preload_results = PreloadResult(
                total_models=total_models,
                already_available=already_available,
                successfully_downloaded=successfully_downloaded,
                failed_downloads=failed_downloads,
                skipped=skipped,
                download_time_seconds=download_time,
                errors=errors,
                downloaded_models=downloaded_models,
                failed_models=failed_models
            )

            logger.info(
                f"Model preloading completed in {download_time:.1f}s: "
                f"{successfully_downloaded} downloaded, {already_available} already available, "
                f"{failed_downloads} failed"
            )

            return self._preload_results

        finally:
            self._preload_in_progress = False

    async def _download_with_timeout(
        self,
        model: ModelMetadata,
        semaphore: asyncio.Semaphore,
        force_redownload: bool
    ) -> bool:
        """Download a model with timeout and semaphore control."""
        async with semaphore:
            try:
                timeout_seconds = self.download_timeout_minutes * 60
                return await asyncio.wait_for(
                    self.storage_manager.download_model(model, force_redownload),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"Download timeout for model {model.model_id}")
                raise
            except Exception as e:
                logger.error(f"Download error for model {model.model_id}: {e}")
                return False

    async def cleanup_old_models(
        self,
        max_age_days: int = 90,
        keep_essential: bool = True
    ) -> List[str]:
        """
        Clean up old models that haven't been used recently.

        Args:
            max_age_days: Maximum age in days before cleanup
            keep_essential: Whether to keep essential models regardless of age

        Returns:
            List of deleted model IDs
        """
        logger.info(f"Starting cleanup of models older than {max_age_days} days")

        # Get essential model IDs if we should keep them
        essential_model_ids = set()
        if keep_essential:
            essential_models = self.get_models_to_preload()
            essential_model_ids = {model.model_id for model in essential_models}

        # Get stored models for cleanup consideration
        stored_models = self.storage_manager.list_stored_models()
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

        models_to_delete = []
        for stored_model in stored_models:
            # Skip essential models if configured to keep them
            if keep_essential and stored_model.model_id in essential_model_ids:
                continue

            # Check last validation or download date
            last_used = stored_model.last_validated or stored_model.downloaded_at
            if last_used < cutoff_date:
                models_to_delete.append(stored_model.model_id)

        # Delete old models
        deleted_models = []
        for model_id in models_to_delete:
            try:
                if self.storage_manager.delete_model(model_id):
                    deleted_models.append(model_id)
                    logger.info(f"Deleted old model {model_id}")
                else:
                    logger.warning(f"Failed to delete model {model_id}")
            except Exception as e:
                logger.error(f"Error deleting model {model_id}: {e}")

        logger.info(f"Cleanup completed: deleted {len(deleted_models)} models")
        return deleted_models

    async def get_startup_summary(self) -> Dict[str, Any]:
        """Get a summary of model status for startup logging."""
        availability = await self.check_model_availability()

        total_registered = len(availability)
        locally_available = len([m for m in availability.values() if m["is_stored"]])
        recommended_for_download = len([m for m in availability.values() if m["download_recommended"]])

        storage_stats = self.storage_manager.get_storage_statistics()

        return {
            "total_registered_models": total_registered,
            "locally_available": locally_available,
            "recommended_for_download": recommended_for_download,
            "preload_policy": self.preload_policy.value,
            "storage_usage_gb": storage_stats["total_size_gb"],
            "last_preload_result": self._preload_results.__dict__ if self._preload_results else None
        }

    def is_preload_in_progress(self) -> bool:
        """Check if preloading is currently in progress."""
        return self._preload_in_progress

    def get_last_preload_result(self) -> Optional[PreloadResult]:
        """Get the result of the last preload operation."""
        return self._preload_results

# Global instance
_preloader_instance = None

def get_model_preloader() -> ModelPreloader:
    """Get the global model preloader instance."""
    global _preloader_instance
    if _preloader_instance is None:
        _preloader_instance = ModelPreloader()
    return _preloader_instance