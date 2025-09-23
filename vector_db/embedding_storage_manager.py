# vector_db/model_storage_manager.py

"""
Model Storage Manager for persistent embedding model storage.

This module provides functionality to:
- Store embedding models persistently in /runtime/models/
- Manage model downloads and caching
- Validate model integrity and compatibility
- Integrate with the embedding model registry
"""

import os
import json
import shutil
import hashlib
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import tempfile

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}. Model storage functionality will be limited.")
    TORCH_AVAILABLE = False

from .embedding_model_registry import EmbeddingProvider, ModelMetadata

logger = logging.getLogger(__name__)

@dataclass
class StoredModelInfo:
    """Information about a stored model."""
    model_id: str
    model_name: str
    provider: str
    storage_path: str
    downloaded_at: datetime
    file_size_mb: float
    checksum: str
    model_type: str  # "sentence_transformer", "transformer", "onnx"
    status: str  # "available", "downloading", "error", "corrupted"
    last_validated: Optional[datetime] = None
    validation_error: Optional[str] = None

class ModelStorageManager:
    """Manages persistent storage of embedding models."""

    def __init__(
        self,
        storage_base_path: str = "runtime/models",
        enable_auto_download: bool = True,
        max_concurrent_downloads: int = 2,
        storage_limit_gb: Optional[float] = None
    ):
        """
        Initialize model storage manager.

        Args:
            storage_base_path: Base path for model storage
            enable_auto_download: Whether to auto-download missing models
            max_concurrent_downloads: Maximum concurrent downloads
            storage_limit_gb: Storage limit in GB (None for unlimited)
        """
        self.storage_base_path = Path(storage_base_path)
        self.enable_auto_download = enable_auto_download
        self.max_concurrent_downloads = max_concurrent_downloads
        self.storage_limit_gb = storage_limit_gb

        # Storage paths
        self.models_dir = self.storage_base_path  # Models stored directly in base path
        self.metadata_file = self.storage_base_path / "model_storage.json"
        self.temp_dir = self.storage_base_path / "temp"

        # In-memory storage tracking
        self.stored_models: Dict[str, StoredModelInfo] = {}
        self._download_semaphore = asyncio.Semaphore(max_concurrent_downloads)
        self._active_downloads: Dict[str, asyncio.Task] = {}

        # Initialize storage
        self._initialize_storage()
        self._load_metadata()
        self._sync_with_registry()

    def _initialize_storage(self):
        """Initialize storage directories."""
        try:
            self.storage_base_path.mkdir(parents=True, exist_ok=True)
            self.models_dir.mkdir(exist_ok=True)
            self.temp_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to initialize storage directories: {e}")
            raise

    def _load_metadata(self):
        """Load stored model metadata."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)

                for model_data in data.get("stored_models", []):
                    # Convert datetime strings back to datetime objects
                    model_data["downloaded_at"] = datetime.fromisoformat(model_data["downloaded_at"])
                    if model_data.get("last_validated"):
                        model_data["last_validated"] = datetime.fromisoformat(model_data["last_validated"])

                    model_info = StoredModelInfo(**model_data)
                    self.stored_models[model_info.model_id] = model_info

            else:
                logger.debug("No existing model metadata found")

        except Exception as e:
            logger.error(f"Failed to load model metadata: {e}")

    def _sync_with_registry(self):
        """Sync stored models with the current model registry."""
        try:
            from .embedding_model_registry import get_embedding_model_registry
            registry = get_embedding_model_registry()
            active_model_ids = {model.model_id for model in registry.list_models()}

            # Find models in storage that are no longer in registry
            stored_model_ids = set(self.stored_models.keys())
            stale_model_ids = stored_model_ids - active_model_ids

            if stale_model_ids:
                logger.debug(f"Found {len(stale_model_ids)} stale model entries in storage: {stale_model_ids}")

                # Remove stale entries
                for model_id in stale_model_ids:
                    logger.debug(f"Removing stale model entry: {model_id}")
                    del self.stored_models[model_id]

                # Save updated metadata
                self._save_metadata()
                logger.debug(f"Cleaned up {len(stale_model_ids)} stale model entries")

        except Exception as e:
            logger.warning(f"Failed to sync with registry: {e}")

    def _save_metadata(self):
        """Save stored model metadata."""
        try:
            data = {"stored_models": []}

            for model_info in self.stored_models.values():
                model_dict = asdict(model_info)
                model_dict["downloaded_at"] = model_info.downloaded_at.isoformat()
                if model_info.last_validated:
                    model_dict["last_validated"] = model_info.last_validated.isoformat()
                data["stored_models"].append(model_dict)

            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.metadata_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)

            temp_file.rename(self.metadata_file)
            logger.debug("Model metadata saved")

        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _get_directory_size(self, path: Path) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to calculate directory size for {path}: {e}")
            return 0.0

    def _get_total_storage_usage(self) -> float:
        """Get total storage usage in GB."""
        total_mb = self._get_directory_size(self.models_dir)
        return total_mb / 1024  # Convert to GB

    def _check_storage_limit(self, estimated_size_mb: float) -> bool:
        """Check if adding a model would exceed storage limit."""
        if not self.storage_limit_gb:
            return True

        current_usage_gb = self._get_total_storage_usage()
        estimated_total_gb = current_usage_gb + (estimated_size_mb / 1024)

        if estimated_total_gb > self.storage_limit_gb:
            logger.warning(
                f"Storage limit check failed: {estimated_total_gb:.2f}GB would exceed "
                f"limit of {self.storage_limit_gb}GB"
            )
            return False

        return True

    def get_model_storage_path(self, model_id: str) -> Path:
        """Get storage path for a model."""
        return self.models_dir / model_id

    def is_model_stored(self, model_id: str) -> bool:
        """Check if a model is stored locally."""
        if model_id not in self.stored_models:
            return False

        model_info = self.stored_models[model_id]
        storage_path = Path(model_info.storage_path)

        # Check if the storage path exists and contains model files
        if not storage_path.exists():
            logger.warning(f"Storage path missing for model {model_id}: {storage_path}")
            model_info.status = "error"
            self._save_metadata()
            return False

        return model_info.status == "available"

    def get_stored_model_info(self, model_id: str) -> Optional[StoredModelInfo]:
        """Get information about a stored model."""
        return self.stored_models.get(model_id)

    async def download_model(
        self,
        model_metadata: ModelMetadata,
        force_redownload: bool = False
    ) -> bool:
        """
        Download and store a model.

        Args:
            model_metadata: Model metadata from registry
            force_redownload: Force redownload even if model exists

        Returns:
            True if successful, False otherwise
        """
        model_id = model_metadata.model_id

        # Check if already downloading
        if model_id in self._active_downloads:
            logger.debug(f"Model {model_id} is already being downloaded")
            try:
                return await self._active_downloads[model_id]
            except Exception as e:
                logger.error(f"Download task for {model_id} failed: {e}")
                return False

        # Check if already stored and not forcing redownload
        if not force_redownload and self.is_model_stored(model_id):
            logger.debug(f"Model {model_id} is already stored")
            return True

        # Create download task
        download_task = asyncio.create_task(
            self._download_model_task(model_metadata, force_redownload)
        )
        self._active_downloads[model_id] = download_task

        try:
            result = await download_task
            return result
        finally:
            self._active_downloads.pop(model_id, None)

    async def _download_model_task(
        self,
        model_metadata: ModelMetadata,
        force_redownload: bool
    ) -> bool:
        """Internal task for downloading a model."""
        model_id = model_metadata.model_id
        model_name = model_metadata.model_name

        async with self._download_semaphore:
            logger.info(f"Downloading model {model_id} ({model_name})")

            try:
                # Check storage limit
                estimated_size_mb = model_metadata.model_size_mb or 500  # Default estimate
                if not self._check_storage_limit(estimated_size_mb):
                    logger.error(f"Storage limit exceeded for model {model_id}")
                    return False

                # Update status to downloading
                if model_id in self.stored_models:
                    self.stored_models[model_id].status = "downloading"
                else:
                    storage_path = self.get_model_storage_path(model_id)
                    self.stored_models[model_id] = StoredModelInfo(
                        model_id=model_id,
                        model_name=model_name,
                        provider=model_metadata.provider.value,
                        storage_path=str(storage_path),
                        downloaded_at=datetime.utcnow(),
                        file_size_mb=0.0,
                        checksum="",
                        model_type="",
                        status="downloading"
                    )

                self._save_metadata()

                # Download based on provider
                if model_metadata.provider == EmbeddingProvider.HUGGINGFACE:
                    success = await self._download_huggingface_model(model_metadata)
                else:
                    logger.error(f"Unsupported provider for download: {model_metadata.provider}")
                    success = False

                # Update status based on result
                stored_model = self.stored_models[model_id]
                if success:
                    stored_model.status = "available"
                    stored_model.downloaded_at = datetime.utcnow()
                    stored_model.last_validated = datetime.utcnow()
                    logger.info(f"Downloaded model {model_id} successfully")
                else:
                    stored_model.status = "error"
                    logger.error(f"Failed to download model {model_id}")

                self._save_metadata()
                return success

            except Exception as e:
                logger.error(f"Error downloading model {model_id}: {e}")
                if model_id in self.stored_models:
                    self.stored_models[model_id].status = "error"
                    self.stored_models[model_id].validation_error = str(e)
                    self._save_metadata()
                return False

    async def _download_huggingface_model(self, model_metadata: ModelMetadata) -> bool:
        """Download a HuggingFace model."""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for HuggingFace model download")
            return False

        model_id = model_metadata.model_id
        model_name = model_metadata.model_name
        storage_path = self.get_model_storage_path(model_id)

        try:
            # Create temporary directory for download
            with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_dir:
                temp_path = Path(temp_dir)

                # Get download kwargs from model metadata
                download_kwargs = model_metadata.download_kwargs or {}

                # Run download in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    success = await loop.run_in_executor(
                        executor,
                        self._download_hf_model_sync,
                        model_name, temp_path, download_kwargs
                    )

                if not success:
                    return False

                # Move from temp to final location
                if storage_path.exists():
                    shutil.rmtree(storage_path)

                storage_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temp_path), str(storage_path))

                # Validate downloaded model structure
                validation_result = self._validate_downloaded_model(storage_path, model_metadata)
                if not validation_result["valid"]:
                    logger.error(f"Model validation failed for {model_id}: {validation_result['error']}")
                    # Clean up invalid model
                    if storage_path.exists():
                        shutil.rmtree(storage_path)
                    return False

                # Calculate model information
                file_size_mb = self._get_directory_size(storage_path)

                # Update stored model info
                stored_model = self.stored_models[model_id]
                stored_model.file_size_mb = file_size_mb
                stored_model.storage_path = str(storage_path)

                # Determine model type
                if (storage_path / "config_sentence_transformers.json").exists():
                    stored_model.model_type = "sentence_transformer"
                elif (storage_path / "config.json").exists():
                    stored_model.model_type = "transformer"
                else:
                    stored_model.model_type = "unknown"

                # Calculate checksum of main model file
                config_file = storage_path / "config.json"
                if config_file.exists():
                    stored_model.checksum = self._calculate_file_checksum(config_file)

                logger.debug(f"HuggingFace model {model_id} downloaded to {storage_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to download HuggingFace model {model_id}: {e}")
            return False

    def _download_hf_model_sync(self, model_name: str, download_path: Path, download_kwargs: Optional[Dict] = None) -> bool:
        """Synchronous HuggingFace model download with configurable parameters."""
        try:
            logger.debug(f"Downloading HuggingFace model {model_name}")

            # Prepare download kwargs
            kwargs = download_kwargs or {}
            logger.debug(f"Using download kwargs: {kwargs}")

            # Try sentence-transformers first
            try:
                model = SentenceTransformer(
                    model_name,
                    cache_folder=str(download_path),
                    **kwargs
                )
                logger.debug(f"Downloaded as sentence-transformer: {model_name}")
                return True
            except Exception as st_error:
                logger.debug(f"Sentence-transformer download failed: {st_error}")

            # Fallback to transformers
            try:
                # Download tokenizer and model separately with kwargs
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(download_path),
                    **kwargs
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=str(download_path),
                    **kwargs
                )
                logger.debug(f"Downloaded as transformer: {model_name}")
                return True
            except Exception as t_error:
                logger.error(f"Transformer download failed: {t_error}")
                return False

        except Exception as e:
            logger.error(f"HuggingFace model download failed: {e}")
            return False

    def _validate_downloaded_model(self, storage_path: Path, model_metadata) -> Dict[str, Any]:
        """Validate a downloaded model structure."""
        try:
            logger.debug(f"Validating downloaded model at {storage_path}")

            # Check if directory exists and is not empty
            if not storage_path.exists():
                return {"valid": False, "error": "Storage path does not exist"}

            if not any(storage_path.iterdir()):
                return {"valid": False, "error": "Storage directory is empty"}

            # Check for essential files
            has_config = False
            model_type = "unknown"

            # Check for sentence-transformers structure
            st_config = storage_path / "config_sentence_transformers.json"
            if st_config.exists():
                has_config = True
                model_type = "sentence_transformer"
                logger.debug(f"Found sentence-transformers config: {st_config}")

            # Check for standard transformers structure
            hf_config = storage_path / "config.json"
            if hf_config.exists():
                has_config = True
                if model_type == "unknown":
                    model_type = "transformer"
                logger.debug(f"Found transformers config: {hf_config}")

                # Validate config.json content
                try:
                    with open(hf_config, 'r') as f:
                        config_data = json.load(f)

                    # Check for required fields
                    if not config_data.get("model_type") and not config_data.get("architectures"):
                        logger.warning(f"Config.json missing model_type and architectures for {storage_path}")
                        # This is not necessarily fatal for sentence-transformers models

                except Exception as e:
                    logger.warning(f"Failed to parse config.json: {e}")

            if not has_config:
                return {"valid": False, "error": "No valid config file found (config.json or config_sentence_transformers.json)"}

            # Check for model files
            model_files = [
                "pytorch_model.bin",
                "model.safetensors",
                "model.onnx",
                "tf_model.h5"
            ]

            has_model_file = any((storage_path / f).exists() for f in model_files)
            if not has_model_file:
                # Check for split model files
                has_model_file = any(
                    f.name.startswith("pytorch_model-") and f.name.endswith(".bin")
                    for f in storage_path.iterdir()
                ) or any(
                    f.name.startswith("model-") and f.name.endswith(".safetensors")
                    for f in storage_path.iterdir()
                )

            if not has_model_file:
                return {"valid": False, "error": "No model weights file found"}

            # For sentence-transformers, check for additional required files
            if model_type == "sentence_transformer":
                required_files = ["modules.json"]
                missing_files = [f for f in required_files if not (storage_path / f).exists()]
                if missing_files:
                    logger.warning(f"Missing sentence-transformers files: {missing_files}")
                    # This might not be fatal, continue validation

            logger.debug(f"Model validation successful: type={model_type}, path={storage_path}")
            return {
                "valid": True,
                "model_type": model_type,
                "config_files": [f for f in ["config.json", "config_sentence_transformers.json"]
                               if (storage_path / f).exists()]
            }

        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return {"valid": False, "error": f"Validation exception: {str(e)}"}

    async def validate_model(self, model_id: str) -> bool:
        """Validate a stored model."""
        if model_id not in self.stored_models:
            return False

        stored_model = self.stored_models[model_id]
        storage_path = Path(stored_model.storage_path)

        try:
            # Check if path exists
            if not storage_path.exists():
                logger.warning(f"Model path does not exist: {storage_path}")
                stored_model.status = "error"
                stored_model.validation_error = "Storage path not found"
                self._save_metadata()
                return False

            # Check if we can load the model
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                validation_result = await loop.run_in_executor(
                    executor,
                    self._validate_model_sync,
                    stored_model
                )

            if validation_result:
                stored_model.status = "available"
                stored_model.last_validated = datetime.utcnow()
                stored_model.validation_error = None
                logger.debug(f"Model {model_id} validation successful")
            else:
                stored_model.status = "corrupted"
                logger.warning(f"Model {model_id} validation failed")

            self._save_metadata()
            return validation_result

        except Exception as e:
            logger.error(f"Error validating model {model_id}: {e}")
            stored_model.status = "error"
            stored_model.validation_error = str(e)
            self._save_metadata()
            return False

    def _validate_model_sync(self, stored_model: StoredModelInfo) -> bool:
        """Synchronous model validation."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for model validation")
            return True  # Assume valid if we can't validate

        try:
            storage_path = Path(stored_model.storage_path)

            if stored_model.model_type == "sentence_transformer":
                # Validate sentence-transformer
                model = SentenceTransformer(str(storage_path))
                # Try to encode a test sentence
                test_embedding = model.encode("test sentence")
                return test_embedding is not None and len(test_embedding) > 0

            elif stored_model.model_type == "transformer":
                # Validate transformer
                tokenizer = AutoTokenizer.from_pretrained(str(storage_path))
                model = AutoModel.from_pretrained(str(storage_path))

                # Try to tokenize and encode
                inputs = tokenizer("test sentence", return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)

                return outputs.last_hidden_state is not None

            else:
                logger.warning(f"Unknown model type for validation: {stored_model.model_type}")
                return True  # Assume valid for unknown types

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def delete_model(self, model_id: str) -> bool:
        """Delete a stored model."""
        try:
            if model_id not in self.stored_models:
                logger.warning(f"Model {model_id} not found in storage")
                return False

            stored_model = self.stored_models[model_id]
            storage_path = Path(stored_model.storage_path)

            # Remove files
            if storage_path.exists():
                shutil.rmtree(storage_path)
                logger.debug(f"Deleted model files for {model_id}")

            # Remove from metadata
            del self.stored_models[model_id]
            self._save_metadata()

            logger.info(f"Deleted model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    def cleanup_old_models(self, max_age_days: int = 90) -> List[str]:
        """Clean up models that haven't been used recently."""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        deleted_models = []

        for model_id, stored_model in list(self.stored_models.items()):
            last_used = stored_model.last_validated or stored_model.downloaded_at

            if last_used < cutoff_date:
                logger.debug(f"Cleaning up old model {model_id} (last used: {last_used})")
                if self.delete_model(model_id):
                    deleted_models.append(model_id)

        return deleted_models

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_models = len(self.stored_models)
        available_models = len([m for m in self.stored_models.values() if m.status == "available"])
        total_size_gb = self._get_total_storage_usage()

        return {
            "total_models": total_models,
            "available_models": available_models,
            "error_models": len([m for m in self.stored_models.values() if m.status == "error"]),
            "downloading_models": len([m for m in self.stored_models.values() if m.status == "downloading"]),
            "total_size_gb": round(total_size_gb, 2),
            "storage_limit_gb": self.storage_limit_gb,
            "storage_usage_percent": round((total_size_gb / self.storage_limit_gb * 100), 2) if self.storage_limit_gb else None,
            "models_by_provider": self._get_models_by_provider(),
            "models_by_type": self._get_models_by_type()
        }

    def _get_models_by_provider(self) -> Dict[str, int]:
        """Get model count by provider."""
        provider_counts = {}
        for model in self.stored_models.values():
            provider = model.provider
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        return provider_counts

    def _get_models_by_type(self) -> Dict[str, int]:
        """Get model count by type."""
        type_counts = {}
        for model in self.stored_models.values():
            model_type = model.model_type
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
        return type_counts

    def list_stored_models(self) -> List[StoredModelInfo]:
        """List all stored models."""
        return list(self.stored_models.values())

    async def cleanup(self):
        """Cleanup resources."""
        # Cancel any active downloads
        for task in self._active_downloads.values():
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._active_downloads:
            await asyncio.gather(*self._active_downloads.values(), return_exceptions=True)

        logger.debug("Model storage manager cleanup completed")

# Global instance
_storage_manager_instance = None

def get_model_storage_manager() -> ModelStorageManager:
    """Get the global model storage manager instance."""
    global _storage_manager_instance
    if _storage_manager_instance is None:
        _storage_manager_instance = ModelStorageManager()
    return _storage_manager_instance