"""
Model Registration Service

Provides functionality to manage registered models and integrate them with the ModelManager.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_

from database.models import RegisteredModel, ModelTest, ModelProviderEnum
from database.connection import SessionLocal
from ..model_manager import ModelManager
from ..base.models import ModelConfig, ModelInfo
from ..config.provider_registry import get_provider_registry

logger = logging.getLogger(__name__)

class ModelRegistrationService:
    """Service for managing registered models and integrating with ModelManager."""
    
    def __init__(self):
        self.model_manager: Optional[ModelManager] = None
        self.provider_registry = get_provider_registry()
        self._loaded_models: Dict[str, RegisteredModel] = {}
        self._model_configs: Dict[str, ModelConfig] = {}
    
    def get_model_manager(self) -> ModelManager:
        """Get or create the ModelManager instance."""
        if self.model_manager is None:
            self.model_manager = ModelManager()
        return self.model_manager
    
    async def load_all_registered_models(self, user_id: Optional[int] = None) -> int:
        """
        Load all registered models from database into ModelManager.
        
        Args:
            user_id: If provided, only load models accessible to this user
            
        Returns:
            Number of models loaded
        """
        loaded_count = 0
        
        with SessionLocal() as db:
            try:
                # Build query for active, non-deleted models
                query = db.query(RegisteredModel).filter(
                    and_(
                        RegisteredModel.is_active == True,
                        RegisteredModel.deleted_at.is_(None)
                    )
                )
                
                # If user_id provided, filter for user's models + public models
                if user_id is not None:
                    query = query.filter(
                        (RegisteredModel.user_id == user_id) |
                        (RegisteredModel.is_public == True)
                    )
                
                models = query.all()
                
                manager = self.get_model_manager()
                
                for model in models:
                    try:
                        model_id = f"registered_{model.id}"
                        
                        # Parse configuration
                        config_dict = model.get_config()
                        provider_config = model.get_provider_config()
                        
                        # Create ModelConfig
                        model_config = ModelConfig.from_dict(config_dict)
                        
                        # Register with ModelManager
                        manager.register_model(
                            model_id=model_id,
                            provider_name=model.provider.value,
                            config=model_config,
                            provider_kwargs=provider_config,
                            fallback_priority=model.fallback_priority
                        )
                        
                        # Store for tracking
                        self._loaded_models[model_id] = model
                        self._model_configs[model_id] = model_config
                        
                        loaded_count += 1
                        logger.debug(f"Loaded registered model: {model.name} (ID: {model_id})")
                        
                    except Exception as e:
                        logger.error(f"Failed to load model {model.name} (ID: {model.id}): {str(e)}")
                        continue
                
                logger.debug(f"Loaded {loaded_count} registered models into ModelManager")
                return loaded_count
                
            except Exception as e:
                logger.error(f"Error loading registered models: {str(e)}")
                return 0
    
    async def reload_model(self, model_id: int) -> bool:
        """
        Reload a specific registered model.
        
        Args:
            model_id: Database ID of the model to reload
            
        Returns:
            True if successful, False otherwise
        """
        with SessionLocal() as db:
            try:
                model = db.query(RegisteredModel).filter(
                    and_(
                        RegisteredModel.id == model_id,
                        RegisteredModel.deleted_at.is_(None)
                    )
                ).first()
                
                if not model:
                    logger.warning(f"Model {model_id} not found for reload")
                    return False
                
                manager = self.get_model_manager()
                registered_model_id = f"registered_{model_id}"
                
                # Unregister existing model if it exists
                if registered_model_id in self._loaded_models:
                    manager.unregister_model(registered_model_id)
                    del self._loaded_models[registered_model_id]
                    del self._model_configs[registered_model_id]
                
                # Re-register if model is active
                if model.is_active:
                    config_dict = model.get_config()
                    provider_config = model.get_provider_config()
                    
                    model_config = ModelConfig.from_dict(config_dict)
                    
                    manager.register_model(
                        model_id=registered_model_id,
                        provider_name=model.provider.value,
                        config=model_config,
                        provider_kwargs=provider_config,
                        fallback_priority=model.fallback_priority
                    )
                    
                    self._loaded_models[registered_model_id] = model
                    self._model_configs[registered_model_id] = model_config
                    
                    logger.debug(f"Reloaded model: {model.name} (ID: {registered_model_id})")
                
                return True
                
            except Exception as e:
                logger.error(f"Error reloading model {model_id}: {str(e)}")
                return False
    
    async def unload_model(self, model_id: int) -> bool:
        """
        Unload a registered model from ModelManager.
        
        Args:
            model_id: Database ID of the model to unload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            manager = self.get_model_manager()
            registered_model_id = f"registered_{model_id}"
            
            if registered_model_id in self._loaded_models:
                manager.unregister_model(registered_model_id)
                del self._loaded_models[registered_model_id]
                del self._model_configs[registered_model_id]
                
                logger.debug(f"Unloaded model: {registered_model_id}")
                return True
            else:
                logger.warning(f"Model {registered_model_id} not currently loaded")
                return False
                
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {str(e)}")
            return False
    
    def get_loaded_models(self) -> Dict[str, RegisteredModel]:
        """Get all currently loaded registered models."""
        return self._loaded_models.copy()
    
    def is_model_loaded(self, model_id: int) -> bool:
        """Check if a model is currently loaded."""
        registered_model_id = f"registered_{model_id}"
        return registered_model_id in self._loaded_models
    
    async def get_model_info_list(self, user_id: Optional[int] = None) -> List[ModelInfo]:
        """
        Get list of ModelInfo objects for all registered models.
        
        Args:
            user_id: If provided, only include models accessible to this user
            
        Returns:
            List of ModelInfo objects
        """
        model_infos = []
        
        with SessionLocal() as db:
            try:
                # Build query
                query = db.query(RegisteredModel).filter(
                    and_(
                        RegisteredModel.is_active == True,
                        RegisteredModel.deleted_at.is_(None)
                    )
                )
                
                if user_id is not None:
                    query = query.filter(
                        (RegisteredModel.user_id == user_id) |
                        (RegisteredModel.is_public == True)
                    )
                
                models = query.all()
                
                for model in models:
                    try:
                        model_info = ModelInfo(
                            model_id=f"registered_{model.id}",
                            model_name=model.model_name,
                            provider=model.provider.value,
                            display_name=model.display_name or model.name,
                            description=model.description,
                            context_window=model.context_window,
                            max_tokens=model.max_tokens,
                            supports_streaming=model.supports_streaming,
                            supports_embeddings=model.supports_embeddings,
                            capabilities=[
                                f"Usage: {model.usage_count} requests",
                                f"Success Rate: {model.success_rate:.1f}%",
                                f"Avg Response: {model.average_response_time:.2f}ms" 
                                    if model.average_response_time else "No response data"
                            ]
                        )
                        model_infos.append(model_info)
                        
                    except Exception as e:
                        logger.error(f"Error creating ModelInfo for {model.name}: {str(e)}")
                        continue
                
                return model_infos
                
            except Exception as e:
                logger.error(f"Error getting model info list: {str(e)}")
                return []
    
    async def update_model_usage_stats(
        self, 
        model_id: int, 
        tokens_used: int = 0, 
        response_time: float = None, 
        success: bool = True,
        estimated_cost: float = None
    ) -> bool:
        """
        Update usage statistics for a registered model.
        
        Args:
            model_id: Database ID of the model
            tokens_used: Number of tokens used in the request
            response_time: Response time in milliseconds
            success: Whether the request was successful
            estimated_cost: Estimated cost of the request
            
        Returns:
            True if successful, False otherwise
        """
        with SessionLocal() as db:
            try:
                model = db.query(RegisteredModel).filter(
                    RegisteredModel.id == model_id
                ).first()
                
                if not model:
                    logger.warning(f"Model {model_id} not found for stats update")
                    return False
                
                # Update usage statistics
                model.update_usage_stats(tokens_used, response_time, success)
                
                if estimated_cost is not None:
                    model.estimated_cost += estimated_cost
                
                db.commit()
                
                logger.debug(f"Updated usage stats for model {model.name}")
                return True
                
            except Exception as e:
                logger.error(f"Error updating usage stats for model {model_id}: {str(e)}")
                db.rollback()
                return False
    
    async def sync_with_database(self, user_id: Optional[int] = None) -> Dict[str, int]:
        """
        Synchronize loaded models with database state.
        
        Args:
            user_id: If provided, only sync models accessible to this user
            
        Returns:
            Dictionary with sync statistics
        """
        stats = {
            'loaded': 0,
            'unloaded': 0,
            'reloaded': 0,
            'errors': 0
        }
        
        try:
            with SessionLocal() as db:
                # Get current database state
                query = db.query(RegisteredModel).filter(
                    RegisteredModel.deleted_at.is_(None)
                )
                
                if user_id is not None:
                    query = query.filter(
                        (RegisteredModel.user_id == user_id) |
                        (RegisteredModel.is_public == True)
                    )
                
                db_models = {f"registered_{m.id}": m for m in query.all()}
                
                # Find models to unload (deleted or deactivated)
                for loaded_id in list(self._loaded_models.keys()):
                    if loaded_id not in db_models or not db_models[loaded_id].is_active:
                        model_id = int(loaded_id.replace("registered_", ""))
                        if await self.unload_model(model_id):
                            stats['unloaded'] += 1
                        else:
                            stats['errors'] += 1
                
                # Find models to load or reload
                for model_id, model in db_models.items():
                    if model.is_active:
                        if model_id in self._loaded_models:
                            # Check if model needs reloading (config changed)
                            current_config = self._model_configs[model_id].to_dict()
                            new_config = model.get_config()
                            
                            if current_config != new_config:
                                db_id = int(model_id.replace("registered_", ""))
                                if await self.reload_model(db_id):
                                    stats['reloaded'] += 1
                                else:
                                    stats['errors'] += 1
                        else:
                            # Load new model
                            try:
                                config_dict = model.get_config()
                                provider_config = model.get_provider_config()
                                
                                model_config = ModelConfig.from_dict(config_dict)
                                
                                manager = self.get_model_manager()
                                manager.register_model(
                                    model_id=model_id,
                                    provider_name=model.provider.value,
                                    config=model_config,
                                    provider_kwargs=provider_config,
                                    fallback_priority=model.fallback_priority
                                )
                                
                                self._loaded_models[model_id] = model
                                self._model_configs[model_id] = model_config
                                
                                stats['loaded'] += 1
                                
                            except Exception as e:
                                logger.error(f"Error loading model {model.name}: {str(e)}")
                                stats['errors'] += 1
            
            logger.debug(f"Model sync completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during model sync: {str(e)}")
            stats['errors'] += 1
            return stats
    
    async def cleanup(self):
        """Clean up resources."""
        if self.model_manager:
            await self.model_manager.cleanup()
        
        self._loaded_models.clear()
        self._model_configs.clear()
        
        logger.debug("Model registration service cleaned up")

# Global instance
_registration_service = None

def get_registration_service() -> ModelRegistrationService:
    """Get global model registration service instance."""
    global _registration_service
    if _registration_service is None:
        _registration_service = ModelRegistrationService()
    return _registration_service