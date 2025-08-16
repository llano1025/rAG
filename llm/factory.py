# llm/factory.py

"""
LLM factory for dynamic model manager setup using registered models.

This module provides factory functions for creating model managers that
load models from the database registration system.
"""

import logging
import asyncio
from typing import Optional, Dict, Any

from .model_manager import ModelManager
from .services.model_registration_service import get_registration_service

logger = logging.getLogger(__name__)

async def create_model_manager_with_registered_models(user_id: Optional[int] = None) -> ModelManager:
    """
    Create a ModelManager with models loaded from the registration system.
    
    This factory function loads models from the database that have been
    registered by users through the dynamic registration system.
    
    Args:
        user_id: Optional user ID to load user-specific models.
                If None, loads all public models.
    
    Returns:
        ModelManager instance with registered models loaded
    """
    manager = ModelManager()
    
    try:
        # Get the registration service
        registration_service = get_registration_service()
        
        # Load registered models from database
        if user_id:
            loaded_count = await registration_service.load_all_registered_models(user_id)
            logger.info(f"Loaded {loaded_count} registered models for user {user_id}")
        else:
            # Load all public models (user_id=None loads public models)
            loaded_count = await registration_service.load_all_registered_models(None)
            logger.info(f"Loaded {loaded_count} public registered models")
            
    except Exception as e:
        logger.error(f"Failed to load registered models: {str(e)}")
        # Don't fall back to static configs - force users to register models
        logger.warning("No models loaded. Users must register models through the admin interface.")
    
    return manager

async def load_models_for_user(manager: ModelManager, user_id: int) -> int:
    """
    Load registered models for a specific user into an existing ModelManager.
    
    Args:
        manager: Existing ModelManager instance
        user_id: User ID to load models for
        
    Returns:
        Number of models loaded
    """
    try:
        registration_service = get_registration_service()
        loaded_count = await registration_service.load_all_registered_models(user_id)
        logger.info(f"Loaded {loaded_count} models for user {user_id}")
        return loaded_count
    except Exception as e:
        logger.error(f"Failed to load models for user {user_id}: {str(e)}")
        return 0

def get_available_models_count() -> int:
    """
    Get the count of available registered models.
    
    Returns:
        Number of registered models available
    """
    try:
        registration_service = get_registration_service()
        return registration_service.get_registered_models_count()
    except Exception as e:
        logger.error(f"Failed to get models count: {str(e)}")
        return 0

# Keep backward compatibility but update implementation
def create_model_manager_with_registered_models_sync(user_id: Optional[int] = None) -> ModelManager:
    """
    Synchronous wrapper for create_model_manager_with_registered_models.
    
    This function runs the async version in a new event loop.
    Use this for synchronous contexts like class initialization.
    
    Args:
        user_id: Optional user ID to load user-specific models.
        
    Returns:
        ModelManager instance with registered models loaded
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we can't use run_until_complete
            # Create a new thread to run the async function
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(create_model_manager_with_registered_models(user_id))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        else:
            # No running loop, safe to use run_until_complete
            return loop.run_until_complete(create_model_manager_with_registered_models(user_id))
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(create_model_manager_with_registered_models(user_id))

def create_model_manager_with_defaults() -> ModelManager:
    """
    Create a ModelManager with default registered models.
    
    Note: This now loads from registered models instead of static configs.
    For new code, prefer create_model_manager_with_registered_models().
    
    Returns:
        ModelManager instance with registered models
    """
    logger.warning("create_model_manager_with_defaults() is deprecated. "
                  "Use create_model_manager_with_registered_models_sync() instead.")
    return create_model_manager_with_registered_models_sync()