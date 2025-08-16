"""
LLM Services Module

Provides model discovery and registration services for dynamic LLM management.
"""

from .model_discovery_service import ModelDiscoveryService, get_discovery_service, DiscoveredModel
from .model_registration_service import ModelRegistrationService, get_registration_service

__all__ = [
    'ModelDiscoveryService',
    'get_discovery_service', 
    'DiscoveredModel',
    'ModelRegistrationService',
    'get_registration_service'
]