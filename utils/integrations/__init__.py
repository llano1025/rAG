"""
Integrations Module for RAG System
Provides integration with external systems and services.
"""

from .external_sources import ExternalSourcesManager, SyncConfiguration, SyncResult, external_sources_manager

__all__ = [
    'ExternalSourcesManager',
    'SyncConfiguration', 
    'SyncResult',
    'external_sources_manager'
]