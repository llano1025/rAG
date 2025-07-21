"""
Plugin System for RAG Application
Provides extensible architecture for custom functionality.
"""

from .plugin_system import (
    BasePlugin,
    DocumentProcessorPlugin,
    SearchEnhancerPlugin,
    LLMProviderPlugin,
    DataSourcePlugin,
    PluginManager,
    PluginType,
    PluginStatus,
    PluginMetadata,
    PluginInstance,
    plugin_manager,
    plugin
)

__all__ = [
    'BasePlugin',
    'DocumentProcessorPlugin', 
    'SearchEnhancerPlugin',
    'LLMProviderPlugin',
    'DataSourcePlugin',
    'PluginManager',
    'PluginType',
    'PluginStatus',
    'PluginMetadata',
    'PluginInstance',
    'plugin_manager',
    'plugin'
]