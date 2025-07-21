"""
Plugin System for RAG Application
Provides extensible architecture for custom functionality.
"""

import os
import sys
import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from pathlib import Path
import yaml
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins supported by the system"""
    DOCUMENT_PROCESSOR = "document_processor"
    SEARCH_ENHANCER = "search_enhancer"
    LLM_PROVIDER = "llm_provider"
    VECTOR_STORE = "vector_store"
    AUTHENTICATION = "authentication"
    NOTIFICATION = "notification"
    DATA_SOURCE = "data_source"
    ANALYZER = "analyzer"
    TRANSFORMER = "transformer"
    WEBHOOK = "webhook"


class PluginStatus(Enum):
    """Plugin status states"""
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Plugin metadata information"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict = field(default_factory=dict)
    api_version: str = "1.0"
    min_system_version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PluginInstance:
    """Represents a loaded plugin instance"""
    metadata: PluginMetadata
    plugin_class: Type
    instance: Any = None
    status: PluginStatus = PluginStatus.LOADED
    config: Dict = field(default_factory=dict)
    error_message: str = None
    load_time: datetime = field(default_factory=datetime.utcnow)


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._hooks = {}
        
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup plugin resources"""
        pass
    
    def get_config_schema(self) -> Dict:
        """Return configuration schema for this plugin"""
        return {}
    
    def validate_config(self, config: Dict) -> bool:
        """Validate plugin configuration"""
        return True
    
    def register_hook(self, event: str, callback: Callable):
        """Register a hook for an event"""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
    
    async def emit_hook(self, event: str, *args, **kwargs):
        """Emit a hook event"""
        if event in self._hooks:
            for callback in self._hooks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f"Error in hook {event}: {e}")


class DocumentProcessorPlugin(BasePlugin):
    """Base class for document processing plugins"""
    
    @abstractmethod
    async def process_document(self, file_path: str, content: str, metadata: Dict) -> Dict:
        """Process a document and return enhanced metadata"""
        pass
    
    @abstractmethod
    def supported_file_types(self) -> List[str]:
        """Return list of supported file extensions"""
        pass


class SearchEnhancerPlugin(BasePlugin):
    """Base class for search enhancement plugins"""
    
    @abstractmethod
    async def enhance_query(self, query: str, context: Dict) -> str:
        """Enhance search query"""
        pass
    
    @abstractmethod
    async def post_process_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Post-process search results"""
        pass


class LLMProviderPlugin(BasePlugin):
    """Base class for LLM provider plugins"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, context: Dict) -> str:
        """Generate response using the LLM"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return information about the model"""
        pass


class DataSourcePlugin(BasePlugin):
    """Base class for external data source plugins"""
    
    @abstractmethod
    async def fetch_documents(self, source_config: Dict) -> List[Dict]:
        """Fetch documents from external source"""
        pass
    
    @abstractmethod
    async def sync_documents(self, last_sync: datetime) -> List[Dict]:
        """Sync documents since last sync time"""
        pass


class PluginManager:
    """Manages loading, configuration, and execution of plugins"""
    
    def __init__(self, plugin_dir: str = "plugins", config_file: str = "plugins.yaml"):
        self.plugin_dir = Path(plugin_dir)
        self.config_file = config_file
        self.plugins: Dict[str, PluginInstance] = {}
        self.plugin_configs = {}
        self._event_handlers = {}
        self._plugin_dependencies = {}
        
        # Ensure plugin directory exists
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize the plugin manager"""
        await self._load_plugin_configs()
        await self._discover_plugins()
        await self._load_plugins()
        await self._resolve_dependencies()
        await self._initialize_plugins()
    
    async def _load_plugin_configs(self):
        """Load plugin configurations from file"""
        config_path = self.plugin_dir / self.config_file
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self.plugin_configs = yaml.safe_load(f) or {}
                logger.info(f"Loaded plugin configurations from {config_path}")
            except Exception as e:
                logger.error(f"Error loading plugin config: {e}")
                self.plugin_configs = {}
        else:
            # Create default config
            default_config = {
                'enabled_plugins': [],
                'plugin_settings': {},
                'global_settings': {
                    'auto_reload': False,
                    'max_load_time': 30,
                    'enable_sandbox': True
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            self.plugin_configs = default_config
    
    async def _discover_plugins(self):
        """Discover plugins in the plugin directory"""
        plugin_files = []
        
        # Search for Python files in plugin directory
        for item in self.plugin_dir.rglob("*.py"):
            if item.name != "__init__.py" and not item.name.startswith("_"):
                plugin_files.append(item)
        
        logger.info(f"Discovered {len(plugin_files)} potential plugin files")
        return plugin_files
    
    async def _load_plugins(self):
        """Load plugins from discovered files"""
        plugin_files = await self._discover_plugins()
        
        for plugin_file in plugin_files:
            try:
                await self._load_plugin_from_file(plugin_file)
            except Exception as e:
                logger.error(f"Error loading plugin from {plugin_file}: {e}")
    
    async def _load_plugin_from_file(self, plugin_file: Path):
        """Load a plugin from a Python file"""
        # Convert file path to module name
        relative_path = plugin_file.relative_to(self.plugin_dir)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")
        
        try:
            # Add plugin directory to Python path
            if str(self.plugin_dir) not in sys.path:
                sys.path.insert(0, str(self.plugin_dir))
            
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin and
                    not inspect.isabstract(obj)):
                    
                    await self._register_plugin_class(name, obj)
                    
        except Exception as e:
            logger.error(f"Error loading plugin module {module_name}: {e}")
    
    async def _register_plugin_class(self, name: str, plugin_class: Type[BasePlugin]):
        """Register a plugin class"""
        try:
            # Create temporary instance to get metadata
            temp_instance = plugin_class({})
            metadata = temp_instance.get_metadata()
            
            # Check if plugin is enabled
            enabled_plugins = self.plugin_configs.get('enabled_plugins', [])
            if enabled_plugins and metadata.name not in enabled_plugins:
                logger.info(f"Plugin {metadata.name} is not enabled, skipping")
                return
            
            # Get plugin configuration
            plugin_config = self.plugin_configs.get('plugin_settings', {}).get(metadata.name, {})
            
            # Create plugin instance
            plugin_instance = PluginInstance(
                metadata=metadata,
                plugin_class=plugin_class,
                config=plugin_config
            )
            
            self.plugins[metadata.name] = plugin_instance
            logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
            
        except Exception as e:
            logger.error(f"Error registering plugin class {name}: {e}")
    
    async def _resolve_dependencies(self):
        """Resolve plugin dependencies"""
        # Build dependency graph
        for plugin_name, plugin_instance in self.plugins.items():
            deps = plugin_instance.metadata.dependencies
            self._plugin_dependencies[plugin_name] = deps
            
            # Check if dependencies are available
            for dep in deps:
                if dep not in self.plugins:
                    logger.warning(f"Plugin {plugin_name} depends on missing plugin: {dep}")
                    plugin_instance.status = PluginStatus.ERROR
                    plugin_instance.error_message = f"Missing dependency: {dep}"
    
    async def _initialize_plugins(self):
        """Initialize all loaded plugins"""
        # Topological sort for dependency order
        initialization_order = self._get_initialization_order()
        
        for plugin_name in initialization_order:
            await self._initialize_plugin(plugin_name)
    
    def _get_initialization_order(self) -> List[str]:
        """Get plugin initialization order based on dependencies"""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(plugin_name):
            if plugin_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {plugin_name}")
            if plugin_name in visited:
                return
                
            temp_visited.add(plugin_name)
            
            # Visit dependencies first
            deps = self._plugin_dependencies.get(plugin_name, [])
            for dep in deps:
                if dep in self.plugins:
                    visit(dep)
            
            temp_visited.remove(plugin_name)
            visited.add(plugin_name)
            order.append(plugin_name)
        
        for plugin_name in self.plugins:
            if plugin_name not in visited:
                try:
                    visit(plugin_name)
                except ValueError as e:
                    logger.error(f"Dependency error: {e}")
                    # Skip this plugin
                    continue
        
        return order
    
    async def _initialize_plugin(self, plugin_name: str):
        """Initialize a specific plugin"""
        plugin_instance = self.plugins[plugin_name]
        
        if plugin_instance.status == PluginStatus.ERROR:
            logger.warning(f"Skipping initialization of plugin {plugin_name} due to errors")
            return
        
        try:
            # Create plugin instance
            plugin_instance.instance = plugin_instance.plugin_class(plugin_instance.config)
            
            # Validate configuration
            if not plugin_instance.instance.validate_config(plugin_instance.config):
                raise ValueError("Invalid plugin configuration")
            
            # Initialize plugin
            success = await plugin_instance.instance.initialize()
            if success:
                plugin_instance.status = PluginStatus.ACTIVE
                logger.info(f"Successfully initialized plugin: {plugin_name}")
            else:
                plugin_instance.status = PluginStatus.INACTIVE
                logger.warning(f"Plugin initialization returned False: {plugin_name}")
                
        except Exception as e:
            plugin_instance.status = PluginStatus.ERROR
            plugin_instance.error_message = str(e)
            logger.error(f"Error initializing plugin {plugin_name}: {e}")
    
    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInstance]:
        """Get all active plugins of a specific type"""
        return [
            plugin for plugin in self.plugins.values()
            if plugin.metadata.plugin_type == plugin_type and plugin.status == PluginStatus.ACTIVE
        ]
    
    async def execute_plugin_method(self, plugin_name: str, method_name: str, *args, **kwargs):
        """Execute a method on a specific plugin"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} not found")
        
        plugin_instance = self.plugins[plugin_name]
        if plugin_instance.status != PluginStatus.ACTIVE:
            raise ValueError(f"Plugin {plugin_name} is not active")
        
        if not hasattr(plugin_instance.instance, method_name):
            raise ValueError(f"Plugin {plugin_name} does not have method {method_name}")
        
        method = getattr(plugin_instance.instance, method_name)
        
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        else:
            return method(*args, **kwargs)
    
    async def broadcast_event(self, event: str, *args, **kwargs):
        """Broadcast an event to all active plugins"""
        for plugin_instance in self.plugins.values():
            if plugin_instance.status == PluginStatus.ACTIVE and plugin_instance.instance:
                try:
                    await plugin_instance.instance.emit_hook(event, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Error broadcasting event {event} to plugin {plugin_instance.metadata.name}: {e}")
    
    async def reload_plugin(self, plugin_name: str):
        """Reload a specific plugin"""
        if plugin_name in self.plugins:
            # Cleanup existing plugin
            plugin_instance = self.plugins[plugin_name]
            if plugin_instance.instance:
                try:
                    await plugin_instance.instance.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up plugin {plugin_name}: {e}")
            
            # Remove from plugins
            del self.plugins[plugin_name]
        
        # Reload configurations
        await self._load_plugin_configs()
        
        # Rediscover and load plugins
        await self._discover_plugins()
        await self._load_plugins()
        
        # Reinitialize if plugin was reloaded
        if plugin_name in self.plugins:
            await self._initialize_plugin(plugin_name)
    
    async def get_plugin_status(self) -> Dict:
        """Get status of all plugins"""
        status = {
            'total_plugins': len(self.plugins),
            'active_plugins': len([p for p in self.plugins.values() if p.status == PluginStatus.ACTIVE]),
            'plugins': {}
        }
        
        for name, plugin in self.plugins.items():
            status['plugins'][name] = {
                'status': plugin.status.value,
                'version': plugin.metadata.version,
                'type': plugin.metadata.plugin_type.value,
                'error_message': plugin.error_message,
                'load_time': plugin.load_time.isoformat()
            }
        
        return status
    
    async def cleanup(self):
        """Cleanup all plugins"""
        for plugin_instance in self.plugins.values():
            if plugin_instance.instance:
                try:
                    await plugin_instance.instance.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up plugin {plugin_instance.metadata.name}: {e}")


# Global plugin manager instance
plugin_manager = PluginManager()


# Decorator for plugin registration
def plugin(plugin_type: PluginType):
    """Decorator for registering plugins"""
    def decorator(cls):
        cls._plugin_type = plugin_type
        return cls
    return decorator