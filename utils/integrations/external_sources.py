"""
External Document Sources Integration Manager
Manages connections and synchronization with external document sources.
Note: Plugin system has been removed - this is a simplified version.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import aiofiles

from database.models import Document, User
from file_processor.text_extractor import TextExtractor
from vector_db.embedding_manager import EmbeddingManager
from utils.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Status of document synchronization"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class SyncConfiguration:
    """Configuration for document source synchronization"""
    source_name: str
    source_type: str  # "confluence", "google_drive", "sharepoint", etc.
    config: Dict
    sync_interval: int = 3600  # seconds
    enabled: bool = True
    auto_sync: bool = True
    last_sync: Optional[datetime] = None
    next_sync: Optional[datetime] = None
    sync_status: SyncStatus = SyncStatus.PENDING
    error_count: int = 0
    max_errors: int = 5
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SyncResult:
    """Result of a synchronization operation"""
    source_name: str
    start_time: datetime
    end_time: datetime
    documents_fetched: int = 0
    documents_processed: int = 0
    documents_added: int = 0
    documents_updated: int = 0
    documents_failed: int = 0
    errors: List[str] = field(default_factory=list)
    status: SyncStatus = SyncStatus.COMPLETED


class ExternalSourcesManager:
    """Manages external document sources integration"""
    
    def __init__(self, config_file: str = "config/external_sources.json"):
        self.config_file = Path(config_file)
        self.sync_configs: Dict[str, SyncConfiguration] = {}
        self.active_syncs: Dict[str, asyncio.Task] = {}
        self.sync_results: List[SyncResult] = []
        self.text_extractor = TextExtractor()
        self.embedding_manager = EmbeddingManager()
        self.metrics = MetricsCollector()
        self._scheduler_task = None
        self._stop_scheduler = False
    
    async def initialize(self):
        """Initialize the external sources manager"""
        try:
            await self._load_configurations()
            await self._start_scheduler()
            logger.info("External sources manager initialized")
        except Exception as e:
            logger.error(f"Error initializing external sources manager: {e}")
            raise
    
    async def _load_configurations(self):
        """Load sync configurations from file"""
        if self.config_file.exists():
            try:
                async with aiofiles.open(self.config_file, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                
                for source_data in data.get('sources', []):
                    config = SyncConfiguration(
                        source_name=source_data['source_name'],
                        plugin_name=source_data['plugin_name'],
                        config=source_data['config'],
                        sync_interval=source_data.get('sync_interval', 3600),
                        enabled=source_data.get('enabled', True),
                        auto_sync=source_data.get('auto_sync', True),
                        last_sync=datetime.fromisoformat(source_data['last_sync']) if source_data.get('last_sync') else None,
                        error_count=source_data.get('error_count', 0),
                        max_errors=source_data.get('max_errors', 5)
                    )
                    
                    # Calculate next sync time
                    if config.last_sync and config.auto_sync:
                        config.next_sync = config.last_sync + timedelta(seconds=config.sync_interval)
                    
                    self.sync_configs[config.source_name] = config
                
                logger.info(f"Loaded {len(self.sync_configs)} external source configurations")
                
            except Exception as e:
                logger.error(f"Error loading external source configurations: {e}")
        else:
            # Create default config file
            await self._save_configurations()
    
    async def _save_configurations(self):
        """Save sync configurations to file"""
        try:
            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'sources': []
            }
            
            for config in self.sync_configs.values():
                source_data = {
                    'source_name': config.source_name,
                    'plugin_name': config.plugin_name,
                    'config': config.config,
                    'sync_interval': config.sync_interval,
                    'enabled': config.enabled,
                    'auto_sync': config.auto_sync,
                    'last_sync': config.last_sync.isoformat() if config.last_sync else None,
                    'error_count': config.error_count,
                    'max_errors': config.max_errors
                }
                data['sources'].append(source_data)
            
            async with aiofiles.open(self.config_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
        except Exception as e:
            logger.error(f"Error saving external source configurations: {e}")
    
    async def add_source(self, source_name: str, source_type: str, config: Dict,
                        sync_interval: int = 3600, auto_sync: bool = True) -> bool:
        """Add a new external document source"""
        try:
            # Validate source type (simplified without plugin system)
            supported_types = ["confluence", "google_drive", "sharepoint", "notion", "custom"]

            if source_type not in supported_types:
                raise ValueError(f"Source type {source_type} not supported. Supported types: {supported_types}")

            # Create sync configuration
            sync_config = SyncConfiguration(
                source_name=source_name,
                source_type=source_type,
                config=config,
                sync_interval=sync_interval,
                auto_sync=auto_sync
            )

            # Basic configuration validation
            required_keys = ["url"] if source_type in ["confluence", "sharepoint"] else ["credentials"]
            for key in required_keys:
                if key not in config:
                    logger.warning(f"Missing recommended config key '{key}' for {source_type}")

            self.sync_configs[source_name] = sync_config
            await self._save_configurations()

            logger.info(f"Added external source: {source_name} (type: {source_type})")
            return True

        except Exception as e:
            logger.error(f"Error adding external source {source_name}: {e}")
            return False
    
    async def remove_source(self, source_name: str) -> bool:
        """Remove an external document source"""
        try:
            if source_name not in self.sync_configs:
                return False
            
            # Stop any active sync
            if source_name in self.active_syncs:
                self.active_syncs[source_name].cancel()
                del self.active_syncs[source_name]
            
            del self.sync_configs[source_name]
            await self._save_configurations()
            
            logger.info(f"Removed external source: {source_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing external source {source_name}: {e}")
            return False
    
    async def sync_source(self, source_name: str, force: bool = False) -> SyncResult:
        """Manually trigger synchronization for a specific source"""
        if source_name not in self.sync_configs:
            raise ValueError(f"Source {source_name} not found")
        
        config = self.sync_configs[source_name]
        
        if not config.enabled and not force:
            raise ValueError(f"Source {source_name} is disabled")
        
        if source_name in self.active_syncs and not self.active_syncs[source_name].done():
            raise ValueError(f"Sync already running for source {source_name}")
        
        # Start sync task
        task = asyncio.create_task(self._perform_sync(config))
        self.active_syncs[source_name] = task
        
        try:
            result = await task
            return result
        finally:
            if source_name in self.active_syncs:
                del self.active_syncs[source_name]
    
    async def _perform_sync(self, config: SyncConfiguration) -> SyncResult:
        """Perform synchronization for a source"""
        start_time = datetime.utcnow()
        result = SyncResult(
            source_name=config.source_name,
            start_time=start_time,
            end_time=start_time
        )
        
        try:
            config.sync_status = SyncStatus.RUNNING
            await self._save_configurations()
            
            # Simplified document fetching (plugin system removed)
            # This is now a placeholder - external source integration would need
            # to be implemented directly based on source_type

            logger.warning(f"External source sync for {config.source_type} is not implemented in simplified version")
            logger.info(f"Sync configuration: {config.source_name} (type: {config.source_type})")

            # Placeholder - in a real implementation, you would add specific
            # integration code for each source_type (confluence, google_drive, etc.)
            documents = []  # No documents fetched in simplified version

            result.documents_fetched = len(documents)
            
            # Process each document
            for doc_data in documents:
                try:
                    processed = await self._process_external_document(doc_data, config.source_name)
                    if processed:
                        result.documents_processed += 1
                        if processed == "added":
                            result.documents_added += 1
                        elif processed == "updated":
                            result.documents_updated += 1
                except Exception as e:
                    result.documents_failed += 1
                    result.errors.append(f"Error processing document {doc_data.get('source_id', 'unknown')}: {e}")
                    logger.error(f"Error processing document: {e}")
            
            # Update sync status
            config.last_sync = start_time
            config.next_sync = start_time + timedelta(seconds=config.sync_interval)
            config.sync_status = SyncStatus.COMPLETED
            config.error_count = 0  # Reset error count on successful sync
            
            result.status = SyncStatus.COMPLETED
            
        except Exception as e:
            config.sync_status = SyncStatus.FAILED
            config.error_count += 1
            result.status = SyncStatus.FAILED
            result.errors.append(str(e))
            logger.error(f"Sync failed for source {config.source_name}: {e}")
            
            # Disable source if too many errors
            if config.error_count >= config.max_errors:
                config.enabled = False
                logger.warning(f"Disabled source {config.source_name} due to repeated failures")
        
        finally:
            result.end_time = datetime.utcnow()
            await self._save_configurations()
            
            # Store result
            self.sync_results.append(result)
            
            # Keep only last 100 results
            if len(self.sync_results) > 100:
                self.sync_results = self.sync_results[-100:]
            
            # Update metrics
            await self._update_sync_metrics(result)
        
        return result
    
    async def _process_external_document(self, doc_data: Dict, source_name: str) -> Optional[str]:
        """Process a document from external source"""
        try:
            # Check if document already exists
            from database.connection import SessionLocal
            
            session = SessionLocal()
            try:
                existing_doc = await session.execute(
                    "SELECT id FROM documents WHERE source_id = ? AND source_type = ?",
                    (doc_data['source_id'], doc_data['source_type'])
                )
                existing = existing_doc.fetchone()
                
                if existing:
                    # Update existing document
                    await session.execute(
                        """UPDATE documents SET 
                           title = ?, content = ?, metadata = ?, updated_at = ?
                           WHERE source_id = ? AND source_type = ?""",
                        (
                            doc_data['title'],
                            doc_data['content'],
                            json.dumps(doc_data['metadata']),
                            datetime.utcnow(),
                            doc_data['source_id'],
                            doc_data['source_type']
                        )
                    )
                    await session.commit()
                    
                    # Update embeddings
                    await self.embedding_manager.update_document_embeddings(
                        existing[0], doc_data['content']
                    )
                    
                    return "updated"
                
                else:
                    # Create new document
                    doc_id = await session.execute(
                        """INSERT INTO documents 
                           (title, content, file_type, source_id, source_type, metadata, 
                            upload_date, status, owner_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', 1)
                           RETURNING id""",
                        (
                            doc_data['title'],
                            doc_data['content'],
                            doc_data['file_type'],
                            doc_data['source_id'],
                            doc_data['source_type'],
                            json.dumps(doc_data['metadata']),
                            datetime.utcnow()
                        )
                    )
                    new_doc_id = doc_id.fetchone()[0]
                    await session.commit()
                    
                    # Generate embeddings
                    await self.embedding_manager.add_document_embeddings(
                        new_doc_id, doc_data['content'], doc_data['metadata']
                    )
                    
                    return "added"
            finally:
                session.close()
        
        except Exception as e:
            logger.error(f"Error processing external document: {e}")
            raise
    
    async def _start_scheduler(self):
        """Start the automatic sync scheduler"""
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def _scheduler_loop(self):
        """Main scheduler loop for automatic synchronization"""
        while not self._stop_scheduler:
            try:
                current_time = datetime.utcnow()
                
                for source_name, config in self.sync_configs.items():
                    if (config.enabled and config.auto_sync and 
                        config.next_sync and current_time >= config.next_sync and
                        source_name not in self.active_syncs):
                        
                        # Start sync task
                        task = asyncio.create_task(self._perform_sync(config))
                        self.active_syncs[source_name] = task
                        
                        # Don't wait for completion, let it run in background
                        task.add_done_callback(lambda t, name=source_name: self.active_syncs.pop(name, None))
                
                # Check every minute
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_sync_metrics(self, result: SyncResult):
        """Update metrics for sync operation"""
        try:
            await self.metrics.record_metric(
                'external_sync_duration',
                (result.end_time - result.start_time).total_seconds(),
                tags={'source': result.source_name, 'status': result.status.value}
            )
            
            await self.metrics.record_metric(
                'external_sync_documents_fetched',
                result.documents_fetched,
                tags={'source': result.source_name}
            )
            
            await self.metrics.record_metric(
                'external_sync_documents_processed',
                result.documents_processed,
                tags={'source': result.source_name}
            )
            
            if result.status == SyncStatus.FAILED:
                await self.metrics.record_metric(
                    'external_sync_errors',
                    len(result.errors),
                    tags={'source': result.source_name}
                )
                
        except Exception as e:
            logger.error(f"Error updating sync metrics: {e}")
    
    async def get_sync_status(self) -> Dict:
        """Get status of all sync configurations"""
        status = {
            'total_sources': len(self.sync_configs),
            'enabled_sources': len([c for c in self.sync_configs.values() if c.enabled]),
            'active_syncs': len(self.active_syncs),
            'sources': {}
        }
        
        for name, config in self.sync_configs.items():
            status['sources'][name] = {
                'plugin_name': config.plugin_name,
                'enabled': config.enabled,
                'auto_sync': config.auto_sync,
                'sync_status': config.sync_status.value,
                'last_sync': config.last_sync.isoformat() if config.last_sync else None,
                'next_sync': config.next_sync.isoformat() if config.next_sync else None,
                'error_count': config.error_count,
                'is_running': name in self.active_syncs
            }
        
        return status
    
    async def get_sync_history(self, source_name: str = None, limit: int = 50) -> List[Dict]:
        """Get sync history"""
        results = self.sync_results
        
        if source_name:
            results = [r for r in results if r.source_name == source_name]
        
        results = sorted(results, key=lambda x: x.start_time, reverse=True)[:limit]
        
        return [
            {
                'source_name': r.source_name,
                'start_time': r.start_time.isoformat(),
                'end_time': r.end_time.isoformat(),
                'duration': (r.end_time - r.start_time).total_seconds(),
                'status': r.status.value,
                'documents_fetched': r.documents_fetched,
                'documents_processed': r.documents_processed,
                'documents_added': r.documents_added,
                'documents_updated': r.documents_updated,
                'documents_failed': r.documents_failed,
                'error_count': len(r.errors),
                'errors': r.errors[:5]  # Only show first 5 errors
            }
            for r in results
        ]
    
    async def cleanup(self):
        """Cleanup resources"""
        self._stop_scheduler = True
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active syncs
        for task in self.active_syncs.values():
            task.cancel()
        
        # Wait for cancellation
        if self.active_syncs:
            await asyncio.gather(*self.active_syncs.values(), return_exceptions=True)


# Global instance
external_sources_manager = ExternalSourcesManager()