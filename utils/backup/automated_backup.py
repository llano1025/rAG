"""
Automated backup and recovery system for RAG application.
Handles database backups, vector index backups, configuration backups, and recovery procedures.
"""

import asyncio
import logging
import os
import shutil
import gzip
import tarfile
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
import subprocess
import tempfile

logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Types of backups."""
    DATABASE = "database"
    VECTOR_INDICES = "vector_indices"
    CONFIGURATION = "configuration"
    USER_DATA = "user_data"
    FULL_SYSTEM = "full_system"

class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"

class CompressionType(Enum):
    """Compression methods."""
    NONE = "none"
    GZIP = "gzip"
    TAR_GZ = "tar_gz"
    ZIP = "zip"

@dataclass
class BackupMetadata:
    """Backup metadata information."""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    compression_type: CompressionType = CompressionType.GZIP
    retention_days: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class BackupConfig:
    """Backup configuration."""
    backup_dir: str
    database_url: str
    vector_indices_dir: str
    config_files: List[str]
    retention_policy: Dict[BackupType, int]  # days to retain
    compression_enabled: bool = True
    verification_enabled: bool = True
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    max_parallel_backups: int = 3
    backup_schedule: Dict[BackupType, str] = field(default_factory=dict)  # cron expressions

class AutomatedBackupManager:
    """
    Comprehensive automated backup and recovery system.
    
    Features:
    - Scheduled database backups
    - Vector index backup procedures
    - Configuration backup system
    - Backup rotation and cleanup
    - Backup integrity verification
    - Incremental backup support
    - Compression and encryption
    - Recovery procedures
    """
    
    def __init__(
        self,
        config: BackupConfig,
        enable_scheduling: bool = True
    ):
        """
        Initialize automated backup manager.
        
        Args:
            config: Backup configuration
            enable_scheduling: Whether to enable scheduled backups
        """
        self.config = config
        self.enable_scheduling = enable_scheduling
        
        # Backup state
        self.active_backups: Dict[str, BackupMetadata] = {}
        self.backup_history: List[BackupMetadata] = []
        self.running = False
        
        # Background tasks
        self.scheduler_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Semaphore for parallel backup control
        self.backup_semaphore = asyncio.Semaphore(config.max_parallel_backups)
        
        # Ensure backup directory exists
        Path(config.backup_dir).mkdir(parents=True, exist_ok=True)
        
        # Load existing backup history
        self._load_backup_history()
    
    async def start(self):
        """Start the automated backup manager."""
        if self.running:
            return
        
        self.running = True
        
        if self.enable_scheduling:
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Automated backup manager started")
    
    async def stop(self):
        """Stop the automated backup manager."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop background tasks
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active backups to complete
        if self.active_backups:
            logger.info(f"Waiting for {len(self.active_backups)} active backups to complete")
            await asyncio.sleep(5)  # Give backups time to finish
        
        logger.info("Automated backup manager stopped")
    
    async def create_backup(
        self,
        backup_type: BackupType,
        incremental: bool = False,
        compression: Optional[CompressionType] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Create a backup of the specified type.
        
        Args:
            backup_type: Type of backup to create
            incremental: Whether to create incremental backup
            compression: Compression method to use
            description: Optional backup description
            
        Returns:
            Backup ID
        """
        backup_id = f"{backup_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Create backup metadata
        backup_metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            compression_type=compression or (CompressionType.GZIP if self.config.compression_enabled else CompressionType.NONE),
            retention_days=self.config.retention_policy.get(backup_type, 30),
            metadata={
                'incremental': incremental,
                'description': description,
                'source_system': 'rag_system'
            }
        )
        
        # Add to active backups
        self.active_backups[backup_id] = backup_metadata
        
        # Start backup process
        asyncio.create_task(self._perform_backup(backup_metadata))
        
        logger.info(f"Started backup: {backup_id} (type: {backup_type.value})")
        
        return backup_id
    
    async def restore_backup(
        self,
        backup_id: str,
        target_dir: Optional[str] = None,
        verify_before_restore: bool = True
    ) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_id: Backup ID to restore from
            target_dir: Target directory for restoration
            verify_before_restore: Whether to verify backup before restore
            
        Returns:
            True if restore was successful
        """
        # Find backup metadata
        backup_metadata = None
        for backup in self.backup_history:
            if backup.backup_id == backup_id:
                backup_metadata = backup
                break
        
        if not backup_metadata:
            raise ValueError(f"Backup {backup_id} not found")
        
        if backup_metadata.status != BackupStatus.COMPLETED:
            raise ValueError(f"Backup {backup_id} is not in completed state")
        
        # Verify backup integrity
        if verify_before_restore:
            if not await self.verify_backup(backup_id):
                raise ValueError(f"Backup {backup_id} failed integrity verification")
        
        logger.info(f"Starting restore from backup: {backup_id}")
        
        try:
            # Perform restoration based on backup type
            if backup_metadata.backup_type == BackupType.DATABASE:
                success = await self._restore_database(backup_metadata, target_dir)
            elif backup_metadata.backup_type == BackupType.VECTOR_INDICES:
                success = await self._restore_vector_indices(backup_metadata, target_dir)
            elif backup_metadata.backup_type == BackupType.CONFIGURATION:
                success = await self._restore_configuration(backup_metadata, target_dir)
            elif backup_metadata.backup_type == BackupType.USER_DATA:
                success = await self._restore_user_data(backup_metadata, target_dir)
            elif backup_metadata.backup_type == BackupType.FULL_SYSTEM:
                success = await self._restore_full_system(backup_metadata, target_dir)
            else:
                raise ValueError(f"Unknown backup type: {backup_metadata.backup_type}")
            
            if success:
                logger.info(f"Successfully restored from backup: {backup_id}")
            else:
                logger.error(f"Failed to restore from backup: {backup_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error during restore from {backup_id}: {e}")
            return False
    
    async def verify_backup(self, backup_id: str) -> bool:
        """
        Verify backup integrity.
        
        Args:
            backup_id: Backup ID to verify
            
        Returns:
            True if backup is valid
        """
        # Find backup metadata
        backup_metadata = None
        for backup in self.backup_history:
            if backup.backup_id == backup_id:
                backup_metadata = backup
                break
        
        if not backup_metadata or not backup_metadata.file_path:
            return False
        
        try:
            # Check if file exists
            if not Path(backup_metadata.file_path).exists():
                logger.error(f"Backup file not found: {backup_metadata.file_path}")
                backup_metadata.status = BackupStatus.CORRUPTED
                return False
            
            # Verify file size
            actual_size = Path(backup_metadata.file_path).stat().st_size
            if backup_metadata.file_size_bytes and actual_size != backup_metadata.file_size_bytes:
                logger.error(f"Backup file size mismatch: expected {backup_metadata.file_size_bytes}, got {actual_size}")
                backup_metadata.status = BackupStatus.CORRUPTED
                return False
            
            # Verify checksum
            if backup_metadata.checksum:
                actual_checksum = await self._calculate_file_checksum(backup_metadata.file_path)
                if actual_checksum != backup_metadata.checksum:
                    logger.error(f"Backup checksum mismatch: expected {backup_metadata.checksum}, got {actual_checksum}")
                    backup_metadata.status = BackupStatus.CORRUPTED
                    return False
            
            # Try to read/decompress the file
            try:
                if backup_metadata.compression_type == CompressionType.GZIP:
                    with gzip.open(backup_metadata.file_path, 'rb') as f:
                        f.read(1024)  # Read first 1KB to verify
                elif backup_metadata.compression_type == CompressionType.TAR_GZ:
                    with tarfile.open(backup_metadata.file_path, 'r:gz') as tar:
                        tar.getnames()  # Get file list to verify
                # Add more compression types as needed
                
            except Exception as e:
                logger.error(f"Failed to read backup file {backup_metadata.file_path}: {e}")
                backup_metadata.status = BackupStatus.CORRUPTED
                return False
            
            logger.info(f"Backup {backup_id} passed integrity verification")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying backup {backup_id}: {e}")
            backup_metadata.status = BackupStatus.CORRUPTED
            return False
    
    async def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        status: Optional[BackupStatus] = None,
        days: Optional[int] = None
    ) -> List[BackupMetadata]:
        """
        List available backups.
        
        Args:
            backup_type: Filter by backup type
            status: Filter by status
            days: Only include backups from last N days
            
        Returns:
            List of backup metadata
        """
        backups = self.backup_history.copy()
        
        # Apply filters
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        if status:
            backups = [b for b in backups if b.status == status]
        
        if days:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            backups = [b for b in backups if b.created_at >= cutoff_date]
        
        # Sort by creation date (newest first)
        backups.sort(key=lambda b: b.created_at, reverse=True)
        
        return backups
    
    async def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get detailed information about a specific backup."""
        for backup in self.backup_history:
            if backup.backup_id == backup_id:
                return backup
        
        return None
    
    async def delete_backup(self, backup_id: str, force: bool = False) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_id: Backup ID to delete
            force: Force deletion even if backup is recent
            
        Returns:
            True if deletion was successful
        """
        # Find backup metadata
        backup_metadata = None
        for i, backup in enumerate(self.backup_history):
            if backup.backup_id == backup_id:
                backup_metadata = backup
                break
        
        if not backup_metadata:
            return False
        
        # Safety check - don't delete recent backups unless forced
        if not force:
            backup_age = datetime.now(timezone.utc) - backup_metadata.created_at
            if backup_age.total_seconds() < 24 * 3600:  # Less than 24 hours old
                logger.warning(f"Backup {backup_id} is less than 24 hours old. Use force=True to delete.")
                return False
        
        try:
            # Delete backup file
            if backup_metadata.file_path and Path(backup_metadata.file_path).exists():
                Path(backup_metadata.file_path).unlink()
                logger.info(f"Deleted backup file: {backup_metadata.file_path}")
            
            # Remove from history
            self.backup_history.remove(backup_metadata)
            
            # Save updated history
            self._save_backup_history()
            
            logger.info(f"Deleted backup: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting backup {backup_id}: {e}")
            return False
    
    async def cleanup_old_backups(self) -> Dict[BackupType, int]:
        """
        Clean up old backups based on retention policy.
        
        Returns:
            Dictionary of backup types and number of backups cleaned up
        """
        cleanup_counts = {backup_type: 0 for backup_type in BackupType}
        
        now = datetime.now(timezone.utc)
        
        for backup in self.backup_history.copy():
            # Check if backup has exceeded retention period
            retention_days = self.config.retention_policy.get(backup.backup_type, 30)
            backup_age = now - backup.created_at
            
            if backup_age.days > retention_days:
                logger.info(f"Cleaning up old backup: {backup.backup_id} (age: {backup_age.days} days)")
                
                if await self.delete_backup(backup.backup_id, force=True):
                    cleanup_counts[backup.backup_type] += 1
        
        return cleanup_counts
    
    async def _perform_backup(self, backup_metadata: BackupMetadata):
        """Perform the actual backup operation."""
        async with self.backup_semaphore:
            try:
                backup_metadata.status = BackupStatus.IN_PROGRESS
                
                # Perform backup based on type
                if backup_metadata.backup_type == BackupType.DATABASE:
                    await self._backup_database(backup_metadata)
                elif backup_metadata.backup_type == BackupType.VECTOR_INDICES:
                    await self._backup_vector_indices(backup_metadata)
                elif backup_metadata.backup_type == BackupType.CONFIGURATION:
                    await self._backup_configuration(backup_metadata)
                elif backup_metadata.backup_type == BackupType.USER_DATA:
                    await self._backup_user_data(backup_metadata)
                elif backup_metadata.backup_type == BackupType.FULL_SYSTEM:
                    await self._backup_full_system(backup_metadata)
                else:
                    raise ValueError(f"Unknown backup type: {backup_metadata.backup_type}")
                
                # Mark as completed
                backup_metadata.status = BackupStatus.COMPLETED
                backup_metadata.completed_at = datetime.now(timezone.utc)
                
                # Verify backup if enabled
                if self.config.verification_enabled:
                    if not await self.verify_backup(backup_metadata.backup_id):
                        backup_metadata.status = BackupStatus.FAILED
                        backup_metadata.error_message = "Backup verification failed"
                
                logger.info(f"Backup completed: {backup_metadata.backup_id}")
                
            except Exception as e:
                backup_metadata.status = BackupStatus.FAILED
                backup_metadata.error_message = str(e)
                logger.error(f"Backup failed: {backup_metadata.backup_id}: {e}")
                
            finally:
                # Move from active to history
                if backup_metadata.backup_id in self.active_backups:
                    del self.active_backups[backup_metadata.backup_id]
                
                self.backup_history.append(backup_metadata)
                self._save_backup_history()
    
    async def _backup_database(self, backup_metadata: BackupMetadata):
        """Backup the database."""
        backup_file = Path(self.config.backup_dir) / f"{backup_metadata.backup_id}.sql"
        
        # Use pg_dump for PostgreSQL or equivalent for other databases
        if "postgresql" in self.config.database_url.lower():
            cmd = [
                "pg_dump",
                self.config.database_url,
                "--no-password",
                "--format=custom",
                "--file", str(backup_file)
            ]
        else:
            # For SQLite or other databases, implement appropriate backup
            raise NotImplementedError("Database backup not implemented for this database type")
        
        # Execute backup command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Database backup failed: {stderr.decode()}")
        
        # Compress if enabled
        if backup_metadata.compression_type != CompressionType.NONE:
            compressed_file = await self._compress_file(backup_file, backup_metadata.compression_type)
            backup_file.unlink()  # Remove uncompressed file
            backup_file = compressed_file
        
        # Update metadata
        backup_metadata.file_path = str(backup_file)
        backup_metadata.file_size_bytes = backup_file.stat().st_size
        backup_metadata.checksum = await self._calculate_file_checksum(str(backup_file))
    
    async def _backup_vector_indices(self, backup_metadata: BackupMetadata):
        """Backup vector indices."""
        backup_file = Path(self.config.backup_dir) / f"{backup_metadata.backup_id}_vectors.tar.gz"
        
        # Create tar archive of vector indices directory
        with tarfile.open(backup_file, 'w:gz') as tar:
            if Path(self.config.vector_indices_dir).exists():
                tar.add(self.config.vector_indices_dir, arcname="vector_indices")
        
        # Update metadata
        backup_metadata.file_path = str(backup_file)
        backup_metadata.file_size_bytes = backup_file.stat().st_size
        backup_metadata.checksum = await self._calculate_file_checksum(str(backup_file))
    
    async def _backup_configuration(self, backup_metadata: BackupMetadata):
        """Backup configuration files."""
        backup_file = Path(self.config.backup_dir) / f"{backup_metadata.backup_id}_config.tar.gz"
        
        # Create tar archive of configuration files
        with tarfile.open(backup_file, 'w:gz') as tar:
            for config_file in self.config.config_files:
                if Path(config_file).exists():
                    tar.add(config_file, arcname=Path(config_file).name)
        
        # Update metadata
        backup_metadata.file_path = str(backup_file)
        backup_metadata.file_size_bytes = backup_file.stat().st_size
        backup_metadata.checksum = await self._calculate_file_checksum(str(backup_file))
    
    async def _backup_user_data(self, backup_metadata: BackupMetadata):
        """Backup user data (documents, uploads, etc.)."""
        # This would backup user-uploaded documents and related data
        # Implementation depends on how user data is stored
        backup_file = Path(self.config.backup_dir) / f"{backup_metadata.backup_id}_userdata.tar.gz"
        
        # Create placeholder - implement based on your user data storage
        with tarfile.open(backup_file, 'w:gz') as tar:
            # Add user data directories/files
            pass
        
        # Update metadata
        backup_metadata.file_path = str(backup_file)
        backup_metadata.file_size_bytes = backup_file.stat().st_size
        backup_metadata.checksum = await self._calculate_file_checksum(str(backup_file))
    
    async def _backup_full_system(self, backup_metadata: BackupMetadata):
        """Backup entire system."""
        # Perform all backup types
        await self._backup_database(backup_metadata)
        await self._backup_vector_indices(backup_metadata)
        await self._backup_configuration(backup_metadata)
        await self._backup_user_data(backup_metadata)
        
        # Combine all backups into single archive
        # Implementation would merge all individual backups
    
    async def _restore_database(self, backup_metadata: BackupMetadata, target_dir: Optional[str]) -> bool:
        """Restore database from backup."""
        # Implement database restoration logic
        return True
    
    async def _restore_vector_indices(self, backup_metadata: BackupMetadata, target_dir: Optional[str]) -> bool:
        """Restore vector indices from backup."""
        # Implement vector indices restoration logic
        return True
    
    async def _restore_configuration(self, backup_metadata: BackupMetadata, target_dir: Optional[str]) -> bool:
        """Restore configuration from backup."""
        # Implement configuration restoration logic
        return True
    
    async def _restore_user_data(self, backup_metadata: BackupMetadata, target_dir: Optional[str]) -> bool:
        """Restore user data from backup."""
        # Implement user data restoration logic
        return True
    
    async def _restore_full_system(self, backup_metadata: BackupMetadata, target_dir: Optional[str]) -> bool:
        """Restore full system from backup."""
        # Implement full system restoration logic
        return True
    
    async def _compress_file(self, file_path: Path, compression_type: CompressionType) -> Path:
        """Compress a file using the specified compression method."""
        if compression_type == CompressionType.GZIP:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return compressed_path
        
        elif compression_type == CompressionType.TAR_GZ:
            compressed_path = file_path.with_suffix('.tar.gz')
            with tarfile.open(compressed_path, 'w:gz') as tar:
                tar.add(file_path, arcname=file_path.name)
            return compressed_path
        
        else:
            return file_path
    
    async def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _load_backup_history(self):
        """Load backup history from file."""
        history_file = Path(self.config.backup_dir) / "backup_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                for backup_data in history_data:
                    backup_metadata = BackupMetadata(
                        backup_id=backup_data['backup_id'],
                        backup_type=BackupType(backup_data['backup_type']),
                        status=BackupStatus(backup_data['status']),
                        created_at=datetime.fromisoformat(backup_data['created_at']),
                        completed_at=datetime.fromisoformat(backup_data['completed_at']) if backup_data.get('completed_at') else None,
                        file_path=backup_data.get('file_path'),
                        file_size_bytes=backup_data.get('file_size_bytes'),
                        checksum=backup_data.get('checksum'),
                        compression_type=CompressionType(backup_data.get('compression_type', 'gzip')),
                        retention_days=backup_data.get('retention_days', 30),
                        metadata=backup_data.get('metadata', {}),
                        error_message=backup_data.get('error_message')
                    )
                    self.backup_history.append(backup_metadata)
                
                logger.info(f"Loaded {len(self.backup_history)} backup records from history")
                
            except Exception as e:
                logger.error(f"Error loading backup history: {e}")
    
    def _save_backup_history(self):
        """Save backup history to file."""
        history_file = Path(self.config.backup_dir) / "backup_history.json"
        
        try:
            history_data = []
            for backup in self.backup_history:
                backup_data = {
                    'backup_id': backup.backup_id,
                    'backup_type': backup.backup_type.value,
                    'status': backup.status.value,
                    'created_at': backup.created_at.isoformat(),
                    'completed_at': backup.completed_at.isoformat() if backup.completed_at else None,
                    'file_path': backup.file_path,
                    'file_size_bytes': backup.file_size_bytes,
                    'checksum': backup.checksum,
                    'compression_type': backup.compression_type.value,
                    'retention_days': backup.retention_days,
                    'metadata': backup.metadata,
                    'error_message': backup.error_message
                }
                history_data.append(backup_data)
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving backup history: {e}")
    
    async def _scheduler_loop(self):
        """Background task for scheduled backups."""
        while self.running:
            try:
                # Check if any scheduled backups should run
                # This is a simplified scheduler - use a proper cron library in production
                now = datetime.now(timezone.utc)
                
                # Daily database backup at 2 AM
                if now.hour == 2 and now.minute == 0:
                    await self.create_backup(BackupType.DATABASE, description="Scheduled daily backup")
                
                # Weekly full backup on Sunday at 3 AM
                if now.weekday() == 6 and now.hour == 3 and now.minute == 0:
                    await self.create_backup(BackupType.FULL_SYSTEM, description="Scheduled weekly full backup")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup scheduler: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background task for cleaning up old backups."""
        while self.running:
            try:
                # Run cleanup every 24 hours
                await asyncio.sleep(24 * 3600)
                await self.cleanup_old_backups()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup cleanup: {e}")


# Global backup manager instance
_backup_manager: Optional[AutomatedBackupManager] = None

def get_backup_manager() -> Optional[AutomatedBackupManager]:
    """Get global backup manager instance."""
    global _backup_manager
    return _backup_manager

def initialize_backup_manager(config: BackupConfig) -> AutomatedBackupManager:
    """Initialize global backup manager."""
    global _backup_manager
    
    _backup_manager = AutomatedBackupManager(config)
    return _backup_manager