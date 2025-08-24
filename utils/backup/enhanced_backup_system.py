"""
Enhanced Automated Backup System for RAG Application
Provides comprehensive backup and recovery capabilities with advanced features.
"""

import asyncio
import logging
import os
import shutil
import tarfile
import zipfile
import boto3
import paramiko
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import aiofiles
import hashlib
import subprocess

from database.connection import get_db
from utils.monitoring.metrics import MetricsCollector
from utils.monitoring.alert_manager import AlertManager

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backup operations"""
    FULL = "full"
    INCREMENTAL = "incremental" 
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupDestination(Enum):
    """Backup destination types"""
    LOCAL = "local"
    S3 = "s3"
    SFTP = "sftp"
    NFS = "nfs"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"


class BackupStatus(Enum):
    """Backup operation status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackupConfiguration:
    """Configuration for backup operations"""
    name: str
    backup_type: BackupType
    destination: BackupDestination
    schedule: str  # Cron expression
    retention_policy: int  # Days to retain backups
    compression: bool = True
    encryption: bool = False
    encryption_key: Optional[str] = None
    destination_config: Dict = field(default_factory=dict)
    include_paths: List[str] = field(default_factory=list)
    exclude_paths: List[str] = field(default_factory=list)
    pre_backup_scripts: List[str] = field(default_factory=list)
    post_backup_scripts: List[str] = field(default_factory=list)
    notification_settings: Dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class BackupJob:
    """Backup job execution details"""
    job_id: str
    config: BackupConfiguration
    status: BackupStatus = BackupStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    files_processed: int = 0
    total_files: int = 0
    bytes_processed: int = 0
    total_bytes: int = 0
    backup_path: Optional[str] = None
    checksum: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RestoreJob:
    """Restore job execution details"""
    job_id: str
    backup_path: str
    destination_path: str
    status: BackupStatus = BackupStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    files_restored: int = 0
    bytes_restored: int = 0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class EnhancedBackupSystem:
    """Enhanced automated backup system"""
    
    def __init__(self, config_file: str = "config/backup_config.json"):
        self.config_file = Path(config_file)
        self.backup_configs: Dict[str, BackupConfiguration] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.backup_history: List[BackupJob] = []
        self.restore_history: List[RestoreJob] = []
        self.metrics = MetricsCollector()
        self.alert_manager = AlertManager()
        self._scheduler_task = None
        self._stop_scheduler = False
        
        # Default paths
        self.default_include_paths = [
            "database/",
            "documents/",
            "config/",
            "logs/",
            "cache/fingerprints/",
            "models/"
        ]
        
        self.default_exclude_paths = [
            "*.log",
            "*.tmp",
            "__pycache__/",
            "*.pyc",
            ".git/",
            "node_modules/"
        ]
    
    async def initialize(self):
        """Initialize the backup system"""
        try:
            await self._load_configurations()
            await self._start_scheduler()
            logger.info("Enhanced backup system initialized")
        except Exception as e:
            logger.error(f"Error initializing backup system: {e}")
            raise
    
    async def _load_configurations(self):
        """Load backup configurations from file"""
        if self.config_file.exists():
            try:
                async with aiofiles.open(self.config_file, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                
                for config_data in data.get('configurations', []):
                    config = BackupConfiguration(
                        name=config_data['name'],
                        backup_type=BackupType(config_data['backup_type']),
                        destination=BackupDestination(config_data['destination']),
                        schedule=config_data['schedule'],
                        retention_policy=config_data['retention_policy'],
                        compression=config_data.get('compression', True),
                        encryption=config_data.get('encryption', False),
                        encryption_key=config_data.get('encryption_key'),
                        destination_config=config_data.get('destination_config', {}),
                        include_paths=config_data.get('include_paths', self.default_include_paths),
                        exclude_paths=config_data.get('exclude_paths', self.default_exclude_paths),
                        pre_backup_scripts=config_data.get('pre_backup_scripts', []),
                        post_backup_scripts=config_data.get('post_backup_scripts', []),
                        notification_settings=config_data.get('notification_settings', {}),
                        enabled=config_data.get('enabled', True)
                    )
                    self.backup_configs[config.name] = config
                
                logger.info(f"Loaded {len(self.backup_configs)} backup configurations")
                
            except Exception as e:
                logger.error(f"Error loading backup configurations: {e}")
        else:
            # Create default configuration
            await self._create_default_config()
    
    async def _create_default_config(self):
        """Create default backup configuration"""
        default_config = BackupConfiguration(
            name="daily_full_backup",
            backup_type=BackupType.FULL,
            destination=BackupDestination.LOCAL,
            schedule="0 2 * * *",  # Daily at 2 AM
            retention_policy=30,   # Keep for 30 days
            destination_config={"path": "backups/"},
            include_paths=self.default_include_paths,
            exclude_paths=self.default_exclude_paths
        )
        
        self.backup_configs[default_config.name] = default_config
        await self._save_configurations()
    
    async def _save_configurations(self):
        """Save backup configurations to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'configurations': []
            }
            
            for config in self.backup_configs.values():
                config_data = {
                    'name': config.name,
                    'backup_type': config.backup_type.value,
                    'destination': config.destination.value,
                    'schedule': config.schedule,
                    'retention_policy': config.retention_policy,
                    'compression': config.compression,
                    'encryption': config.encryption,
                    'encryption_key': config.encryption_key,
                    'destination_config': config.destination_config,
                    'include_paths': config.include_paths,
                    'exclude_paths': config.exclude_paths,
                    'pre_backup_scripts': config.pre_backup_scripts,
                    'post_backup_scripts': config.post_backup_scripts,
                    'notification_settings': config.notification_settings,
                    'enabled': config.enabled
                }
                data['configurations'].append(config_data)
            
            async with aiofiles.open(self.config_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
        except Exception as e:
            logger.error(f"Error saving backup configurations: {e}")
    
    async def create_backup(self, config_name: str, backup_type: Optional[BackupType] = None) -> str:
        """Create a backup job"""
        if config_name not in self.backup_configs:
            raise ValueError(f"Backup configuration {config_name} not found")
        
        config = self.backup_configs[config_name]
        if backup_type:
            # Override backup type for this job
            config = BackupConfiguration(**config.__dict__)
            config.backup_type = backup_type
        
        # Generate job ID
        job_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{config_name}"
        
        # Create backup job
        job = BackupJob(
            job_id=job_id,
            config=config
        )
        
        # Start backup in background
        task = asyncio.create_task(self._run_backup(job))
        self.active_jobs[job_id] = task
        
        return job_id
    
    async def _run_backup(self, job: BackupJob):
        """Run the backup process"""
        try:
            job.status = BackupStatus.RUNNING
            job.started_at = datetime.utcnow()
            
            logger.info(f"Starting backup job {job.job_id}")
            
            # Run pre-backup scripts
            await self._run_scripts(job.config.pre_backup_scripts, "pre-backup")
            
            # Create database backup
            await self._backup_database(job)
            
            # Create file system backup
            await self._backup_files(job)
            
            # Upload to destination
            await self._upload_backup(job)
            
            # Calculate checksum
            if job.backup_path:
                job.checksum = await self._calculate_checksum(job.backup_path)
            
            # Run post-backup scripts
            await self._run_scripts(job.config.post_backup_scripts, "post-backup")
            
            # Clean up old backups
            await self._cleanup_old_backups(job.config)
            
            # Complete job
            job.status = BackupStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            
            # Store in history
            self.backup_history.append(job)
            
            # Send notification
            await self._send_backup_notification(job, success=True)
            
            logger.info(f"Backup job {job.job_id} completed successfully")
            
        except Exception as e:
            job.status = BackupStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            
            self.backup_history.append(job)
            
            await self._send_backup_notification(job, success=False)
            
            logger.error(f"Backup job {job.job_id} failed: {e}")
        
        finally:
            # Remove from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    async def _backup_database(self, job: BackupJob):
        """Create database backup"""
        try:
            backup_dir = Path("temp_backup") / job.job_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Export database
            db_backup_path = backup_dir / "database_backup.sql"
            
            # This is a simplified database backup - in practice, you'd use proper database tools
            async with get_db_session() as session:
                # Get all table names
                tables_result = await session.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = [row[0] for row in tables_result.fetchall()]
                
                backup_sql = []
                for table in tables:
                    # Get table schema
                    schema_result = await session.execute(f"SELECT sql FROM sqlite_master WHERE name='{table}'")
                    schema = schema_result.fetchone()
                    if schema:
                        backup_sql.append(f"{schema[0]};")
                    
                    # Get table data
                    data_result = await session.execute(f"SELECT * FROM {table}")
                    rows = data_result.fetchall()
                    
                    if rows:
                        columns_result = await session.execute(f"PRAGMA table_info({table})")
                        columns = [col[1] for col in columns_result.fetchall()]
                        
                        for row in rows:
                            values = []
                            for value in row:
                                if value is None:
                                    values.append("NULL")
                                elif isinstance(value, str):
                                    escaped_value = value.replace("'", "''")
                                    values.append(f"'{escaped_value}'")
                                else:
                                    values.append(str(value))
                            
                            backup_sql.append(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join(values)});")
            
            # Write backup SQL
            async with aiofiles.open(db_backup_path, 'w') as f:
                await f.write('\n'.join(backup_sql))
            
            job.progress = 20.0
            logger.info(f"Database backup created: {db_backup_path}")
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            raise
    
    async def _backup_files(self, job: BackupJob):
        """Create file system backup"""
        try:
            backup_dir = Path("temp_backup") / job.job_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine what files to include
            files_to_backup = []
            total_size = 0
            
            for include_path in job.config.include_paths:
                path = Path(include_path)
                if path.exists():
                    if path.is_file():
                        if not self._is_excluded(str(path), job.config.exclude_paths):
                            files_to_backup.append(path)
                            total_size += path.stat().st_size
                    else:
                        for file_path in path.rglob("*"):
                            if file_path.is_file() and not self._is_excluded(str(file_path), job.config.exclude_paths):
                                files_to_backup.append(file_path)
                                total_size += file_path.stat().st_size
            
            job.total_files = len(files_to_backup)
            job.total_bytes = total_size
            
            # Create archive
            archive_name = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            if job.config.compression:
                archive_path = backup_dir / f"{archive_name}.tar.gz"
                mode = 'w:gz'
            else:
                archive_path = backup_dir / f"{archive_name}.tar"
                mode = 'w'
            
            with tarfile.open(archive_path, mode) as tar:
                processed_bytes = 0
                for i, file_path in enumerate(files_to_backup):
                    try:
                        tar.add(file_path, arcname=str(file_path))
                        processed_bytes += file_path.stat().st_size
                        job.files_processed = i + 1
                        job.bytes_processed = processed_bytes
                        job.progress = 20.0 + (processed_bytes / total_size) * 50.0  # 20-70%
                    except Exception as e:
                        logger.warning(f"Error adding file {file_path} to backup: {e}")
            
            job.backup_path = str(archive_path)
            logger.info(f"File backup created: {archive_path}")
            
        except Exception as e:
            logger.error(f"Error creating file backup: {e}")
            raise
    
    def _is_excluded(self, file_path: str, exclude_patterns: List[str]) -> bool:
        """Check if file should be excluded from backup"""
        import fnmatch
        
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(file_path, pattern) or pattern in file_path:
                return True
        return False
    
    async def _upload_backup(self, job: BackupJob):
        """Upload backup to destination"""
        if not job.backup_path:
            return
        
        try:
            if job.config.destination == BackupDestination.LOCAL:
                await self._upload_to_local(job)
            elif job.config.destination == BackupDestination.S3:
                await self._upload_to_s3(job)
            elif job.config.destination == BackupDestination.SFTP:
                await self._upload_to_sftp(job)
            # Add other destination handlers as needed
            
            job.progress = 90.0
            
        except Exception as e:
            logger.error(f"Error uploading backup: {e}")
            raise
    
    async def _upload_to_local(self, job: BackupJob):
        """Upload backup to local destination"""
        dest_config = job.config.destination_config
        dest_path = Path(dest_config.get('path', 'backups/'))
        dest_path.mkdir(parents=True, exist_ok=True)
        
        backup_file = Path(job.backup_path)
        final_path = dest_path / backup_file.name
        
        shutil.move(str(backup_file), str(final_path))
        job.backup_path = str(final_path)
    
    async def _upload_to_s3(self, job: BackupJob):
        """Upload backup to AWS S3"""
        dest_config = job.config.destination_config
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=dest_config.get('access_key_id'),
            aws_secret_access_key=dest_config.get('secret_access_key'),
            region_name=dest_config.get('region', 'us-east-1')
        )
        
        bucket = dest_config['bucket']
        prefix = dest_config.get('prefix', 'backups/')
        
        backup_file = Path(job.backup_path)
        s3_key = f"{prefix}{backup_file.name}"
        
        # Upload file
        s3_client.upload_file(str(backup_file), bucket, s3_key)
        
        # Update backup path to S3 location
        job.backup_path = f"s3://{bucket}/{s3_key}"
        
        # Clean up local file
        backup_file.unlink()
    
    async def _upload_to_sftp(self, job: BackupJob):
        """Upload backup to SFTP server"""
        dest_config = job.config.destination_config
        
        # Create SFTP connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        ssh.connect(
            hostname=dest_config['hostname'],
            username=dest_config['username'],
            password=dest_config.get('password'),
            key_filename=dest_config.get('key_filename'),
            port=dest_config.get('port', 22)
        )
        
        sftp = ssh.open_sftp()
        
        backup_file = Path(job.backup_path)
        remote_path = f"{dest_config.get('remote_path', '/backups/')}{backup_file.name}"
        
        # Upload file
        sftp.put(str(backup_file), remote_path)
        
        # Update backup path to SFTP location
        job.backup_path = f"sftp://{dest_config['hostname']}{remote_path}"
        
        # Clean up
        sftp.close()
        ssh.close()
        backup_file.unlink()
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of backup file"""
        hash_md5 = hashlib.md5()
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def _cleanup_old_backups(self, config: BackupConfiguration):
        """Clean up old backups based on retention policy"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=config.retention_policy)
            
            # Clean up based on destination type
            if config.destination == BackupDestination.LOCAL:
                backup_dir = Path(config.destination_config.get('path', 'backups/'))
                if backup_dir.exists():
                    for backup_file in backup_dir.glob("backup_*.tar*"):
                        if datetime.fromtimestamp(backup_file.stat().st_mtime) < cutoff_date:
                            backup_file.unlink()
                            logger.info(f"Deleted old backup: {backup_file}")
            
            # Add cleanup for other destination types as needed
            
        except Exception as e:
            logger.warning(f"Error cleaning up old backups: {e}")
    
    async def _run_scripts(self, scripts: List[str], script_type: str):
        """Run pre/post backup scripts"""
        for script in scripts:
            try:
                logger.info(f"Running {script_type} script: {script}")
                process = await asyncio.create_subprocess_shell(
                    script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.warning(f"{script_type} script failed: {stderr.decode()}")
                else:
                    logger.info(f"{script_type} script completed successfully")
                    
            except Exception as e:
                logger.error(f"Error running {script_type} script {script}: {e}")
    
    async def _send_backup_notification(self, job: BackupJob, success: bool):
        """Send backup notification"""
        if not job.config.notification_settings:
            return
        
        try:
            message = f"Backup job {job.job_id} "
            if success:
                message += f"completed successfully. Files: {job.files_processed}, Size: {job.bytes_processed} bytes"
            else:
                message += f"failed. Error: {job.error_message}"
            
            # Send alert
            await self.alert_manager.send_alert(
                title=f"Backup {'Success' if success else 'Failure'}",
                message=message,
                level='info' if success else 'error',
                tags=['backup', job.config.name]
            )
            
        except Exception as e:
            logger.error(f"Error sending backup notification: {e}")
    
    async def _start_scheduler(self):
        """Start the backup scheduler"""
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while not self._stop_scheduler:
            try:
                current_time = datetime.utcnow()
                
                for config_name, config in self.backup_configs.items():
                    if config.enabled and self._should_run_backup(config, current_time):
                        # Start backup
                        job_id = await self.create_backup(config_name)
                        logger.info(f"Scheduled backup started: {job_id}")
                
                # Check every minute
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in backup scheduler: {e}")
                await asyncio.sleep(60)
    
    def _should_run_backup(self, config: BackupConfiguration, current_time: datetime) -> bool:
        """Check if backup should run based on schedule"""
        # This is a simplified cron parser - in practice, use a proper cron library
        # For now, just check if it's the right hour for daily backups
        if config.schedule.startswith("0 "):
            hour = int(config.schedule.split()[1])
            return current_time.hour == hour and current_time.minute == 0
        
        return False
    
    async def restore_backup(self, backup_path: str, destination_path: str) -> str:
        """Restore from backup"""
        job_id = f"restore_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        job = RestoreJob(
            job_id=job_id,
            backup_path=backup_path,
            destination_path=destination_path
        )
        
        # Start restore in background
        task = asyncio.create_task(self._run_restore(job))
        self.active_jobs[job_id] = task
        
        return job_id
    
    async def _run_restore(self, job: RestoreJob):
        """Run the restore process"""
        try:
            job.status = BackupStatus.RUNNING
            job.started_at = datetime.utcnow()
            
            logger.info(f"Starting restore job {job.job_id}")
            
            # Download backup if needed
            local_backup_path = await self._download_backup(job.backup_path)
            
            # Extract backup
            dest_path = Path(job.destination_path)
            dest_path.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(local_backup_path, 'r:*') as tar:
                tar.extractall(dest_path)
                job.progress = 80.0
            
            # Restore database if backup contains database dump
            db_backup = dest_path / "database_backup.sql"
            if db_backup.exists():
                await self._restore_database(str(db_backup))
                job.progress = 95.0
            
            job.status = BackupStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            
            self.restore_history.append(job)
            
            logger.info(f"Restore job {job.job_id} completed successfully")
            
        except Exception as e:
            job.status = BackupStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            
            self.restore_history.append(job)
            
            logger.error(f"Restore job {job.job_id} failed: {e}")
        
        finally:
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    async def _download_backup(self, backup_path: str) -> str:
        """Download backup from remote location if needed"""
        if backup_path.startswith('s3://'):
            # Download from S3
            # Implementation would go here
            pass
        elif backup_path.startswith('sftp://'):
            # Download from SFTP
            # Implementation would go here
            pass
        
        # For local files, return as-is
        return backup_path
    
    async def _restore_database(self, backup_file: str):
        """Restore database from backup file"""
        try:
            async with aiofiles.open(backup_file, 'r') as f:
                sql_content = await f.read()
            
            async with get_db_session() as session:
                # Execute SQL statements
                statements = sql_content.split(';')
                for statement in statements:
                    statement = statement.strip()
                    if statement:
                        await session.execute(statement)
                await session.commit()
            
            logger.info("Database restored successfully")
            
        except Exception as e:
            logger.error(f"Error restoring database: {e}")
            raise
    
    async def get_backup_status(self) -> Dict:
        """Get status of backup system"""
        return {
            'total_configurations': len(self.backup_configs),
            'active_jobs': len(self.active_jobs),
            'recent_backups': len([j for j in self.backup_history if (datetime.utcnow() - j.created_at).days <= 7]),
            'configurations': {
                name: {
                    'enabled': config.enabled,
                    'backup_type': config.backup_type.value,
                    'destination': config.destination.value,
                    'schedule': config.schedule,
                    'retention_policy': config.retention_policy
                }
                for name, config in self.backup_configs.items()
            }
        }
    
    async def cleanup(self):
        """Cleanup backup system resources"""
        self._stop_scheduler = True
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active jobs
        for task in self.active_jobs.values():
            task.cancel()
        
        if self.active_jobs:
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)


# Global instance
enhanced_backup_system = EnhancedBackupSystem()