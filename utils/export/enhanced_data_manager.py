"""
Enhanced data export/import management system for RAG application.
Provides comprehensive data management, migration, and synchronization capabilities.
"""

import asyncio
import logging
import json
import csv
import gzip
import tarfile
import zipfile
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text
import aiofiles
import hashlib

from database.models import User, Document, DocumentChunk, VectorIndex
from utils.export.data_exporter import DataExporter
from utils.backup.automated_backup import BackupManager

logger = logging.getLogger(__name__)

class DataFormat(Enum):
    """Data export/import formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    SQL = "sql"
    XML = "xml"
    YAML = "yaml"

class CompressionFormat(Enum):
    """Compression formats."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    TAR_GZ = "tar_gz"

class ImportMode(Enum):
    """Import modes."""
    CREATE_ONLY = "create_only"  # Only create new records
    UPDATE_ONLY = "update_only"  # Only update existing records
    UPSERT = "upsert"  # Create or update
    REPLACE = "replace"  # Replace all data
    MERGE = "merge"  # Intelligent merge

class DataSyncDirection(Enum):
    """Data synchronization directions."""
    EXPORT = "export"
    IMPORT = "import"
    BIDIRECTIONAL = "bidirectional"

@dataclass
class ExportConfig:
    """Export configuration."""
    format: DataFormat
    compression: CompressionFormat = CompressionFormat.NONE
    include_metadata: bool = True
    include_relationships: bool = True
    batch_size: int = 1000
    filter_conditions: Optional[Dict[str, Any]] = None
    selected_fields: Optional[List[str]] = None
    date_range: Optional[Tuple[datetime, datetime]] = None

@dataclass
class ImportConfig:
    """Import configuration."""
    format: DataFormat
    mode: ImportMode = ImportMode.UPSERT
    batch_size: int = 1000
    validate_data: bool = True
    skip_duplicates: bool = True
    create_missing_users: bool = False
    field_mapping: Optional[Dict[str, str]] = None
    transformation_rules: Optional[Dict[str, Any]] = None

@dataclass
class DataOperation:
    """Data operation tracking."""
    operation_id: str
    operation_type: str  # export, import, sync, migration
    status: str  # pending, processing, completed, failed
    source: Optional[str] = None
    target: Optional[str] = None
    progress: float = 0.0
    total_records: int = 0
    processed_records: int = 0
    error_records: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedDataManager:
    """
    Enhanced data export/import management system.
    
    Features:
    - Multiple data formats support
    - Batch processing with progress tracking
    - Data validation and transformation
    - Incremental imports/exports
    - Data synchronization
    - Compression and optimization
    - Relationship preservation
    - Migration utilities
    """
    
    def __init__(
        self,
        export_dir: str = "data_exports",
        import_dir: str = "data_imports",
        temp_dir: str = "data_temp",
        max_file_size_mb: int = 500
    ):
        """
        Initialize enhanced data manager.
        
        Args:
            export_dir: Directory for export files
            import_dir: Directory for import files
            temp_dir: Temporary directory for processing
            max_file_size_mb: Maximum file size in MB
        """
        self.export_dir = Path(export_dir)
        self.import_dir = Path(import_dir)
        self.temp_dir = Path(temp_dir)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        # Create directories
        for dir_path in [self.export_dir, self.import_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Operation tracking
        self.active_operations: Dict[str, DataOperation] = {}
        self.operation_history: List[DataOperation] = []
        
        # Data processors
        self.format_processors = {
            DataFormat.JSON: self._process_json,
            DataFormat.CSV: self._process_csv,
            DataFormat.EXCEL: self._process_excel,
            DataFormat.PARQUET: self._process_parquet,
            DataFormat.SQL: self._process_sql,
            DataFormat.XML: self._process_xml,
            DataFormat.YAML: self._process_yaml
        }
        
        # Base data exporter
        self.base_exporter = DataExporter(str(self.export_dir))
    
    async def export_users(
        self,
        config: ExportConfig,
        db: Session,
        user_filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export user data.
        
        Args:
            config: Export configuration
            db: Database session
            user_filter: Optional user filtering criteria
            
        Returns:
            Operation ID
        """
        operation_id = f"export_users_{int(datetime.now(timezone.utc).timestamp())}"
        
        operation = DataOperation(
            operation_id=operation_id,
            operation_type="export_users",
            status="pending",
            target=str(self.export_dir)
        )
        
        self.active_operations[operation_id] = operation
        
        # Start export process
        asyncio.create_task(self._export_users_process(operation, config, db, user_filter))
        
        return operation_id
    
    async def export_documents(
        self,
        config: ExportConfig,
        db: Session,
        user_id: Optional[int] = None,
        include_content: bool = False,
        include_vectors: bool = False
    ) -> str:
        """
        Export document data.
        
        Args:
            config: Export configuration
            db: Database session
            user_id: Filter by specific user
            include_content: Include document content
            include_vectors: Include vector embeddings
            
        Returns:
            Operation ID
        """
        operation_id = f"export_documents_{int(datetime.now(timezone.utc).timestamp())}"
        
        operation = DataOperation(
            operation_id=operation_id,
            operation_type="export_documents",
            status="pending",
            target=str(self.export_dir),
            metadata={
                'user_id': user_id,
                'include_content': include_content,
                'include_vectors': include_vectors
            }
        )
        
        self.active_operations[operation_id] = operation
        
        # Start export process
        asyncio.create_task(self._export_documents_process(operation, config, db))
        
        return operation_id
    
    async def export_complete_system(
        self,
        config: ExportConfig,
        db: Session,
        include_vectors: bool = True,
        include_user_data: bool = True
    ) -> str:
        """
        Export complete system data.
        
        Args:
            config: Export configuration
            db: Database session
            include_vectors: Include vector data
            include_user_data: Include user-specific data
            
        Returns:
            Operation ID
        """
        operation_id = f"export_system_{int(datetime.now(timezone.utc).timestamp())}"
        
        operation = DataOperation(
            operation_id=operation_id,
            operation_type="export_system",
            status="pending",
            target=str(self.export_dir),
            metadata={
                'include_vectors': include_vectors,
                'include_user_data': include_user_data
            }
        )
        
        self.active_operations[operation_id] = operation
        
        # Start export process
        asyncio.create_task(self._export_system_process(operation, config, db))
        
        return operation_id
    
    async def import_users(
        self,
        file_path: str,
        config: ImportConfig,
        db: Session
    ) -> str:
        """
        Import user data.
        
        Args:
            file_path: Path to import file
            config: Import configuration
            db: Database session
            
        Returns:
            Operation ID
        """
        operation_id = f"import_users_{int(datetime.now(timezone.utc).timestamp())}"
        
        operation = DataOperation(
            operation_id=operation_id,
            operation_type="import_users",
            status="pending",
            source=file_path
        )
        
        self.active_operations[operation_id] = operation
        
        # Start import process
        asyncio.create_task(self._import_users_process(operation, file_path, config, db))
        
        return operation_id
    
    async def import_documents(
        self,
        file_path: str,
        config: ImportConfig,
        db: Session,
        user_id: Optional[int] = None
    ) -> str:
        """
        Import document data.
        
        Args:
            file_path: Path to import file
            config: Import configuration
            db: Database session
            user_id: Associate with specific user
            
        Returns:
            Operation ID
        """
        operation_id = f"import_documents_{int(datetime.now(timezone.utc).timestamp())}"
        
        operation = DataOperation(
            operation_id=operation_id,
            operation_type="import_documents",
            status="pending",
            source=file_path,
            metadata={'user_id': user_id}
        )
        
        self.active_operations[operation_id] = operation
        
        # Start import process
        asyncio.create_task(self._import_documents_process(operation, file_path, config, db))
        
        return operation_id
    
    async def synchronize_data(
        self,
        source_config: Dict[str, Any],
        target_config: Dict[str, Any],
        sync_direction: DataSyncDirection = DataSyncDirection.BIDIRECTIONAL,
        db: Session = None
    ) -> str:
        """
        Synchronize data between systems.
        
        Args:
            source_config: Source configuration
            target_config: Target configuration
            sync_direction: Synchronization direction
            db: Database session
            
        Returns:
            Operation ID
        """
        operation_id = f"sync_data_{int(datetime.now(timezone.utc).timestamp())}"
        
        operation = DataOperation(
            operation_id=operation_id,
            operation_type="synchronize_data",
            status="pending",
            source=str(source_config),
            target=str(target_config),
            metadata={'sync_direction': sync_direction.value}
        )
        
        self.active_operations[operation_id] = operation
        
        # Start sync process
        asyncio.create_task(self._synchronize_data_process(operation, source_config, target_config, sync_direction, db))
        
        return operation_id
    
    async def migrate_data_format(
        self,
        source_file: str,
        source_format: DataFormat,
        target_format: DataFormat,
        target_file: Optional[str] = None
    ) -> str:
        """
        Migrate data from one format to another.
        
        Args:
            source_file: Source file path
            source_format: Source data format
            target_format: Target data format
            target_file: Optional target file path
            
        Returns:
            Operation ID
        """
        operation_id = f"migrate_format_{int(datetime.now(timezone.utc).timestamp())}"
        
        if not target_file:
            source_path = Path(source_file)
            target_file = str(source_path.with_suffix(f'.{target_format.value}'))
        
        operation = DataOperation(
            operation_id=operation_id,
            operation_type="migrate_format",
            status="pending",
            source=source_file,
            target=target_file,
            metadata={
                'source_format': source_format.value,
                'target_format': target_format.value
            }
        )
        
        self.active_operations[operation_id] = operation
        
        # Start migration process
        asyncio.create_task(self._migrate_format_process(operation, source_file, source_format, target_format, target_file))
        
        return operation_id
    
    async def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a data operation."""
        operation = self.active_operations.get(operation_id)
        if not operation:
            # Check history
            for hist_op in self.operation_history:
                if hist_op.operation_id == operation_id:
                    operation = hist_op
                    break
        
        if not operation:
            return None
        
        return {
            'operation_id': operation.operation_id,
            'operation_type': operation.operation_type,
            'status': operation.status,
            'progress': operation.progress,
            'total_records': operation.total_records,
            'processed_records': operation.processed_records,
            'error_records': operation.error_records,
            'created_at': operation.created_at.isoformat(),
            'started_at': operation.started_at.isoformat() if operation.started_at else None,
            'completed_at': operation.completed_at.isoformat() if operation.completed_at else None,
            'error_message': operation.error_message,
            'metadata': operation.metadata,
            'source': operation.source,
            'target': operation.target
        }
    
    async def list_operations(
        self,
        operation_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List data operations with optional filtering."""
        all_operations = list(self.active_operations.values()) + self.operation_history
        
        # Apply filters
        if operation_type:
            all_operations = [op for op in all_operations if op.operation_type == operation_type]
        
        if status:
            all_operations = [op for op in all_operations if op.status == status]
        
        # Sort by creation time (newest first)
        all_operations.sort(key=lambda op: op.created_at, reverse=True)
        
        # Limit results
        all_operations = all_operations[:limit]
        
        # Convert to dictionaries
        return [await self.get_operation_status(op.operation_id) for op in all_operations]
    
    async def cancel_operation(self, operation_id: str) -> bool:
        """Cancel an active operation."""
        if operation_id not in self.active_operations:
            return False
        
        operation = self.active_operations[operation_id]
        operation.status = "cancelled"
        operation.completed_at = datetime.now(timezone.utc)
        
        # Move to history
        self.operation_history.append(operation)
        del self.active_operations[operation_id]
        
        logger.info(f"Cancelled operation: {operation_id}")
        
        return True
    
    async def _export_users_process(
        self,
        operation: DataOperation,
        config: ExportConfig,
        db: Session,
        user_filter: Optional[Dict[str, Any]]
    ):
        """Process user data export."""
        try:
            operation.status = "processing"
            operation.started_at = datetime.now(timezone.utc)
            
            # Query users
            query = db.query(User)
            
            # Apply filters
            if user_filter:
                for field, value in user_filter.items():
                    if hasattr(User, field):
                        query = query.filter(getattr(User, field) == value)
            
            # Apply date range
            if config.date_range:
                start_date, end_date = config.date_range
                query = query.filter(User.created_at.between(start_date, end_date))
            
            users = query.all()
            operation.total_records = len(users)
            
            # Convert to export format
            export_data = []
            for i, user in enumerate(users):
                user_data = {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'is_active': user.is_active,
                    'is_superuser': user.is_superuser,
                    'created_at': user.created_at.isoformat(),
                    'updated_at': user.updated_at.isoformat() if user.updated_at else None
                }
                
                # Include relationships if requested
                if config.include_relationships:
                    user_data['documents_count'] = len(user.documents)
                    user_data['roles'] = [role.name for role in user.roles] if hasattr(user, 'roles') else []
                
                # Apply field selection
                if config.selected_fields:
                    user_data = {k: v for k, v in user_data.items() if k in config.selected_fields}
                
                export_data.append(user_data)
                
                # Update progress
                operation.processed_records = i + 1
                operation.progress = (operation.processed_records / operation.total_records) * 100
            
            # Export data
            output_file = await self._export_data_to_file(export_data, config, "users")
            
            operation.status = "completed"
            operation.completed_at = datetime.now(timezone.utc)
            operation.target = output_file
            
            logger.info(f"Completed user export: {operation.operation_id}")
            
        except Exception as e:
            operation.status = "failed"
            operation.error_message = str(e)
            operation.completed_at = datetime.now(timezone.utc)
            logger.error(f"User export failed: {operation.operation_id}: {e}")
            
        finally:
            # Move to history
            if operation.operation_id in self.active_operations:
                self.operation_history.append(operation)
                del self.active_operations[operation.operation_id]
    
    async def _export_documents_process(
        self,
        operation: DataOperation,
        config: ExportConfig,
        db: Session
    ):
        """Process document data export."""
        try:
            operation.status = "processing"
            operation.started_at = datetime.now(timezone.utc)
            
            # Query documents
            query = db.query(Document).filter(Document.is_deleted == False)
            
            # Apply user filter
            if operation.metadata.get('user_id'):
                query = query.filter(Document.user_id == operation.metadata['user_id'])
            
            # Apply date range
            if config.date_range:
                start_date, end_date = config.date_range
                query = query.filter(Document.created_at.between(start_date, end_date))
            
            documents = query.all()
            operation.total_records = len(documents)
            
            # Convert to export format
            export_data = []
            for i, doc in enumerate(documents):
                doc_data = {
                    'id': doc.id,
                    'user_id': doc.user_id,
                    'filename': doc.filename,
                    'title': doc.title,
                    'description': doc.description,
                    'content_type': doc.content_type,
                    'file_size': doc.file_size,
                    'status': doc.status.value if doc.status else None,
                    'is_public': doc.is_public,
                    'created_at': doc.created_at.isoformat(),
                    'updated_at': doc.updated_at.isoformat() if doc.updated_at else None
                }
                
                # Include content if requested
                if operation.metadata.get('include_content'):
                    doc_data['extracted_text'] = doc.extracted_text
                
                # Include metadata
                if config.include_metadata:
                    doc_data['metadata'] = doc.get_metadata_dict()
                    doc_data['tags'] = doc.get_tags()
                
                # Include relationships
                if config.include_relationships:
                    doc_data['chunks_count'] = len(doc.chunks) if hasattr(doc, 'chunks') else 0
                    doc_data['versions_count'] = len(doc.versions) if hasattr(doc, 'versions') else 0
                
                export_data.append(doc_data)
                
                # Update progress
                operation.processed_records = i + 1
                operation.progress = (operation.processed_records / operation.total_records) * 100
            
            # Export data
            output_file = await self._export_data_to_file(export_data, config, "documents")
            
            operation.status = "completed"
            operation.completed_at = datetime.now(timezone.utc)
            operation.target = output_file
            
            logger.info(f"Completed document export: {operation.operation_id}")
            
        except Exception as e:
            operation.status = "failed"
            operation.error_message = str(e)
            operation.completed_at = datetime.now(timezone.utc)
            logger.error(f"Document export failed: {operation.operation_id}: {e}")
            
        finally:
            # Move to history
            if operation.operation_id in self.active_operations:
                self.operation_history.append(operation)
                del self.active_operations[operation.operation_id]
    
    async def _export_system_process(
        self,
        operation: DataOperation,
        config: ExportConfig,
        db: Session
    ):
        """Process complete system export."""
        try:
            operation.status = "processing"
            operation.started_at = datetime.now(timezone.utc)
            
            # Export multiple data types
            export_components = []
            
            # Users
            if operation.metadata.get('include_user_data', True):
                users_data = await self._get_users_data(db, config)
                export_components.append(('users', users_data))
            
            # Documents
            documents_data = await self._get_documents_data(db, config, operation.metadata)
            export_components.append(('documents', documents_data))
            
            # System configuration
            config_data = await self._get_system_config_data()
            export_components.append(('configuration', config_data))
            
            # Calculate total records
            operation.total_records = sum(len(data) for _, data in export_components)
            
            # Create combined export
            combined_data = {}
            processed = 0
            
            for component_name, data in export_components:
                combined_data[component_name] = data
                processed += len(data)
                operation.processed_records = processed
                operation.progress = (processed / operation.total_records) * 100
            
            # Add export metadata
            combined_data['export_metadata'] = {
                'export_time': datetime.now(timezone.utc).isoformat(),
                'export_version': '1.0',
                'system_version': 'rag_system_v1',
                'components': list(combined_data.keys()),
                'total_records': operation.total_records
            }
            
            # Export data
            output_file = await self._export_data_to_file(combined_data, config, "system_export")
            
            operation.status = "completed"
            operation.completed_at = datetime.now(timezone.utc)
            operation.target = output_file
            
            logger.info(f"Completed system export: {operation.operation_id}")
            
        except Exception as e:
            operation.status = "failed"
            operation.error_message = str(e)
            operation.completed_at = datetime.now(timezone.utc)
            logger.error(f"System export failed: {operation.operation_id}: {e}")
            
        finally:
            # Move to history
            if operation.operation_id in self.active_operations:
                self.operation_history.append(operation)
                del self.active_operations[operation.operation_id]
    
    async def _export_data_to_file(
        self,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
        config: ExportConfig,
        prefix: str
    ) -> str:
        """Export data to file in specified format."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_filename = f"{prefix}_{timestamp}"
        
        # Process data based on format
        processor = self.format_processors.get(config.format)
        if not processor:
            raise ValueError(f"Unsupported format: {config.format}")
        
        output_file = await processor(data, base_filename, config, "export")
        
        # Apply compression if requested
        if config.compression != CompressionFormat.NONE:
            compressed_file = await self._compress_file(output_file, config.compression)
            Path(output_file).unlink()  # Remove uncompressed file
            return compressed_file
        
        return output_file
    
    async def _compress_file(self, file_path: str, compression: CompressionFormat) -> str:
        """Compress a file using specified compression format."""
        input_path = Path(file_path)
        
        if compression == CompressionFormat.GZIP:
            output_path = input_path.with_suffix(input_path.suffix + '.gz')
            with open(input_path, 'rb') as f_in:
                with gzip.open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        
        elif compression == CompressionFormat.ZIP:
            output_path = input_path.with_suffix('.zip')
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(input_path, input_path.name)
        
        elif compression == CompressionFormat.TAR_GZ:
            output_path = input_path.with_suffix('.tar.gz')
            with tarfile.open(output_path, 'w:gz') as tar:
                tar.add(input_path, arcname=input_path.name)
        
        else:
            return file_path
        
        return str(output_path)
    
    # Format processors
    async def _process_json(self, data: Any, filename: str, config: Any, mode: str) -> str:
        """Process JSON format."""
        output_file = self.export_dir / f"{filename}.json"
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(data, indent=2, default=str))
        
        return str(output_file)
    
    async def _process_csv(self, data: List[Dict[str, Any]], filename: str, config: Any, mode: str) -> str:
        """Process CSV format."""
        output_file = self.export_dir / f"{filename}.csv"
        
        if not data:
            return str(output_file)
        
        # Get all possible fields
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        
        fieldnames = sorted(list(all_fields))
        
        async with aiofiles.open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            await f.write(','.join(fieldnames) + '\n')  # Header
            
            for item in data:
                # Convert all values to strings and handle None
                row = {k: str(v) if v is not None else '' for k, v in item.items()}
                line = ','.join(f'"{row.get(field, "")}"' for field in fieldnames)
                await f.write(line + '\n')
        
        return str(output_file)
    
    async def _process_excel(self, data: List[Dict[str, Any]], filename: str, config: Any, mode: str) -> str:
        """Process Excel format."""
        output_file = self.export_dir / f"{filename}.xlsx"
        
        # Use pandas for Excel export
        df = pd.DataFrame(data)
        df.to_excel(output_file, index=False)
        
        return str(output_file)
    
    async def _process_parquet(self, data: List[Dict[str, Any]], filename: str, config: Any, mode: str) -> str:
        """Process Parquet format."""
        output_file = self.export_dir / f"{filename}.parquet"
        
        # Use pandas for Parquet export
        df = pd.DataFrame(data)
        df.to_parquet(output_file, index=False)
        
        return str(output_file)
    
    async def _process_sql(self, data: Any, filename: str, config: Any, mode: str) -> str:
        """Process SQL format."""
        output_file = self.export_dir / f"{filename}.sql"
        
        # Generate SQL INSERT statements
        sql_statements = []
        
        if isinstance(data, list) and data:
            # Assume it's a list of records for a single table
            table_name = filename.split('_')[0]  # Use prefix as table name
            
            for record in data:
                columns = list(record.keys())
                values = [f"'{v}'" if isinstance(v, str) else str(v) for v in record.values()]
                
                sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)});"
                sql_statements.append(sql)
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write('\n'.join(sql_statements))
        
        return str(output_file)
    
    async def _process_xml(self, data: Any, filename: str, config: Any, mode: str) -> str:
        """Process XML format."""
        output_file = self.export_dir / f"{filename}.xml"
        
        # Simple XML generation
        xml_content = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_content.append('<data>')
        
        if isinstance(data, list):
            for item in data:
                xml_content.append('  <item>')
                for key, value in item.items():
                    xml_content.append(f'    <{key}>{value}</{key}>')
                xml_content.append('  </item>')
        elif isinstance(data, dict):
            for key, value in data.items():
                xml_content.append(f'  <{key}>{value}</{key}>')
        
        xml_content.append('</data>')
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write('\n'.join(xml_content))
        
        return str(output_file)
    
    async def _process_yaml(self, data: Any, filename: str, config: Any, mode: str) -> str:
        """Process YAML format."""
        output_file = self.export_dir / f"{filename}.yaml"
        
        try:
            import yaml
            
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(yaml.dump(data, default_flow_style=False, default_scalar_style=''))
        
        except ImportError:
            # Fallback to JSON if YAML not available
            return await self._process_json(data, filename, config, mode)
        
        return str(output_file)
    
    # Helper methods for data retrieval
    async def _get_users_data(self, db: Session, config: ExportConfig) -> List[Dict[str, Any]]:
        """Get users data for export."""
        users = db.query(User).all()
        return [
            {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_active': user.is_active,
                'is_superuser': user.is_superuser,
                'created_at': user.created_at.isoformat(),
                'updated_at': user.updated_at.isoformat() if user.updated_at else None
            }
            for user in users
        ]
    
    async def _get_documents_data(
        self,
        db: Session,
        config: ExportConfig,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get documents data for export."""
        documents = db.query(Document).filter(Document.is_deleted == False).all()
        
        docs_data = []
        for doc in documents:
            doc_data = {
                'id': doc.id,
                'user_id': doc.user_id,
                'filename': doc.filename,
                'title': doc.title,
                'description': doc.description,
                'content_type': doc.content_type,
                'file_size': doc.file_size,
                'status': doc.status.value if doc.status else None,
                'is_public': doc.is_public,
                'created_at': doc.created_at.isoformat(),
                'updated_at': doc.updated_at.isoformat() if doc.updated_at else None
            }
            
            if config.include_metadata:
                doc_data['metadata'] = doc.get_metadata_dict()
                doc_data['tags'] = doc.get_tags()
            
            docs_data.append(doc_data)
        
        return docs_data
    
    async def _get_system_config_data(self) -> Dict[str, Any]:
        """Get system configuration data."""
        return {
            'system_name': 'RAG Document Processing System',
            'version': '1.0.0',
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'features': [
                'document_processing',
                'vector_search',
                'user_management',
                'api_access',
                'monitoring'
            ]
        }
    
    # Placeholder methods for import processes
    async def _import_users_process(self, operation: DataOperation, file_path: str, config: ImportConfig, db: Session):
        """Process user data import."""
        # Implement user import logic
        pass
    
    async def _import_documents_process(self, operation: DataOperation, file_path: str, config: ImportConfig, db: Session):
        """Process document data import."""
        # Implement document import logic
        pass
    
    async def _synchronize_data_process(self, operation: DataOperation, source_config: Dict[str, Any], target_config: Dict[str, Any], sync_direction: DataSyncDirection, db: Session):
        """Process data synchronization."""
        # Implement data synchronization logic
        pass
    
    async def _migrate_format_process(self, operation: DataOperation, source_file: str, source_format: DataFormat, target_format: DataFormat, target_file: str):
        """Process format migration."""
        # Implement format migration logic
        pass


# Global enhanced data manager instance
_enhanced_data_manager: Optional[EnhancedDataManager] = None

def get_enhanced_data_manager() -> Optional[EnhancedDataManager]:
    """Get global enhanced data manager instance."""
    global _enhanced_data_manager
    return _enhanced_data_manager

def initialize_enhanced_data_manager(
    export_dir: str = "data_exports",
    import_dir: str = "data_imports"
) -> EnhancedDataManager:
    """Initialize global enhanced data manager."""
    global _enhanced_data_manager
    
    _enhanced_data_manager = EnhancedDataManager(export_dir, import_dir)
    return _enhanced_data_manager