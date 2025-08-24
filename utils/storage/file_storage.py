"""
File storage manager for handling original document files.
Provides secure storage, retrieval, and management of uploaded files.
"""

import os
import hashlib
import shutil
import logging
from typing import Optional, BinaryIO, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class FileStorageError(Exception):
    """Raised when file storage operations fail."""
    pass

class FileStorageManager:
    """
    Manages storage and retrieval of original document files.
    
    Files are organized in a directory structure:
    storage_root/
    ├── documents/
    │   ├── 2025/
    │   │   ├── 01/
    │   │   │   ├── user_1/
    │   │   │   │   ├── doc_123_abc123.pdf
    │   │   │   │   ├── doc_124_def456.jpg
    """
    
    def __init__(self, storage_root: str = "data/documents"):
        """
        Initialize file storage manager.
        
        Args:
            storage_root: Root directory for file storage
        """
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileStorageManager initialized with storage root: {self.storage_root}")
    
    def _generate_file_path(
        self, 
        user_id: int, 
        document_id: int, 
        filename: str, 
        file_hash: str
    ) -> Path:
        """
        Generate a secure file path for storage.
        
        Args:
            user_id: User ID
            document_id: Document ID
            filename: Original filename
            file_hash: SHA-256 hash of file content
            
        Returns:
            Path object for file storage
        """
        # Use current date for organization
        now = datetime.now()
        year = now.strftime("%Y")
        month = now.strftime("%m")
        
        # Extract file extension
        file_ext = Path(filename).suffix.lower()
        
        # Create secure filename: doc_{id}_{hash_prefix}{extension}
        hash_prefix = file_hash[:8]  # First 8 chars of hash
        secure_filename = f"doc_{document_id}_{hash_prefix}{file_ext}"
        
        # Build path: storage_root/documents/YYYY/MM/user_ID/filename
        file_path = self.storage_root / "documents" / year / month / f"user_{user_id}" / secure_filename
        
        return file_path
    
    def store_file(
        self, 
        file_content: bytes, 
        user_id: int, 
        document_id: int, 
        filename: str, 
        file_hash: Optional[str] = None
    ) -> str:
        """
        Store a file and return its storage path.
        
        Args:
            file_content: File content as bytes
            user_id: User ID
            document_id: Document ID
            filename: Original filename
            file_hash: Optional pre-computed file hash
            
        Returns:
            String path where file is stored
            
        Raises:
            FileStorageError: If storage operation fails
        """
        try:
            # Calculate hash if not provided
            if not file_hash:
                file_hash = hashlib.sha256(file_content).hexdigest()
            
            # Generate storage path
            file_path = self._generate_file_path(user_id, document_id, filename, file_hash)
            
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Verify file was written correctly
            if not file_path.exists() or file_path.stat().st_size != len(file_content):
                raise FileStorageError(f"Failed to verify file storage: {file_path}")
            
            logger.info(f"File stored successfully: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to store file for document {document_id}: {e}")
            raise FileStorageError(f"File storage failed: {e}")
    
    def retrieve_file(self, file_path: str) -> bytes:
        """
        Retrieve file content from storage.
        
        Args:
            file_path: Path to stored file
            
        Returns:
            File content as bytes
            
        Raises:
            FileStorageError: If retrieval fails
        """
        try:
            path = Path(file_path)
            
            # Security check: ensure path is within storage root
            if not self._is_safe_path(path):
                raise FileStorageError(f"Unsafe file path: {file_path}")
            
            if not path.exists():
                raise FileStorageError(f"File not found: {file_path}")
            
            with open(path, 'rb') as f:
                content = f.read()
            
            logger.debug(f"File retrieved successfully: {file_path}")
            return content
            
        except FileStorageError:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_path}: {e}")
            raise FileStorageError(f"File retrieval failed: {e}")
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if deleted successfully, False if file didn't exist
            
        Raises:
            FileStorageError: If deletion fails
        """
        try:
            path = Path(file_path)
            
            # Security check: ensure path is within storage root
            if not self._is_safe_path(path):
                raise FileStorageError(f"Unsafe file path: {file_path}")
            
            if not path.exists():
                logger.warning(f"File not found for deletion: {file_path}")
                return False
            
            path.unlink()
            
            # Clean up empty directories
            self._cleanup_empty_directories(path.parent)
            
            logger.info(f"File deleted successfully: {file_path}")
            return True
            
        except FileStorageError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            raise FileStorageError(f"File deletion failed: {e}")
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in storage.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            path = Path(file_path)
            return self._is_safe_path(path) and path.exists()
        except Exception:
            return False
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get information about a stored file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            
            if not self._is_safe_path(path) or not path.exists():
                raise FileStorageError(f"File not found: {file_path}")
            
            stat = path.stat()
            
            return {
                'path': str(path),
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'exists': True
            }
            
        except FileStorageError:
            raise
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise FileStorageError(f"Failed to get file info: {e}")
    
    def _is_safe_path(self, path: Path) -> bool:
        """
        Check if a path is safe (within storage root).
        
        Args:
            path: Path to check
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Resolve both paths to absolute paths
            resolved_path = path.resolve()
            resolved_root = self.storage_root.resolve()
            
            # Check if file path is within storage root
            return str(resolved_path).startswith(str(resolved_root))
            
        except Exception:
            return False
    
    def _cleanup_empty_directories(self, directory: Path):
        """
        Remove empty directories up the tree.
        
        Args:
            directory: Directory to start cleanup from
        """
        try:
            # Only clean up within our storage root
            if not self._is_safe_path(directory):
                return
            
            # Don't remove the storage root itself
            if directory == self.storage_root:
                return
            
            # Check if directory is empty and remove it
            if directory.exists() and directory.is_dir():
                try:
                    # Try to remove if empty
                    directory.rmdir()
                    logger.debug(f"Removed empty directory: {directory}")
                    
                    # Recursively clean up parent directories
                    self._cleanup_empty_directories(directory.parent)
                    
                except OSError:
                    # Directory not empty, stop cleanup
                    pass
                    
        except Exception as e:
            logger.debug(f"Error during directory cleanup: {e}")

# Global instance
_file_storage_manager = None

def get_file_storage_manager() -> FileStorageManager:
    """Get the global file storage manager instance."""
    global _file_storage_manager
    if _file_storage_manager is None:
        _file_storage_manager = FileStorageManager()
    return _file_storage_manager