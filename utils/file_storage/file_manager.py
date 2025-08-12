"""
File storage manager for handling original document files.
Provides secure file storage, retrieval, and management functionality.
"""

import os
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Optional, BinaryIO
from datetime import datetime

logger = logging.getLogger(__name__)

class FileStorageManager:
    """Manages storage and retrieval of original document files."""
    
    def __init__(self, storage_root: str = "storage/documents"):
        """
        Initialize file storage manager.
        
        Args:
            storage_root: Root directory for file storage
        """
        logger.info(f"🚀 FileManager: Initializing file storage manager...")
        logger.info(f"📁 FileManager: Storage root path: {storage_root}")
        
        self.storage_root = Path(storage_root)
        logger.info(f"📂 FileManager: Resolved storage root: {self.storage_root.absolute()}")
        
        # Create storage directory if it doesn't exist
        try:
            logger.info(f"🔨 FileManager: Creating storage directory structure...")
            self.storage_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ FileManager: Storage directory created/verified")
        except OSError as create_err:
            logger.error(f"❌ FileManager: Failed to create storage directory: {create_err}")
            raise IOError(f"Cannot create storage directory: {create_err}")
        
        # Validate storage directory
        self._validate_storage_directory()
        
        logger.info(f"✅ FileManager: File storage initialized successfully at: {self.storage_root.absolute()}")
    
    def _validate_storage_directory(self):
        """Validate storage directory exists and is accessible."""
        logger.info(f"🔍 FileManager: Validating storage directory...")
        
        # Check if directory exists
        if not self.storage_root.exists():
            logger.error(f"❌ FileManager: Storage directory does not exist: {self.storage_root}")
            raise IOError(f"Storage directory does not exist: {self.storage_root}")
            
        # Check if it's actually a directory
        if not self.storage_root.is_dir():
            logger.error(f"❌ FileManager: Storage path is not a directory: {self.storage_root}")
            raise IOError(f"Storage path is not a directory: {self.storage_root}")
            
        # Check read permissions
        if not os.access(self.storage_root, os.R_OK):
            logger.error(f"🔒 FileManager: Storage directory is not readable: {self.storage_root}")
            raise IOError(f"Storage directory is not readable: {self.storage_root}")
            
        # Check write permissions
        if not os.access(self.storage_root, os.W_OK):
            logger.error(f"🔒 FileManager: Storage directory is not writable: {self.storage_root}")
            raise IOError(f"Storage directory is not writable: {self.storage_root}")
            
        # Check execute permissions (needed to access subdirectories)
        if not os.access(self.storage_root, os.X_OK):
            logger.error(f"🔒 FileManager: Storage directory is not executable: {self.storage_root}")
            raise IOError(f"Storage directory is not executable: {self.storage_root}")
            
        # Test write capability with a temporary file
        try:
            test_file = self.storage_root / ".test_write"
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()  # Delete test file
            logger.info(f"✅ FileManager: Write test passed")
        except OSError as test_err:
            logger.error(f"❌ FileManager: Write test failed: {test_err}")
            raise IOError(f"Cannot write to storage directory: {test_err}")
            
        logger.info(f"✅ FileManager: Storage directory validation passed")
    
    def _generate_file_path(self, file_hash: str, original_filename: str) -> Path:
        """
        Generate secure file path based on hash and filename.
        
        Args:
            file_hash: SHA-256 hash of file content
            original_filename: Original filename with extension
            
        Returns:
            Path object for file storage
        """
        # Use first 2 chars of hash for directory structure (avoids too many files in one dir)
        hash_dir = file_hash[:2]
        hash_subdir = file_hash[2:4]
        
        # Keep original extension for proper MIME type detection
        file_ext = Path(original_filename).suffix.lower()
        filename = f"{file_hash}{file_ext}"
        
        return self.storage_root / hash_dir / hash_subdir / filename
    
    def save_file(self, file_content: bytes, original_filename: str, file_hash: Optional[str] = None) -> str:
        """
        Save file content to storage.
        
        Args:
            file_content: Binary file content
            original_filename: Original filename with extension
            file_hash: Optional pre-calculated hash
            
        Returns:
            Relative file path for database storage
            
        Raises:
            IOError: If file cannot be saved
        """
        logger.info(f"💾 FileManager: Starting file save - Filename: {original_filename}")
        logger.info(f"📊 FileManager: File size: {len(file_content)} bytes")
        
        try:
            # Calculate hash if not provided
            if not file_hash:
                logger.info(f"🔐 FileManager: Calculating SHA-256 hash...")
                file_hash = hashlib.sha256(file_content).hexdigest()
                logger.info(f"🔐 FileManager: Hash calculated: {file_hash[:16]}...")
            
            # Generate storage path
            logger.info(f"📁 FileManager: Generating storage path...")
            file_path = self._generate_file_path(file_hash, original_filename)
            logger.info(f"📂 FileManager: Generated path: {file_path}")
            logger.info(f"🗂️ FileManager: Storage root: {self.storage_root}")
            
            # Check storage root exists and is writable
            if not self.storage_root.exists():
                logger.error(f"❌ FileManager: Storage root does not exist: {self.storage_root}")
                raise IOError(f"Storage root directory does not exist: {self.storage_root}")
            
            if not os.access(self.storage_root, os.W_OK):
                logger.error(f"🔒 FileManager: Storage root is not writable: {self.storage_root}")
                raise IOError(f"Storage root directory is not writable: {self.storage_root}")
            
            # Create directory structure
            logger.info(f"📁 FileManager: Creating directory structure...")
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ FileManager: Directory structure created: {file_path.parent}")
            except OSError as dir_err:
                logger.error(f"❌ FileManager: Failed to create directory structure: {dir_err}")
                raise IOError(f"Cannot create directory structure: {dir_err}")
            
            # Check if file already exists (deduplication)
            if file_path.exists():
                logger.info(f"🔄 FileManager: File already exists, verifying integrity...")
                try:
                    existing_size = file_path.stat().st_size
                    if existing_size == len(file_content):
                        logger.info(f"✅ FileManager: Existing file verified (size: {existing_size}), skipping save")
                        return str(file_path.relative_to(self.storage_root))
                    else:
                        logger.warning(f"⚠️ FileManager: Size mismatch - existing: {existing_size}, new: {len(file_content)}")
                except OSError as stat_err:
                    logger.warning(f"⚠️ FileManager: Cannot stat existing file: {stat_err}, overwriting...")
            
            # Check available disk space (rough estimate)
            try:
                stat_result = os.statvfs(file_path.parent)
                available_space = stat_result.f_bavail * stat_result.f_frsize
                logger.info(f"💽 FileManager: Available disk space: {available_space / (1024*1024):.1f} MB")
                
                if available_space < len(file_content) * 2:  # Need at least 2x file size as buffer
                    logger.error(f"❌ FileManager: Insufficient disk space")
                    raise IOError(f"Insufficient disk space: need {len(file_content)}, available {available_space}")
                    
            except (OSError, AttributeError) as space_err:
                logger.warning(f"⚠️ FileManager: Cannot check disk space: {space_err}")
            
            # Save file
            logger.info(f"💾 FileManager: Writing file to disk...")
            try:
                with open(file_path, 'wb') as f:
                    bytes_written = f.write(file_content)
                    logger.info(f"✏️ FileManager: Bytes written: {bytes_written}")
                    
                    if bytes_written != len(file_content):
                        raise IOError(f"Incomplete write: {bytes_written} != {len(file_content)}")
                        
            except OSError as write_err:
                logger.error(f"❌ FileManager: File write failed: {write_err}")
                raise IOError(f"Cannot write file: {write_err}")
            
            # Verify file was saved correctly
            logger.info(f"🔍 FileManager: Verifying saved file...")
            try:
                if not file_path.exists():
                    logger.error(f"❌ FileManager: File does not exist after save")
                    raise IOError(f"File verification failed: file does not exist after save")
                
                saved_size = file_path.stat().st_size
                logger.info(f"📏 FileManager: Saved file size: {saved_size} bytes")
                
                if saved_size != len(file_content):
                    logger.error(f"❌ FileManager: Size mismatch - expected: {len(file_content)}, actual: {saved_size}")
                    raise IOError(f"File verification failed: size mismatch ({saved_size} != {len(file_content)})")
                    
            except OSError as verify_err:
                logger.error(f"❌ FileManager: File verification error: {verify_err}")
                raise IOError(f"File verification failed: {verify_err}")
            
            relative_path = str(file_path.relative_to(self.storage_root))
            logger.info(f"✅ FileManager: File saved successfully!")
            logger.info(f"📋 FileManager: Relative path: {relative_path}")
            return relative_path
            
        except Exception as e:
            logger.error(f"Failed to save file {original_filename}: {e}")
            raise IOError(f"Could not save file: {e}")
    
    def get_file(self, file_path: str) -> Optional[bytes]:
        """
        Retrieve file content from storage.
        
        Args:
            file_path: Relative file path from database
            
        Returns:
            File content as bytes, or None if not found
        """
        try:
            full_path = self.storage_root / file_path
            
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                return None
            
            # Security check: ensure path is within storage root
            if not self._is_safe_path(full_path):
                logger.error(f"Unsafe file path access attempt: {full_path}")
                return None
            
            with open(full_path, 'rb') as f:
                content = f.read()
            
            logger.debug(f"File retrieved successfully: {full_path}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_path}: {e}")
            return None
    
    def get_file_stream(self, file_path: str) -> Optional[BinaryIO]:
        """
        Get file as stream for large files.
        
        Args:
            file_path: Relative file path from database
            
        Returns:
            File stream or None if not found
        """
        logger.info(f"🔓 FileManager: Opening file stream for: {file_path}")
        
        try:
            full_path = self.storage_root / file_path
            logger.info(f"📂 FileManager: Full path resolved to: {full_path}")
            logger.info(f"🗂️ FileManager: Storage root: {self.storage_root}")
            
            # Check if file exists
            exists = full_path.exists()
            logger.info(f"🔍 FileManager: File exists check: {exists}")
            
            if not exists:
                logger.warning(f"❌ FileManager: File not found at: {full_path}")
                return None
            
            # Check file size
            try:
                file_size = full_path.stat().st_size
                logger.info(f"📏 FileManager: File size: {file_size} bytes")
            except Exception as size_err:
                logger.warning(f"⚠️ FileManager: Could not get file size: {size_err}")
            
            # Security check
            if not self._is_safe_path(full_path):
                logger.error(f"🚨 FileManager: Unsafe file path access attempt: {full_path}")
                return None
            
            logger.info(f"🔓 FileManager: Opening file in binary read mode: {full_path}")
            file_stream = open(full_path, 'rb')
            logger.info(f"✅ FileManager: File stream opened successfully")
            
            return file_stream
            
        except PermissionError as pe:
            logger.error(f"🔒 FileManager: Permission denied accessing file {file_path}: {pe}")
            return None
        except FileNotFoundError as fnf:
            logger.error(f"❌ FileManager: File not found {file_path}: {fnf}")
            return None
        except Exception as e:
            logger.error(f"💥 FileManager: Unexpected error opening file stream {file_path}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.exception("Full exception traceback:")
            return None
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists in storage.
        
        Args:
            file_path: Relative file path from database
            
        Returns:
            True if file exists, False otherwise
        """
        logger.info(f"🔍 FileManager: Checking if file exists: {file_path}")
        
        try:
            full_path = self.storage_root / file_path
            logger.info(f"📂 FileManager: Full path for existence check: {full_path}")
            
            exists = full_path.exists()
            logger.info(f"📋 FileManager: File existence result: {exists}")
            
            if exists:
                # Additional checks for file readability
                if full_path.is_file():
                    logger.info(f"✅ FileManager: Confirmed as regular file")
                    try:
                        size = full_path.stat().st_size
                        logger.info(f"📏 FileManager: File size: {size} bytes")
                    except Exception as stat_err:
                        logger.warning(f"⚠️ FileManager: Could not stat file: {stat_err}")
                else:
                    logger.warning(f"⚠️ FileManager: Path exists but is not a regular file")
                    return False
                
                # Check if path is safe
                is_safe = self._is_safe_path(full_path)
                logger.info(f"🔐 FileManager: Path safety check: {is_safe}")
                
                return is_safe
            
            return False
            
        except Exception as e:
            logger.error(f"💥 FileManager: Error checking file existence {file_path}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            return False
    
    def get_file_size(self, file_path: str) -> Optional[int]:
        """
        Get file size in bytes.
        
        Args:
            file_path: Relative file path from database
            
        Returns:
            File size in bytes, or None if file not found
        """
        try:
            full_path = self.storage_root / file_path
            
            if not full_path.exists() or not self._is_safe_path(full_path):
                return None
                
            return full_path.stat().st_size
            
        except Exception as e:
            logger.error(f"Failed to get file size {file_path}: {e}")
            return None
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete file from storage.
        
        Args:
            file_path: Relative file path from database
            
        Returns:
            True if file was deleted, False otherwise
        """
        try:
            full_path = self.storage_root / file_path
            
            if not full_path.exists():
                logger.info(f"File already deleted or doesn't exist: {full_path}")
                return True
            
            # Security check
            if not self._is_safe_path(full_path):
                logger.error(f"Unsafe file deletion attempt: {full_path}")
                return False
            
            full_path.unlink()
            
            # Clean up empty directories
            try:
                full_path.parent.rmdir()  # Only removes if empty
                full_path.parent.parent.rmdir()  # Only removes if empty
            except OSError:
                pass  # Directory not empty, which is fine
            
            logger.info(f"File deleted successfully: {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    def _is_safe_path(self, file_path: Path) -> bool:
        """
        Check if file path is within storage root (security check).
        
        Args:
            file_path: Path to check
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Resolve paths to handle symlinks and relative paths
            resolved_path = file_path.resolve()
            resolved_root = self.storage_root.resolve()
            
            # Check if the resolved path is within the storage root
            return resolved_path.is_relative_to(resolved_root)
            
        except Exception:
            return False
    
    def get_storage_stats(self) -> dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            total_files = 0
            total_size = 0
            
            for file_path in self.storage_root.rglob('*'):
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
            
            return {
                'total_files': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'storage_root': str(self.storage_root.absolute())
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {
                'total_files': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'storage_root': str(self.storage_root.absolute())
            }


# Global file storage manager instance
_file_manager = None

def get_file_manager() -> FileStorageManager:
    """Get global file storage manager instance."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileStorageManager()
    return _file_manager

def init_file_manager(storage_root: str = "storage/documents") -> FileStorageManager:
    """Initialize file storage manager with custom settings."""
    global _file_manager
    _file_manager = FileStorageManager(storage_root)
    return _file_manager