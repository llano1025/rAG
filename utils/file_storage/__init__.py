"""
File storage utilities for document management.
"""

from .file_manager import FileStorageManager, get_file_manager, init_file_manager

__all__ = [
    'FileStorageManager',
    'get_file_manager', 
    'init_file_manager'
]