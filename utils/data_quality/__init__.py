"""
Data Quality Module for RAG System
Provides duplicate detection and data quality assurance functionality.
"""

from .duplicate_detector import DuplicateDetector, DuplicateMatch, DuplicateType, create_duplicate_detector

__all__ = [
    'DuplicateDetector',
    'DuplicateMatch', 
    'DuplicateType',
    'create_duplicate_detector'
]