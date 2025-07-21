# metadata_extractor.py
from typing import Union, Optional
from pathlib import Path
import datetime
import hashlib
from dataclasses import dataclass
import fitz  # PyMuPDF
from docx import Document
import tempfile
import os
from .type_detector import FileTypeDetector

@dataclass
class DocumentMetadata:
    """Data class for document metadata."""
    file_name: str
    file_type: str
    file_size: int
    creation_date: datetime.datetime
    modification_date: datetime.datetime
    author: Optional[str] = None
    title: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    checksum: Optional[str] = None

class MetadataExtractor:
    """Extracts metadata from documents."""

    def extract(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """Extract metadata from a document."""
        path = Path(file_path)
        stats = path.stat()
        
        metadata = DocumentMetadata(
            file_name=path.name,
            file_type=FileTypeDetector().detect_type(file_path),
            file_size=stats.st_size,
            creation_date=datetime.datetime.fromtimestamp(stats.st_ctime),
            modification_date=datetime.datetime.fromtimestamp(stats.st_mtime)
        )

        # Calculate checksum
        metadata.checksum = self._calculate_checksum(file_path)
        
        # Extract format-specific metadata
        self._extract_specific_metadata(file_path, metadata)
        
        return metadata

    async def extract_metadata(self, file_content: bytes, content_type: str, filename: str = None) -> DocumentMetadata:
        """Extract metadata from document content bytes (async method expected by controllers)."""
        # Create temporary file from bytes content
        with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(content_type)) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_path = temp_file.name
        
        try:
            # Extract metadata using existing method
            metadata = self.extract(temp_path)
            
            # Override filename if provided
            if filename:
                metadata.file_name = filename
            
            # Override file size with actual content size
            metadata.file_size = len(file_content)
            metadata.file_type = content_type
            
            # Recalculate checksum from content
            metadata.checksum = hashlib.sha256(file_content).hexdigest()
            
            return metadata
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _get_file_extension(self, content_type: str) -> str:
        """Get appropriate file extension for content type."""
        extensions = {
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'text/plain': '.txt',
            'text/html': '.html',
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/tiff': '.tiff'
        }
        return extensions.get(content_type, '.tmp')

    def _calculate_checksum(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _extract_specific_metadata(self, file_path: Union[str, Path], metadata: DocumentMetadata):
        """Extract format-specific metadata."""
        if metadata.file_type == 'application/pdf':
            with fitz.open(str(file_path)) as doc:
                metadata.page_count = len(doc)
                metadata.title = doc.metadata.get('title')
                metadata.author = doc.metadata.get('author')
        elif metadata.file_type.startswith('application/vnd.openxmlformats-officedocument'):
            doc = Document(str(file_path))
            metadata.page_count = len(doc.sections)
            core_properties = doc.core_properties
            metadata.title = core_properties.title
            metadata.author = core_properties.author