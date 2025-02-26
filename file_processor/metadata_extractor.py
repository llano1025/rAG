# metadata_extractor.py
from typing import Union, Optional
from pathlib import Path
import datetime
import hashlib
from dataclasses import dataclass
import fitz  # PyMuPDF
from docx import Document
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