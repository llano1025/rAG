# text_extractor.py
from typing import Union, Optional
from pathlib import Path
import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup
import logging
import tempfile
import os
from io import BytesIO
from .type_detector import FileTypeDetector, UnsupportedFileType
from .ocr_processor import OCRProcessor

class TextExtractor:
    """Extracts text content from various document formats."""
    
    def __init__(self):
        self.type_detector = FileTypeDetector()

    def extract(self, file_path: Union[str, Path], mime_type: Optional[str] = None) -> str:
        """Extract text from a document based on its type."""
        if mime_type is None:
            mime_type = self.type_detector.detect_type(file_path)

        extractors = {
            'application/pdf': self._extract_from_pdf,
            'application/msword': self._extract_from_doc,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._extract_from_docx,
            'text/plain': self._extract_from_text,
            'text/html': self._extract_from_html,
            'image/jpeg': self._extract_from_image,
            'image/png': self._extract_from_image,
            'image/tiff': self._extract_from_image
        }

        extractor = extractors.get(mime_type)
        if not extractor:
            raise UnsupportedFileType(f"No text extractor for MIME type: {mime_type}")

        return extractor(file_path)

    async def extract_text(self, file_content: bytes, content_type: str, filename: str = None) -> str:
        """Extract text from document content bytes (async method expected by controllers)."""
        # Create temporary file from bytes content
        with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(content_type)) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_path = temp_file.name
        
        try:
            # Use existing extract method with temporary file
            return self.extract(temp_path, content_type)
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

    def _extract_from_pdf(self, file_path: Union[str, Path]) -> str:
        """Extract text from PDF files."""
        text = []
        with fitz.open(str(file_path)) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)

    def _extract_from_docx(self, file_path: Union[str, Path]) -> str:
        """Extract text from DOCX files."""
        doc = Document(str(file_path))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    def _extract_from_text(self, file_path: Union[str, Path]) -> str:
        """Extract text from plain text files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _extract_from_html(self, file_path: Union[str, Path]) -> str:
        """Extract text from HTML files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text(separator='\n', strip=True)

    def _extract_from_doc(self, file_path: Union[str, Path]) -> str:
        """Extract text from DOC files (legacy Word format)."""
        # For now, treat as unsupported - would need python-docx2txt or similar
        # This is a placeholder implementation
        try:
            # Attempt to read as text (limited support)
            with open(file_path, 'rb') as f:
                content = f.read()
                # Basic text extraction - not ideal but functional
                text = content.decode('utf-8', errors='ignore')
                # Clean up binary data
                return ''.join(c for c in text if c.isprintable() or c.isspace())
        except Exception as e:
            logging.warning(f"Failed to extract text from DOC file {file_path}: {e}")
            return "[Could not extract text from DOC file - consider converting to DOCX]"

    def _extract_from_image(self, file_path: Union[str, Path]) -> str:
        """Extract text from images using OCR."""
        ocr_processor = OCRProcessor()
        return ocr_processor.process_image(file_path)
