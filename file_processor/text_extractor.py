# text_extractor.py
from typing import Union, Optional
from pathlib import Path
import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup
import logging
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

    def _extract_from_image(self, file_path: Union[str, Path]) -> str:
        """Extract text from images using OCR."""
        return OCRProcessor().process_image(file_path)
