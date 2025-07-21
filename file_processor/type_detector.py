from typing import Union, BinaryIO, Dict, Set
from pathlib import Path
import magic
import logging
from enum import Enum

class FileType(Enum):
    """Enumeration of supported file types."""
    PDF = "application/pdf"
    DOC = "application/msword"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    TEXT = "text/plain"
    HTML = "text/html"
    JPEG = "image/jpeg"
    PNG = "image/png"
    TIFF = "image/tiff"

class UnsupportedFileType(Exception):
    """Raised when file type is not supported."""
    pass

class FileTypeDetector:
    """Detects and validates file types using magic numbers and extensions."""
    
    MIME_TO_EXTENSION: Dict[str, str] = {
        FileType.PDF.value: '.pdf',
        FileType.DOC.value: '.doc',
        FileType.DOCX.value: '.docx',
        FileType.TEXT.value: '.txt',
        FileType.HTML.value: '.html',
        FileType.JPEG.value: '.jpg',
        FileType.PNG.value: '.png',
        FileType.TIFF.value: '.tiff'
    }

    IMAGE_MIME_TYPES: Set[str] = {
        FileType.JPEG.value,
        FileType.PNG.value,
        FileType.TIFF.value
    }

    DOCUMENT_MIME_TYPES: Set[str] = {
        FileType.PDF.value,
        FileType.DOC.value,
        FileType.DOCX.value,
        FileType.TEXT.value,
        FileType.HTML.value
    }

    def __init__(self):
        """Initialize the FileTypeDetector with magic for MIME type detection."""
        self.mime = magic.Magic(mime=True)
        self.logger = logging.getLogger(__name__)

    def detect(self, file_path: Union[str, Path, BinaryIO]) -> str:
        """
        Detect MIME type of a file.

        Args:
            file_path: Path to the file or file-like object to analyze

        Returns:
            str: Detected MIME type

        Raises:
            UnsupportedFileType: If the file type is not supported
            IOError: If there are issues reading the file
        """
        try:
            if isinstance(file_path, (str, Path)):
                mime_type = self.mime.from_file(str(file_path))
            else:
                content = file_path.read(2048)
                mime_type = self.mime.from_buffer(content)
                file_path.seek(0)  # Reset file pointer

            self.logger.debug(f"Detected MIME type: {mime_type}")
            
            if mime_type not in self.MIME_TO_EXTENSION:
                raise UnsupportedFileType(f"Unsupported file type: {mime_type}")
            
            return mime_type

        except (IOError, OSError) as e:
            self.logger.error(f"Error reading file: {e}")
            raise IOError(f"Failed to read file: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during type detection: {e}")
            raise

    def validate_extension(self, file_path: Union[str, Path], mime_type: str) -> bool:
        """
        Validate if file extension matches detected MIME type.

        Args:
            file_path: Path to the file
            mime_type: MIME type to validate against

        Returns:
            bool: True if extension matches MIME type, False otherwise
        """
        extension = Path(str(file_path)).suffix.lower()
        expected_extension = self.MIME_TO_EXTENSION.get(mime_type)
        
        is_valid = extension == expected_extension
        self.logger.debug(f"Extension validation: {extension} vs {expected_extension} -> {is_valid}")
        
        return is_valid

    def is_image(self, mime_type: str) -> bool:
        """
        Check if the given MIME type represents an image format.

        Args:
            mime_type: MIME type to check

        Returns:
            bool: True if the MIME type represents an image, False otherwise
        """
        return mime_type in self.IMAGE_MIME_TYPES

    def is_document(self, mime_type: str) -> bool:
        """
        Check if the given MIME type represents a document format.

        Args:
            mime_type: MIME type to check

        Returns:
            bool: True if the MIME type represents a document, False otherwise
        """
        return mime_type in self.DOCUMENT_MIME_TYPES

    def get_extension(self, mime_type: str) -> str:
        """
        Get the expected file extension for a MIME type.

        Args:
            mime_type: MIME type to get extension for

        Returns:
            str: The corresponding file extension

        Raises:
            UnsupportedFileType: If the MIME type is not supported
        """
        if mime_type not in self.MIME_TO_EXTENSION:
            raise UnsupportedFileType(f"No extension mapping for MIME type: {mime_type}")
        
        return self.MIME_TO_EXTENSION[mime_type]

    def detect_type(self, file_path: Union[str, Path] = None, file_content: bytes = None, filename: str = None) -> str:
        """Detect MIME type from file path or content bytes (method expected by controllers).
        
        Args:
            file_path: Path to file (for backward compatibility)
            file_content: File content as bytes
            filename: Original filename (for extension validation)
            
        Returns:
            str: Detected MIME type
            
        Raises:
            UnsupportedFileType: If the file type is not supported
            ValueError: If neither file_path nor file_content is provided
        """
        if file_path is not None:
            # Use existing detect method for file path
            return self.detect(file_path)
        elif file_content is not None:
            # Detect from bytes content
            try:
                mime_type = self.mime.from_buffer(file_content)
                self.logger.debug(f"Detected MIME type from content: {mime_type}")
                
                if mime_type not in self.MIME_TO_EXTENSION:
                    raise UnsupportedFileType(f"Unsupported file type: {mime_type}")
                
                # Validate against filename extension if provided
                if filename and not self._validate_extension_against_filename(filename, mime_type):
                    self.logger.warning(f"Extension mismatch for {filename}: detected {mime_type}")
                
                return mime_type
                
            except Exception as e:
                self.logger.error(f"Error detecting type from content: {e}")
                raise UnsupportedFileType(f"Failed to detect file type: {e}")
        else:
            raise ValueError("Either file_path or file_content must be provided")
    
    def _validate_extension_against_filename(self, filename: str, mime_type: str) -> bool:
        """Validate filename extension against MIME type.
        
        Args:
            filename: The filename to validate
            mime_type: The detected MIME type
            
        Returns:
            bool: True if extension matches MIME type
        """
        extension = Path(filename).suffix.lower()
        expected_extension = self.MIME_TO_EXTENSION.get(mime_type)
        
        # Handle common extension variations
        extension_mappings = {
            '.jpeg': '.jpg',
            '.htm': '.html',
            '.tif': '.tiff'
        }
        
        normalized_extension = extension_mappings.get(extension, extension)
        is_valid = normalized_extension == expected_extension
        
        self.logger.debug(f"Extension validation: {extension} -> {normalized_extension} vs {expected_extension} -> {is_valid}")
        
        return is_valid