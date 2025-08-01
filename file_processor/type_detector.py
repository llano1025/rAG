from typing import Union, BinaryIO, Dict, Set
from pathlib import Path
import logging
from enum import Enum

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    logging.warning("python-magic not available, file type detection will use basic methods")
    magic = None
    MAGIC_AVAILABLE = False

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
    GIF = "image/gif"

class UnsupportedFileType(Exception):
    """Raised when file type is not supported."""
    pass

class FileTypeDetector:
    """Detects and validates file types using magic numbers and extensions."""
    
    MIME_TO_EXTENSION: Dict[str, str] = {
        'application/pdf': '.pdf',
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'text/plain': '.txt',
        'text/html': '.html',
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/tiff': '.tiff',
        'image/gif': '.gif',
        # MIME type variations (before normalization)
        'image/jpg': '.jpg',
        'image/tif': '.tiff',
        'text/htm': '.html',
        # Additional MIME type variations
        'image/pjpeg': '.jpg',  # Progressive JPEG
        'image/x-png': '.png',  # Alternative PNG
        # Additional common JPEG variations
        'image/jp2': '.jpg',    # JPEG 2000
        'image/jpx': '.jpg',    # JPEG 2000 extended
        'image/jpm': '.jpg',    # JPEG 2000 compound
        # Common variations that might be detected by different systems
        'image/jpeg2000': '.jpg',
        'image/jpeg2000-image': '.jpg',
        # Text file variations
        'text/markdown': '.md',
        'text/csv': '.csv',
        'application/json': '.json',
        # Additional document types
        'text/x-markdown': '.md',
        'application/x-json': '.json',
        # Additional image formats and variations
        'image/webp': '.webp',
        'image/bmp': '.bmp',
        'image/x-ms-bmp': '.bmp',
        'image/svg+xml': '.svg',
        'image/x-icon': '.ico',
        'image/vnd.microsoft.icon': '.ico',
        # Additional text formats
        'text/rtf': '.rtf',
        'application/rtf': '.rtf',
        'text/x-python': '.py',
        'application/x-python-code': '.py',
        'text/x-java-source': '.java',
        'text/javascript': '.js',
        'application/javascript': '.js',
        'text/css': '.css',
        'application/xml': '.xml',
        'text/xml': '.xml',
    }

    IMAGE_MIME_TYPES: Set[str] = {
        'image/jpeg',
        'image/png',
        'image/tiff',
        'image/gif'
    }

    DOCUMENT_MIME_TYPES: Set[str] = {
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'text/html'
    }

    # MIME type normalization mapping
    MIME_TYPE_NORMALIZATIONS: Dict[str, str] = {
        # JPEG normalizations - all variants should map to standard image/jpeg
        'image/jpg': 'image/jpeg',
        'image/pjpeg': 'image/jpeg',  # Progressive JPEG
        'image/jp2': 'image/jpeg',    # JPEG 2000
        'image/jpx': 'image/jpeg',    # JPEG 2000 extended
        'image/jpm': 'image/jpeg',    # JPEG 2000 compound
        'image/jpeg2000': 'image/jpeg',
        'image/jpeg2000-image': 'image/jpeg',
        # TIFF normalizations
        'image/tif': 'image/tiff',
        # PNG normalizations
        'image/x-png': 'image/png',   # Alternative PNG
        # HTML normalizations
        'text/htm': 'text/html',
        # Text file normalizations
        'text/x-markdown': 'text/markdown',
        'application/x-json': 'application/json',
        # BMP normalizations
        'image/x-ms-bmp': 'image/bmp',
        # ICO normalizations
        'image/vnd.microsoft.icon': 'image/x-icon',
        # RTF normalizations
        'application/rtf': 'text/rtf',
        # Code file normalizations
        'application/x-python-code': 'text/x-python',
        'application/javascript': 'text/javascript',
        # XML normalizations
        'text/xml': 'application/xml',
    }

    def __init__(self):
        """Initialize the FileTypeDetector with magic for MIME type detection."""
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TypeDetector: Initializing FileTypeDetector...")
        self.logger.info(f"TypeDetector: MAGIC_AVAILABLE = {MAGIC_AVAILABLE}")
        
        magic_available = MAGIC_AVAILABLE  # Use local variable to avoid scoping issues
        magic_working = False
        
        if magic_available:
            try:
                self.mime = magic.Magic(mime=True)
                self.logger.info(f"TypeDetector: python-magic initialized successfully")
                magic_working = True
                
                # Test magic functionality with different content types
                test_cases = [
                    (b"test", "text/plain test"),
                    (b"\x89PNG\r\n\x1a\n", "PNG header test"),
                    (b"\xff\xd8\xff", "JPEG header test")
                ]
                
                for test_data, description in test_cases:
                    try:
                        test_result = self.mime.from_buffer(test_data)
                        self.logger.info(f"TypeDetector: Magic {description}: {test_result}")
                    except Exception as te:
                        self.logger.warning(f"TypeDetector: Magic {description} failed: {te}")
                        magic_working = False
                        
            except Exception as e:
                self.logger.error(f"TypeDetector: Failed to initialize python-magic: {e}")
                self.logger.exception(f"TypeDetector: Full python-magic initialization error:")
                self.mime = None
                magic_working = False
        else:
            self.mime = None
            self.logger.warning(f"TypeDetector: python-magic not available, using filename-based detection")
            
        # Log the final status
        self.logger.info(f"TypeDetector: Final configuration - Global MAGIC_AVAILABLE: {MAGIC_AVAILABLE}, Magic working: {magic_working}, mime instance: {self.mime is not None}")
        self.logger.info(f"TypeDetector: Available MIME mappings: {len(self.MIME_TO_EXTENSION)} types")
        self.logger.info(f"TypeDetector: Supported extensions: {list(self.MIME_TO_EXTENSION.values())}")
        self.logger.info(f"TypeDetector: MIME_TO_EXTENSION contents: {self.MIME_TO_EXTENSION}")
        self.logger.info(f"TypeDetector: 'image/jpeg' in MIME_TO_EXTENSION: {'image/jpeg' in self.MIME_TO_EXTENSION}")
    
    def _normalize_mime_type(self, mime_type: str) -> str:
        """Normalize MIME type to handle common variations."""
        return self.MIME_TYPE_NORMALIZATIONS.get(mime_type, mime_type)

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
            if not MAGIC_AVAILABLE:
                self.logger.info("python-magic not available, using extension-based detection")
                # Fallback to extension-based detection
                return self._detect_by_extension(file_path)
            
            if isinstance(file_path, (str, Path)):
                mime_type = self.mime.from_file(str(file_path))
            else:
                content = file_path.read(2048)
                mime_type = self.mime.from_buffer(content)
                file_path.seek(0)  # Reset file pointer

            # Normalize MIME type to handle variations
            normalized_mime_type = self._normalize_mime_type(mime_type)
            self.logger.info(f"MIME type detection - Original: {mime_type}, Normalized: {normalized_mime_type}")
            self.logger.info(f"MIME type detection - Available mappings: {list(self.MIME_TO_EXTENSION.keys())}")
            self.logger.info(f"MIME type detection - Normalized type in mappings: {normalized_mime_type in self.MIME_TO_EXTENSION}")
            
            if normalized_mime_type not in self.MIME_TO_EXTENSION:
                self.logger.error(f"MIME type validation failed - {normalized_mime_type} not in supported types")
                raise UnsupportedFileType(f"Unsupported file type: {normalized_mime_type} (original: {mime_type})")
            
            return normalized_mime_type

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
                if not MAGIC_AVAILABLE:
                    self.logger.info("python-magic not available for content detection, trying filename extension")
                    # Use filename extension if available
                    if filename:
                        return self._detect_by_filename(filename)
                    else:
                        self.logger.error("Cannot detect file type: python-magic unavailable and no filename provided")
                        raise UnsupportedFileType("Cannot detect file type without python-magic library and no filename provided")
                
                mime_type = self.mime.from_buffer(file_content)
                # Normalize MIME type
                normalized_mime_type = self._normalize_mime_type(mime_type)
                self.logger.info(f"MIME type detection from content - Original: {mime_type}, Normalized: {normalized_mime_type}")
                self.logger.info(f"MIME type detection from content - Available mappings: {list(self.MIME_TO_EXTENSION.keys())}")
                self.logger.info(f"MIME type detection from content - Normalized type in mappings: {normalized_mime_type in self.MIME_TO_EXTENSION}")
                
                if normalized_mime_type not in self.MIME_TO_EXTENSION:
                    self.logger.error(f"MIME type validation failed from content - {normalized_mime_type} not in supported types")
                    self.logger.error(f"Current MIME_TO_EXTENSION keys: {list(self.MIME_TO_EXTENSION.keys())}")
                    self.logger.error(f"Normalized mime type: {repr(normalized_mime_type)}")
                    self.logger.error(f"Original mime type: {repr(mime_type)}")
                    self.logger.error(f"Type of normalized_mime_type: {type(normalized_mime_type)}")
                    
                    # Enhanced fallback mechanisms
                    
                    # Fallback 1: Try to detect from filename if available
                    if filename:
                        self.logger.warning(f"Attempting filename-based detection for {filename}")
                        try:
                            filename_mime_type = self._detect_by_filename(filename)
                            self.logger.info(f"Filename-based detection successful: {filename_mime_type}")
                            return filename_mime_type
                        except UnsupportedFileType:
                            self.logger.warning(f"Filename-based detection also failed for {filename}")
                    
                    # Fallback 2: Check if it's a generic image type we should support
                    if normalized_mime_type.startswith('image/'):
                        self.logger.warning(f"Unknown image MIME type {normalized_mime_type}, attempting generic image handling")
                        if filename:
                            # Extract extension and try to map to a known image type
                            extension = Path(filename).suffix.lower()
                            if extension in ['.jpg', '.jpeg']:
                                self.logger.info(f"Treating unknown image type as JPEG based on extension {extension}")
                                return 'image/jpeg'
                            elif extension in ['.png']:
                                self.logger.info(f"Treating unknown image type as PNG based on extension {extension}")
                                return 'image/png'
                            elif extension in ['.gif']:
                                self.logger.info(f"Treating unknown image type as GIF based on extension {extension}")
                                return 'image/gif'
                            elif extension in ['.tiff', '.tif']:
                                self.logger.info(f"Treating unknown image type as TIFF based on extension {extension}")
                                return 'image/tiff'
                    
                    # Fallback 3: Check if it's a text type we should support
                    if normalized_mime_type.startswith('text/'):
                        self.logger.warning(f"Unknown text MIME type {normalized_mime_type}, attempting generic text handling")
                        # For unknown text types, default to plain text if we have a reasonable filename
                        if filename and any(filename.lower().endswith(ext) for ext in ['.txt', '.text', '.log']):
                            self.logger.info(f"Treating unknown text type as plain text based on filename {filename}")
                            return 'text/plain'
                    
                    # If all fallbacks fail, raise the original error
                    raise UnsupportedFileType(f"Unsupported file type: {normalized_mime_type} (original: {mime_type})")
                
                # Validate against filename extension if provided
                if filename and not self._validate_extension_against_filename(filename, normalized_mime_type):
                    self.logger.warning(f"Extension mismatch for {filename}: detected {normalized_mime_type}")
                
                return normalized_mime_type
                
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
    
    def _detect_by_extension(self, file_path: Union[str, Path, BinaryIO]) -> str:
        """Fallback method to detect MIME type by file extension when magic is not available."""
        if isinstance(file_path, (str, Path)):
            filename = str(file_path)
        else:
            # For file-like objects, we can't determine extension without filename
            raise UnsupportedFileType("Cannot determine file type from file-like object without python-magic library")
        
        return self._detect_by_filename(filename)
    
    def _detect_by_filename(self, filename: str) -> str:
        """Detect MIME type based on file extension."""
        extension = Path(filename).suffix.lower()
        
        # Handle common extension variations
        extension_mappings = {
            '.jpeg': '.jpg',
            '.htm': '.html',
            '.tif': '.tiff'
        }
        
        normalized_extension = extension_mappings.get(extension, extension)
        
        # Find MIME type by extension
        for mime_type, expected_ext in self.MIME_TO_EXTENSION.items():
            if expected_ext == normalized_extension:
                self.logger.debug(f"Detected MIME type by extension: {mime_type}")
                return mime_type
        
        raise UnsupportedFileType(f"Unsupported file extension: {extension}")