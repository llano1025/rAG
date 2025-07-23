# ocr_processor.py
from typing import Union, Optional, Literal
from pathlib import Path
import logging
import base64
import requests
from io import BytesIO
import json
from enum import Enum

# Optional imports with availability flags
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    logging.warning("pytesseract not available, OCR functionality will be limited")
    pytesseract = None
    PYTESSERACT_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logging.warning("PIL/Pillow not available, image processing will be limited")
    Image = None
    PIL_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    logging.warning("PyMuPDF not available for OCR, PDF processing will be limited")
    fitz = None
    PYMUPDF_AVAILABLE = False

class OCRMethod(Enum):
    TESSERACT = "tesseract"
    VISION_LLM = "vision_llm"

class OCRError(Exception):
    """Raised when OCR processing fails."""
    pass

class OCRProcessor:
    """Handles OCR processing for images and scanned documents using multiple methods."""

    def __init__(self, 
                 method: OCRMethod = OCRMethod.TESSERACT,
                 language: str = 'eng',
                 vision_llm_config: Optional[dict] = None):
        """
        Initialize OCR processor with specified method and configuration.
        
        Args:
            method: OCR method to use (tesseract or vision_llm)
            language: Language for Tesseract OCR
            vision_llm_config: Configuration for Vision LLM including:
                - api_key: API key for the Vision LLM service
                - model: Model name/version
                - endpoint: API endpoint URL
                - Additional model-specific parameters
        """
        self.method = method
        self.language = language
        self.vision_llm_config = vision_llm_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Check if dependencies are available for the selected method
        if self.method == OCRMethod.TESSERACT:
            if not PYTESSERACT_AVAILABLE or not PIL_AVAILABLE:
                self.logger.warning("Tesseract OCR dependencies not available, method may not work")
        
        # Validate config if using Vision LLM
        if self.method == OCRMethod.VISION_LLM and not self._validate_llm_config():
            raise ValueError("Invalid Vision LLM configuration")

    def _validate_llm_config(self) -> bool:
        """Validate Vision LLM configuration."""
        required_fields = ['api_key', 'model', 'endpoint']
        return all(field in self.vision_llm_config for field in required_fields)

    def process(self, 
                image_path: Union[str, Path], 
                method: Optional[OCRMethod] = None) -> str:
        """
        Process an image file and extract text using specified OCR method.
        
        Args:
            image_path: Path to the image file
            method: Override default OCR method for this specific image
        
        Returns:
            Extracted text from the image
        """
        try:
            if not PIL_AVAILABLE:
                raise OCRError("PIL/Pillow is required for image processing but is not available")
            
            method = method or self.method
            image = Image.open(str(image_path))
            
            if method == OCRMethod.TESSERACT:
                return self._process_with_tesseract(image)
            elif method == OCRMethod.VISION_LLM:
                return self._process_with_vision_llm(image)
            else:
                raise OCRError(f"Unsupported OCR method: {method}")
                
        except Exception as e:
            self.logger.error(f"OCR processing failed: {str(e)}")
            raise OCRError(f"Failed to process image: {str(e)}")

    def _process_with_tesseract(self, image: Image.Image) -> str:
        """Process image using Tesseract OCR."""
        return pytesseract.image_to_string(image, lang=self.language)

    def _process_with_vision_llm(self, image: Image.Image) -> str:
        """Process image using Vision LLM."""
        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format=image.format or 'PNG')
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # Prepare request for the Vision LLM API
            headers = {
                'Authorization': f"Bearer {self.vision_llm_config['api_key']}",
                'Content-Type': 'application/json'
            }
            
            # Construct payload based on the model's requirements
            payload = {
                'model': self.vision_llm_config['model'],
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'Extract and return all visible text from this image.'
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                **self.vision_llm_config.get('additional_params', {})
            }

            # Make API request
            response = requests.post(
                self.vision_llm_config['endpoint'],
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            # Parse response
            result = response.json()
            
            # Extract text from response based on API format
            extracted_text = self._parse_llm_response(result)
            
            return extracted_text

        except requests.exceptions.RequestException as e:
            logging.error(f"Vision LLM API request failed: {str(e)}")
            raise OCRError(f"Vision LLM processing failed: {str(e)}")
        except Exception as e:
            logging.error(f"Vision LLM processing failed: {str(e)}")
            raise OCRError(f"Vision LLM processing failed: {str(e)}")

    def _parse_llm_response(self, response: dict) -> str:
        """
        Parse the Vision LLM API response to extract text.
        Override this method if using a different Vision LLM API.
        """
        try:
            # Default parsing for standard response format
            # Adjust based on your specific Vision LLM API response structure
            if 'choices' in response:
                return response['choices'][0]['message']['content']
            elif 'text' in response:
                return response['text']
            else:
                raise OCRError("Unexpected Vision LLM response format")
        except Exception as e:
            raise OCRError(f"Failed to parse Vision LLM response: {str(e)}")

    def process_pdf_page(self, 
                        pdf_path: Union[str, Path], 
                        page_num: int,
                        method: Optional[OCRMethod] = None) -> str:
        """
        Process a specific PDF page using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to process
            method: Override default OCR method for this specific page
        """
        try:
            with fitz.open(str(pdf_path)) as doc:
                page = doc[page_num]
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                return self.process_image(img, method)
        except Exception as e:
            logging.error(f"PDF OCR processing failed: {str(e)}")
            raise OCRError(f"Failed to process PDF page: {str(e)}")

    def batch_process(self, 
                     image_paths: list[Union[str, Path]], 
                     method: Optional[OCRMethod] = None) -> dict[str, str]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of paths to image files
            method: Override default OCR method for this batch
        
        Returns:
            Dictionary mapping image paths to extracted text
        """
        results = {}
        for image_path in image_paths:
            try:
                results[str(image_path)] = self.process_image(image_path, method)
            except OCRError as e:
                logging.error(f"Failed to process {image_path}: {str(e)}")
                results[str(image_path)] = f"ERROR: {str(e)}"
        return results