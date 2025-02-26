# image_processor.py
from typing import Union, Dict
from pathlib import Path
from PIL import Image
import logging
from .ocr_processor import OCRProcessor

class ImageProcessingError(Exception):
    """Raised when image processing fails."""
    pass

class ImageProcessor:
    """Processes and analyzes images."""

    def __init__(self):
        self.ocr = OCRProcessor()

    def process(self, image_path: Union[str, Path]) -> Dict:
        """Process an image and return extracted information."""
        try:
            image = Image.open(str(image_path))
            
            # Extract basic image information
            info = {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'dpi': image.info.get('dpi'),
            }
            
            # Perform OCR if needed
            info['text'] = self.ocr.process(image_path)
            
            return info
        except Exception as e:
            logging.error(f"Image processing failed: {str(e)}")
            raise ImageProcessingError(f"Failed to process image: {str(e)}")