# ocr_processor.py
from typing import Union, Optional, List, Dict
from pathlib import Path
import logging
import base64
import requests
from io import BytesIO
import time
import random
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

class VisionProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"

class OCRError(Exception):
    """Raised when OCR processing fails."""
    pass

class OCRProcessor:
    """Handles OCR processing for images and scanned documents using multiple methods."""

    def __init__(self, 
                 method: OCRMethod = OCRMethod.TESSERACT,
                 language: str = 'eng',
                 vision_llm_config: Optional[dict] = None,
                 vision_provider: VisionProvider = VisionProvider.GEMINI,
                 enable_fallback: bool = True,
                 timeout: int = 60):
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
            vision_provider: Vision LLM provider to use (openai, gemini, claude)
            enable_fallback: Whether to fallback to Tesseract if Vision LLM fails
            timeout: Timeout in seconds for Vision LLM API calls
        """
        self.method = method
        self.language = language
        self.vision_llm_config = vision_llm_config or {}
        self.vision_provider = vision_provider
        self.enable_fallback = enable_fallback
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Check if dependencies are available for the selected method
        if self.method == OCRMethod.TESSERACT:
            if not PYTESSERACT_AVAILABLE or not PIL_AVAILABLE:
                self.logger.warning("Tesseract OCR dependencies not available, method may not work")
        
        # Validate config if using Vision LLM
        if self.method == OCRMethod.VISION_LLM and not self._validate_llm_config():
            if self.enable_fallback:
                self.logger.warning("Invalid Vision LLM configuration, will fallback to Tesseract")
                self.method = OCRMethod.TESSERACT
            else:
                raise ValueError("Invalid Vision LLM configuration")

    def _validate_llm_config(self) -> bool:
        """Validate Vision LLM configuration based on provider."""
        if self.vision_provider == VisionProvider.OPENAI:
            required_fields = ['api_key', 'model', 'endpoint']
        elif self.vision_provider == VisionProvider.GEMINI:
            required_fields = ['api_key', 'model']
        elif self.vision_provider == VisionProvider.CLAUDE:
            required_fields = ['api_key', 'model', 'endpoint']
        else:
            self.logger.error(f"Unsupported vision provider: {self.vision_provider}")
            return False
        
        is_valid = all(field in self.vision_llm_config for field in required_fields)
        if not is_valid:
            missing_fields = [field for field in required_fields if field not in self.vision_llm_config]
            self.logger.error(f"Missing required fields for {self.vision_provider.value}: {missing_fields}")
        
        return is_valid

    def process(self, 
                image_path: Union[str, Path], 
                method: Optional[OCRMethod] = None,
                vision_provider: Optional[VisionProvider] = None) -> str:
        """
        Process an image file and extract text using specified OCR method.
        
        Args:
            image_path: Path to the image file
            method: Override default OCR method for this specific image
            vision_provider: Override vision provider for this specific image
        
        Returns:
            Extracted text from the image
        """
        try:
            if not PIL_AVAILABLE:
                raise OCRError("PIL/Pillow is required for image processing but is not available")
            
            method = method or self.method
            vision_provider = vision_provider or self.vision_provider
            image = Image.open(str(image_path))
            
            # Preprocess image (resize if too large)
            image = self._preprocess_image(image)
            
            if method == OCRMethod.TESSERACT:
                return self._process_with_tesseract(image)
            elif method == OCRMethod.VISION_LLM:
                try:
                    return self._process_with_vision_llm(image, vision_provider)
                except OCRError as e:
                    if self.enable_fallback:
                        self.logger.warning(f"Vision LLM failed, falling back to Tesseract: {str(e)}")
                        return self._process_with_tesseract(image)
                    else:
                        raise e
            else:
                raise OCRError(f"Unsupported OCR method: {method}")
                
        except Exception as e:
            self.logger.error(f"OCR processing failed: {str(e)}")
            if self.enable_fallback and method == OCRMethod.VISION_LLM:
                try:
                    self.logger.warning("Attempting fallback to Tesseract after unexpected error")
                    image = Image.open(str(image_path))
                    image = self._preprocess_image(image)
                    return self._process_with_tesseract(image)
                except Exception as fallback_e:
                    self.logger.error(f"Fallback also failed: {str(fallback_e)}")
            raise OCRError(f"Failed to process image: {str(e)}")

    def _process_with_tesseract(self, image: Image.Image) -> str:
        """Process image using Tesseract OCR."""
        return pytesseract.image_to_string(image, lang=self.language)

    def _preprocess_image(self, image: Image.Image, max_size: int = 2048) -> Image.Image:
        """Preprocess image to optimize for OCR processing."""
        try:
            # Check if image is too large and resize if needed
            width, height = image.size
            if width > max_size or height > max_size:
                # Calculate new size maintaining aspect ratio
                if width > height:
                    new_width = max_size
                    new_height = int((height * max_size) / width)
                else:
                    new_height = max_size
                    new_width = int((width * max_size) / height)
                
                self.logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed (for consistent processing)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed, using original: {str(e)}")
            return image
    
    def _process_with_vision_llm(self, image: Image.Image, provider: VisionProvider = None) -> str:
        """Process image using Vision LLM with retry logic."""
        provider = provider or self.vision_provider
        
        for attempt in range(3):  # Retry up to 3 times
            try:
                if provider == VisionProvider.OPENAI:
                    return self._process_with_openai_vision(image, attempt)
                elif provider == VisionProvider.GEMINI:
                    return self._process_with_gemini_vision(image, attempt)
                elif provider == VisionProvider.CLAUDE:
                    return self._process_with_claude_vision(image, attempt)
                else:
                    raise OCRError(f"Unsupported vision provider: {provider}")
                    
            except requests.exceptions.Timeout:
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                if attempt < 2:
                    self.logger.warning(f"Vision LLM timeout, retrying in {wait_time:.1f}s (attempt {attempt + 1}/3)")
                    time.sleep(wait_time)
                    continue
                else:
                    raise OCRError(f"Vision LLM processing timed out after 3 attempts")
            except requests.exceptions.RequestException as e:
                if attempt < 2 and "rate limit" in str(e).lower():
                    wait_time = (2 ** attempt) * 2 + random.uniform(0, 2)  # Longer wait for rate limits
                    self.logger.warning(f"Rate limit hit, retrying in {wait_time:.1f}s (attempt {attempt + 1}/3)")
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"Vision LLM API request failed: {str(e)}")
                    raise OCRError(f"Vision LLM processing failed: {str(e)}")
            except Exception as e:
                self.logger.error(f"Vision LLM processing failed: {str(e)}")
                raise OCRError(f"Vision LLM processing failed: {str(e)}")
        
        raise OCRError("All retry attempts exhausted")

    def _process_with_openai_vision(self, image: Image.Image, attempt: int = 0) -> str:
        """Process image using OpenAI Vision API."""
        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format='PNG')
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            headers = {
                'Authorization': f"Bearer {self.vision_llm_config['api_key']}",
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.vision_llm_config.get('model', 'gpt-4o-2024-08-06'),
                'messages': [{
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': 'Extract and return all visible text from this image. Return only the text content without any additional commentary.'
                    }, {
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:image/png;base64,{img_base64}"
                        }
                    }]
                }],
                **self.vision_llm_config.get('additional_params', {})
            }

            endpoint = self.vision_llm_config.get('endpoint', 'https://api.openai.com/v1/chat/completions')
            response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return content.strip() if content else ""
            else:
                raise OCRError("Unexpected OpenAI response format")
                
        except Exception as e:
            self.logger.error(f"OpenAI Vision processing failed (attempt {attempt + 1}): {str(e)}")
            raise
    
    def _process_with_gemini_vision(self, image: Image.Image, attempt: int = 0) -> str:
        """Process image using Google Gemini Vision API."""
        try:
            import google.generativeai as genai
            
            # Configure Gemini
            genai.configure(api_key=self.vision_llm_config['api_key'])
            
            # Initialize the model
            model_name = self.vision_llm_config.get('model', 'gemini-2.5-flash')
            model = genai.GenerativeModel(model_name)
            
            # Convert image to format Gemini expects
            buffered = BytesIO()
            image.save(buffered, format='PNG')
            image_data = buffered.getvalue()
            
            # Create prompt for OCR
            prompt = "Extract and return all visible text from this image. Return only the text content without any additional commentary or formatting."
            
            # Generate content with timeout handling
            import concurrent.futures
            
            def generate_content():
                return model.generate_content([prompt, {'mime_type': 'image/png', 'data': image_data}])
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(generate_content)
                try:
                    response = future.result(timeout=self.timeout)
                    if response and response.text:
                        return response.text.strip()
                    else:
                        raise OCRError("Empty response from Gemini")
                except concurrent.futures.TimeoutError:
                    raise requests.exceptions.Timeout("Gemini API timeout")
                
        except ImportError:
            self.logger.error("google-generativeai package not installed")
            raise OCRError("Gemini Vision requires google-generativeai package")
        except Exception as e:
            self.logger.error(f"Gemini Vision processing failed (attempt {attempt + 1}): {str(e)}")
            raise
    
    def _process_with_claude_vision(self, image: Image.Image, attempt: int = 0) -> str:
        """Process image using Anthropic Claude Vision API."""
        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format='PNG')
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            headers = {
                'x-api-key': self.vision_llm_config['api_key'],
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
            
            payload = {
                'model': self.vision_llm_config.get('model', 'claude-3-5-sonnet-20241022'),
                'max_tokens': self.vision_llm_config.get('max_tokens', 4096),
                'messages': [{
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': 'Extract and return all visible text from this image. Return only the text content without any additional commentary.'
                    }, {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': 'image/png',
                            'data': img_base64
                        }
                    }]
                }]
            }

            endpoint = self.vision_llm_config.get('endpoint', 'https://api.anthropic.com/v1/messages')
            response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            if 'content' in result and len(result['content']) > 0:
                content = result['content'][0]['text']
                return content.strip() if content else ""
            else:
                raise OCRError("Unexpected Claude response format")
                
        except Exception as e:
            self.logger.error(f"Claude Vision processing failed (attempt {attempt + 1}): {str(e)}")
            raise

    def process_image(self, image: Image.Image, method: Optional[OCRMethod] = None, vision_provider: Optional[VisionProvider] = None) -> str:
        """Process an image object directly."""
        try:
            method = method or self.method
            vision_provider = vision_provider or self.vision_provider
            
            # Preprocess image
            image = self._preprocess_image(image)
            
            if method == OCRMethod.TESSERACT:
                return self._process_with_tesseract(image)
            elif method == OCRMethod.VISION_LLM:
                try:
                    return self._process_with_vision_llm(image, vision_provider)
                except OCRError as e:
                    if self.enable_fallback:
                        self.logger.warning(f"Vision LLM failed, falling back to Tesseract: {str(e)}")
                        return self._process_with_tesseract(image)
                    else:
                        raise e
            else:
                raise OCRError(f"Unsupported OCR method: {method}")
                
        except Exception as e:
            self.logger.error(f"Image OCR processing failed: {str(e)}")
            if self.enable_fallback and method == OCRMethod.VISION_LLM:
                try:
                    self.logger.warning("Attempting fallback to Tesseract after unexpected error")
                    return self._process_with_tesseract(image)
                except Exception as fallback_e:
                    self.logger.error(f"Fallback also failed: {str(fallback_e)}")
            raise OCRError(f"Failed to process image: {str(e)}")
    
    def process_pdf_page(self, 
                        pdf_path: Union[str, Path], 
                        page_num: int,
                        method: Optional[OCRMethod] = None,
                        vision_provider: Optional[VisionProvider] = None) -> str:
        """
        Process a specific PDF page using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to process
            method: Override default OCR method for this specific page
            vision_provider: Override vision provider for this specific page
        """
        try:
            if not PYMUPDF_AVAILABLE:
                raise OCRError("PyMuPDF is required for PDF page processing")
                
            with fitz.open(str(pdf_path)) as doc:
                page = doc[page_num]
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                return self.process_image(img, method, vision_provider)
        except Exception as e:
            self.logger.error(f"PDF OCR processing failed: {str(e)}")
            raise OCRError(f"Failed to process PDF page: {str(e)}")

    def batch_process(self, 
                     image_paths: List[Union[str, Path]], 
                     method: Optional[OCRMethod] = None,
                     vision_provider: Optional[VisionProvider] = None) -> Dict[str, str]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of paths to image files
            method: Override default OCR method for this batch
            vision_provider: Override vision provider for this batch
        
        Returns:
            Dictionary mapping image paths to extracted text
        """
        results = {}
        for image_path in image_paths:
            try:
                results[str(image_path)] = self.process(image_path, method, vision_provider)
            except OCRError as e:
                self.logger.error(f"Failed to process {image_path}: {str(e)}")
                results[str(image_path)] = f"ERROR: {str(e)}"
        return results