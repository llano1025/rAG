"""
OCR-specific API routes for enhanced image and document processing.
Provides endpoints for OCR processing, preview, and configuration.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import tempfile
import os
from pathlib import Path

from ..middleware.auth import get_current_active_user
from ..controllers.document_controller import DocumentController
from file_processor.ocr_processor import OCRProcessor, OCRMethod, OCRError
from file_processor.type_detector import FileTypeDetector
from database.models import User
from utils.security.audit_logger import log_user_action
from database.connection import get_db

router = APIRouter(prefix="/ocr", tags=["ocr"])
logger = logging.getLogger(__name__)

class OCRMethodInfo(BaseModel):
    """Information about available OCR methods."""
    method: str
    display_name: str
    description: str
    supported_formats: List[str]
    requires_api_key: bool
    estimated_cost: Optional[str] = None
    quality_level: str  # "basic", "good", "excellent"

class OCRResult(BaseModel):
    """OCR processing result."""
    extracted_text: str
    confidence_score: Optional[float] = None
    method_used: str
    processing_time_ms: float
    language_detected: Optional[str] = None
    metadata: Dict[str, Any] = {}

class OCRPreviewRequest(BaseModel):
    """Request for OCR preview processing."""
    method: Optional[str] = "tesseract"
    language: Optional[str] = "eng"
    preprocessing_options: Optional[Dict[str, Any]] = {}

class BatchOCRRequest(BaseModel):
    """Request for batch OCR processing."""
    method: Optional[str] = "tesseract"
    language: Optional[str] = "eng"
    return_confidence: bool = True
    preprocessing_options: Optional[Dict[str, Any]] = {}

@router.get("/methods", response_model=List[OCRMethodInfo])
async def get_ocr_methods(
    current_user: User = Depends(get_current_active_user)
):
    """Get available OCR methods and their configurations."""
    try:
        methods = [
            OCRMethodInfo(
                method="tesseract",
                display_name="Tesseract OCR",
                description="Traditional OCR engine with good accuracy for printed text",
                supported_formats=["image/jpeg", "image/png", "image/tiff", "image/gif"],
                requires_api_key=False,
                estimated_cost="Free",
                quality_level="good"
            ),
            OCRMethodInfo(
                method="vision_llm",
                display_name="Vision LLM",
                description="AI-powered OCR with excellent accuracy for complex layouts and handwriting",
                supported_formats=["image/jpeg", "image/png", "image/tiff", "image/gif"],
                requires_api_key=True,
                estimated_cost="~$0.01-0.05 per image",
                quality_level="excellent"
            )
        ]
        
        return methods
        
    except Exception as e:
        logger.error(f"Failed to get OCR methods: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve OCR methods"
        )

@router.post("/process", response_model=OCRResult)
async def process_image_ocr(
    file: UploadFile = File(...),
    method: str = Form("tesseract"),
    language: str = Form("eng"),
    return_confidence: bool = Form(True),
    current_user: User = Depends(get_current_active_user),
    db = Depends(get_db)
):
    """Process a single image with OCR and return extracted text."""
    logger.info(f"=== OCR PROCESS ROUTE CALLED ===")
    logger.info(f"OCR Route: /api/ocr/process - Filename: {file.filename}")
    logger.info(f"OCR Route: File content type: {file.content_type}")
    logger.info(f"OCR Route: OCR method: {method}, language: {language}")
    logger.info(f"OCR Route: This is NOT the document upload route!")
    
    import time
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only image files are supported for OCR processing"
            )
        
        # Validate OCR method
        try:
            ocr_method = OCRMethod(method)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid OCR method: {method}. Supported methods: tesseract, vision_llm"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_path = temp_file.name
        
        try:
            # Initialize OCR processor
            vision_llm_config = None
            if ocr_method == OCRMethod.VISION_LLM:
                from config import get_settings
                settings = get_settings()
                if not settings.OPENAI_API_KEY:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Vision LLM requires OpenAI API key configuration"
                    )
                vision_llm_config = {
                    'api_key': settings.OPENAI_API_KEY,
                    'model': 'gpt-4o-2024-08-06',  # Vision-capable model
                    'endpoint': 'https://api.openai.com/v1/chat/completions',
                    'additional_params': {
                        'max_tokens': 4096,
                        'temperature': 0.1
                    }
                }
            
            ocr_processor = OCRProcessor(
                method=ocr_method,
                language=language,
                vision_llm_config=vision_llm_config
            )
            
            # Process image
            extracted_text = ocr_processor.process(temp_path)
            processing_time = (time.time() - start_time) * 1000
            
            # Calculate confidence score (simplified - can be enhanced)
            confidence_score = None
            if return_confidence:
                # For Tesseract, we could get actual confidence
                # For Vision LLM, we use a heuristic based on text length and completeness
                if ocr_method == OCRMethod.TESSERACT:
                    confidence_score = 0.85  # Placeholder - implement actual confidence
                else:
                    # Simple heuristic for Vision LLM
                    confidence_score = min(0.95, 0.7 + (len(extracted_text.strip()) / 1000))
            
            result = OCRResult(
                extracted_text=extracted_text,
                confidence_score=confidence_score,
                method_used=method,
                processing_time_ms=processing_time,
                language_detected=language,  # Could be enhanced with actual detection
                metadata={
                    "file_name": file.filename,
                    "file_size": len(file_content),
                    "content_type": file.content_type
                }
            )
            
            # Log OCR processing
            await log_user_action(
                user_id=current_user.id,
                action="ocr_process",
                resource_type="image",
                details={
                    "method": method,
                    "file_name": file.filename,
                    "file_size": len(file_content),
                    "processing_time_ms": processing_time,
                    "text_length": len(extracted_text)
                },
                db=db
            )
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except OCRError as e:
        logger.error(f"OCR processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"OCR processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"OCR endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OCR processing failed due to server error"
        )

@router.post("/preview", response_model=OCRResult)
async def preview_ocr_processing(
    file: UploadFile = File(...),
    method: str = Form("tesseract"),
    language: str = Form("eng"),
    current_user: User = Depends(get_current_active_user)
):
    """Preview OCR processing without saving results."""
    # Use the same processing logic as the main process endpoint
    # but don't save to database or create document records
    return await process_image_ocr(
        file=file,
        method=method,
        language=language,
        return_confidence=True,
        current_user=current_user,
        db=next(get_db())
    )

@router.post("/batch", response_model=List[OCRResult])
async def batch_process_ocr(
    files: List[UploadFile] = File(...),
    request: BatchOCRRequest = Depends(),
    current_user: User = Depends(get_current_active_user),
    db = Depends(get_db)
):
    """Process multiple images with OCR in batch."""
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size limited to 10 images"
        )
    
    results = []
    for file in files:
        try:
            # Process each file individually
            result = await process_image_ocr(
                file=file,
                method=request.method,
                language=request.language,
                return_confidence=request.return_confidence,
                current_user=current_user,
                db=db
            )
            results.append(result)
        except Exception as e:
            # Include error results in batch
            error_result = OCRResult(
                extracted_text=f"ERROR: {str(e)}",
                confidence_score=0.0,
                method_used=request.method,
                processing_time_ms=0.0,
                metadata={
                    "file_name": file.filename,
                    "error": True,
                    "error_message": str(e)
                }
            )
            results.append(error_result)
    
    # Log batch processing
    await log_user_action(
        user_id=current_user.id,
        action="ocr_batch_process",
        resource_type="batch",
        details={
            "method": request.method,
            "file_count": len(files),
            "successful_count": len([r for r in results if not r.metadata.get("error", False)])
        },
        db=db
    )
    
    return results

@router.get("/languages")
async def get_supported_languages(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of supported OCR languages."""
    # Common Tesseract language codes
    languages = [
        {"code": "eng", "name": "English", "supported_by": ["tesseract", "vision_llm"]},
        {"code": "spa", "name": "Spanish", "supported_by": ["tesseract", "vision_llm"]},
        {"code": "fra", "name": "French", "supported_by": ["tesseract", "vision_llm"]},
        {"code": "deu", "name": "German", "supported_by": ["tesseract", "vision_llm"]},
        {"code": "ita", "name": "Italian", "supported_by": ["tesseract", "vision_llm"]},
        {"code": "por", "name": "Portuguese", "supported_by": ["tesseract", "vision_llm"]},
        {"code": "rus", "name": "Russian", "supported_by": ["tesseract", "vision_llm"]},
        {"code": "ara", "name": "Arabic", "supported_by": ["tesseract", "vision_llm"]},
        {"code": "chi_sim", "name": "Chinese (Simplified)", "supported_by": ["tesseract", "vision_llm"]},
        {"code": "chi_tra", "name": "Chinese (Traditional)", "supported_by": ["tesseract", "vision_llm"]},
        {"code": "jpn", "name": "Japanese", "supported_by": ["tesseract", "vision_llm"]},
        {"code": "kor", "name": "Korean", "supported_by": ["tesseract", "vision_llm"]},
    ]
    
    return languages

@router.get("/config")
async def get_ocr_config(
    current_user: User = Depends(get_current_active_user)
):
    """Get current OCR configuration and capabilities."""
    from config import get_settings
    settings = get_settings()
    
    config = {
        "ocr_enabled": settings.OCR_ENABLED,
        "default_language": settings.OCR_LANGUAGE,
        "tesseract_available": True,  # Assume available - could check dynamically
        "vision_llm_available": bool(settings.OPENAI_API_KEY),
        "max_file_size_mb": 50,
        "supported_formats": ["image/jpeg", "image/png", "image/tiff", "image/gif"],
        "batch_limit": 10
    }
    
    return config