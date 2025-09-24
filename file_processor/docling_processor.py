# docling_processor.py
"""
Simple Docling OCR processor using Vision Language Model pipeline.
"""

from typing import Union, Optional
from pathlib import Path
import logging
import tempfile
import os

# Optional imports with availability flags
try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions, InferenceFramework, TransformersModelType
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.pipeline_options_vlm_model import (
        ApiVlmOptions,
        InferenceFramework,
        InlineVlmOptions,
        ResponseFormat,
        TransformersModelType,
        TransformersPromptStyle,
    )

    DOCLING_AVAILABLE = True
except ImportError:
    logging.debug("Docling not available")
    DocumentConverter = None
    InputFormat = None
    PdfFormatOption = None
    VlmPipeline = None
    VlmPipelineOptions = None
    vlm_model_specs = None
    DOCLING_AVAILABLE = False

logger = logging.getLogger(__name__)

class DoclingError(Exception):
    """Raised when Docling processing fails."""
    pass

class DoclingProcessor:
    """Simple Docling processor using VLM pipeline."""

    def __init__(self):
        if not DOCLING_AVAILABLE:
            raise DoclingError("Docling library is not available")
        
        self.pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,  # <-- change the model here
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=self.pipeline_options,
                ),
            }
        )

    def extract_text(self, file_path: Union[str, Path]) -> str:
        """Extract text from document using Docling VLM pipeline."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise DoclingError(f"File not found: {file_path}")

            # Simple conversion following the example
            doc = self.converter.convert(source=file_path).document
            return doc.export_to_markdown()

        except Exception as e:
            logger.error(f"Docling text extraction failed for {file_path}: {e}")
            raise DoclingError(f"Text extraction failed: {e}")

    def extract_from_bytes(self, file_content: bytes, filename: str = "document") -> str:
        """Extract text from document content bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_path = temp_file.name

        try:
            return self.extract_text(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

def create_docling_processor() -> Optional[DoclingProcessor]:
    """Create a DoclingProcessor instance with error handling."""
    try:
        return DoclingProcessor()
    except DoclingError as e:
        logger.debug(f"Could not create DoclingProcessor: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating DoclingProcessor: {e}")
        return None