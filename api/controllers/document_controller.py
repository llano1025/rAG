# api/controllers/document_controller.py
from fastapi import HTTPException, UploadFile
from typing import List, Optional, Dict
from datetime import datetime

from file_processor.type_detector import FileTypeDetector
from file_processor.text_extractor import TextExtractor
from file_processor.ocr_processor import OCRProcessor
from file_processor.metadata_extractor import MetadataExtractor
from file_processor.image_processor import ImageProcessor
from api.controllers.vector_controller import VectorController

class DocumentController:
    def __init__(
        self,
        type_detector: FileTypeDetector,
        text_extractor: TextExtractor,
        ocr_processor: OCRProcessor,
        metadata_extractor: MetadataExtractor,
        image_processor: ImageProcessor,
        vector_controller: VectorController
    ):
        self.type_detector = type_detector
        self.text_extractor = text_extractor
        self.ocr_processor = ocr_processor
        self.metadata_extractor = metadata_extractor
        self.image_processor = image_processor
        self.vector_controller = vector_controller

    async def upload_document(
        self,
        file: UploadFile,
        folder_id: Optional[str] = None,
        tags: List[str] = None
    ) -> dict:
        try:
            # Detect file type
            file_type = await self.type_detector.detect(file)
            
            # Extract content based on file type
            if file_type.is_image:
                content = await self.ocr_processor.process(file)
                image_data = await self.image_processor.process(file)
            else:
                content = await self.text_extractor.extract(file)
                image_data = None
            
            # Extract metadata
            metadata = await self.metadata_extractor.extract(file)
            
            # Create document record
            document = {
                "name": file.filename,
                "type": file_type,
                "folder_id": folder_id,
                "tags": tags or [],
                "created_at": datetime.utcnow(),
                "metadata": metadata,
                "content": content,
                "image_data": image_data
            }

            # Process vectors using VectorController
            vector_metadata = {
                "document_name": document["name"],
                "file_type": document["type"],
                "tags": document["tags"],
                **metadata
            }
            vector_result = await self.vector_controller.process_document(
                content=content,
                document_id=str(document.get("id")),  # Assuming document has an ID
                metadata=vector_metadata
            )
            
            # Add vector processing results to document
            document.update({
                "chunk_count": vector_result["chunk_count"],
                "chunks": vector_result["chunks"],
                "version": vector_result["version"]
            })
            
            return document
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))