from typing import List, Tuple, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}. Embedding functionality will be limited.")
    TORCH_AVAILABLE = False
    torch = None
    AutoModel = None
    AutoTokenizer = None
    np = None

from .chunking import Chunk
from database.models import Document, DocumentChunk

logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass

class EmbeddingManager:
    """Manages document embeddings with contextual support."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = None,
        batch_size: int = 32
    ):
        if not TORCH_AVAILABLE:
            raise EmbeddingError("PyTorch and transformers are required for embedding functionality")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        try:
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize embedding model: {e}")

    async def generate_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate both content and context embeddings for chunks."""
        try:
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                
                # Generate content embeddings
                texts = [chunk.text for chunk in batch]
                content_embeddings = self._batch_encode(texts)
                
                # Generate context embeddings
                context_texts = [
                    chunk.context_text if chunk.context_text 
                    else chunk.text for chunk in batch
                ]
                context_embeddings = self._batch_encode(context_texts)
                
                # Assign embeddings to chunks
                for chunk, content_emb, context_emb in zip(
                    batch, content_embeddings, context_embeddings
                ):
                    chunk.embedding = content_emb
                    chunk.context_embedding = context_emb
            
            return chunks
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into embeddings."""
        with torch.no_grad():
            # Tokenize with special tokens
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            outputs = self.model(**inputs)
            
            # Use pooled output (CLS token) for sentence embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
            
            return embeddings

    async def generate_query_embeddings(
        self,
        query: str,
        query_context: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate embeddings for query and its context."""
        try:
            # Generate query embedding
            query_embedding = self._batch_encode([query])[0]
            
            # Generate context embedding if provided
            if query_context:
                context_embedding = self._batch_encode([query_context])[0]
            else:
                context_embedding = query_embedding
            
            return query_embedding, context_embedding
            
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate query embeddings: {str(e)}"
            )

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self._batch_encode(batch_texts)
                all_embeddings.extend(batch_embeddings.tolist())
            
            return all_embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    async def batch_generate_embeddings(
        self,
        documents: List[Document],
        db: Session,
        storage_manager = None,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings for multiple documents in batch.
        
        Args:
            documents: List of Document objects to process
            db: Database session
            storage_manager: Vector storage manager for storing embeddings
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            start_time = datetime.utcnow()
            total_documents = len(documents)
            processed_count = 0
            error_count = 0
            total_chunks = 0
            
            logger.info(f"Starting batch embedding generation for {total_documents} documents")
            
            results = {
                "processed_documents": [],
                "failed_documents": [],
                "statistics": {}
            }
            
            # Process documents in smaller batches to manage memory
            doc_batch_size = min(5, total_documents)  # Process 5 documents at a time
            
            for i in range(0, total_documents, doc_batch_size):
                batch_docs = documents[i:i + doc_batch_size]
                
                # Process each document in the batch
                for document in batch_docs:
                    try:
                        # Get document chunks
                        chunks = db.query(DocumentChunk).filter(
                            DocumentChunk.document_id == document.id
                        ).all()
                        
                        if not chunks:
                            logger.warning(f"No chunks found for document {document.id}")
                            continue
                        
                        # Prepare texts for embedding generation
                        content_texts = [chunk.text for chunk in chunks]
                        context_texts = []
                        
                        for chunk in chunks:
                            # Create context text from surrounding chunks
                            context_parts = []
                            if chunk.context_before:
                                context_parts.append(chunk.context_before)
                            context_parts.append(chunk.text)
                            if chunk.context_after:
                                context_parts.append(chunk.context_after)
                            context_texts.append(" ".join(context_parts))
                        
                        # Generate embeddings in batches
                        content_embeddings = await self.generate_embeddings(content_texts)
                        context_embeddings = await self.generate_embeddings(context_texts)
                        
                        # Store embeddings if storage manager provided
                        if storage_manager:
                            index_name = f"doc_{document.id}"
                            chunk_ids = [chunk.chunk_id for chunk in chunks]
                            
                            # Prepare metadata for each chunk
                            metadata_list = []
                            for chunk in chunks:
                                metadata_list.append({
                                    "chunk_id": chunk.chunk_id,
                                    "document_id": document.id,
                                    "chunk_index": chunk.chunk_index,
                                    "text_length": chunk.text_length,
                                    "page_number": chunk.page_number,
                                    "section_title": chunk.section_title,
                                    "embedding_model": self.model_name if hasattr(self, 'model_name') else "unknown"
                                })
                            
                            # Add vectors to storage
                            added_ids = await storage_manager.add_vectors(
                                index_name=index_name,
                                content_vectors=content_embeddings,
                                context_vectors=context_embeddings,
                                metadata_list=metadata_list,
                                chunk_ids=chunk_ids
                            )
                            
                            logger.info(f"Added {len(added_ids)} vectors for document {document.id}")
                        
                        # Update processing statistics
                        processed_count += 1
                        total_chunks += len(chunks)
                        
                        results["processed_documents"].append({
                            "document_id": document.id,
                            "filename": document.filename,
                            "chunks_processed": len(chunks),
                            "embeddings_generated": len(content_embeddings)
                        })
                        
                        # Call progress callback if provided
                        if progress_callback:
                            await progress_callback(processed_count, total_documents)
                        
                    except Exception as e:
                        error_count += 1
                        error_msg = f"Failed to process document {document.id}: {str(e)}"
                        logger.error(error_msg)
                        
                        results["failed_documents"].append({
                            "document_id": document.id,
                            "filename": document.filename,
                            "error": str(e)
                        })
                
                # Small delay between batches to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            # Calculate final statistics
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            results["statistics"] = {
                "total_documents": total_documents,
                "processed_documents": processed_count,
                "failed_documents": error_count,
                "total_chunks": total_chunks,
                "processing_time_seconds": processing_time,
                "average_time_per_document": processing_time / total_documents if total_documents > 0 else 0,
                "chunks_per_second": total_chunks / processing_time if processing_time > 0 else 0,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            logger.info(f"Batch embedding generation completed: {processed_count}/{total_documents} documents processed in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Batch processing failed: {str(e)}")

    async def batch_index_documents(
        self,
        document_ids: List[int],
        db: Session,
        storage_manager = None,
        create_indices: bool = True,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Batch index multiple documents into vector database.
        
        Args:
            document_ids: List of document IDs to index
            db: Database session
            storage_manager: Vector storage manager
            create_indices: Whether to create new indices for documents
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with indexing results and statistics
        """
        try:
            start_time = datetime.utcnow()
            
            # Get documents from database
            documents = db.query(Document).filter(
                Document.id.in_(document_ids),
                Document.is_deleted == False
            ).all()
            
            if not documents:
                return {
                    "indexed_documents": [],
                    "failed_documents": [],
                    "statistics": {
                        "total_documents": 0,
                        "indexed_documents": 0,
                        "failed_documents": 0
                    }
                }
            
            logger.info(f"Starting batch indexing for {len(documents)} documents")
            
            # Create indices for documents if needed
            if create_indices and storage_manager:
                for document in documents:
                    index_name = f"doc_{document.id}"
                    
                    # Create index with default parameters
                    success = await storage_manager.create_index(
                        index_name=index_name,
                        embedding_dimension=384,  # Default for sentence-transformers
                        faiss_index_type="HNSW",
                        user_id=document.user_id,
                        document_id=document.id,
                        db=db
                    )
                    
                    if not success:
                        logger.warning(f"Failed to create index for document {document.id}")
            
            # Generate embeddings and store them
            results = await self.batch_generate_embeddings(
                documents=documents,
                db=db,
                storage_manager=storage_manager,
                progress_callback=progress_callback
            )
            
            # Update document processing status
            for doc_result in results["processed_documents"]:
                document = db.query(Document).filter(
                    Document.id == doc_result["document_id"]
                ).first()
                
                if document:
                    document.status = "completed"
                    document.processed_at = datetime.utcnow()
            
            # Mark failed documents
            for doc_result in results["failed_documents"]:
                document = db.query(Document).filter(
                    Document.id == doc_result["document_id"]
                ).first()
                
                if document:
                    document.status = "failed"
                    document.processing_error = doc_result["error"]
            
            db.commit()
            
            # Update results with indexing information
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            results["statistics"].update({
                "indexing_time_seconds": processing_time,
                "indices_created": len(documents) if create_indices else 0,
                "documents_indexed": len(results["processed_documents"])
            })
            
            logger.info(f"Batch indexing completed: {len(results['processed_documents'])} documents indexed")
            
            return results
            
        except Exception as e:
            db.rollback()
            logger.error(f"Batch indexing failed: {str(e)}")
            raise EmbeddingError(f"Batch indexing failed: {str(e)}")

    async def reindex_document(
        self,
        document_id: int,
        db: Session,
        storage_manager = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Re-index a single document, updating its embeddings.
        
        Args:
            document_id: ID of document to re-index
            db: Database session
            storage_manager: Vector storage manager
            force: Whether to force re-indexing even if already indexed
            
        Returns:
            Dictionary with re-indexing results
        """
        try:
            # Get document
            document = db.query(Document).filter(
                Document.id == document_id,
                Document.is_deleted == False
            ).first()
            
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Check if document needs re-indexing
            if not force and document.status == "completed":
                return {
                    "document_id": document_id,
                    "status": "skipped",
                    "reason": "already_indexed"
                }
            
            logger.info(f"Re-indexing document {document_id}")
            
            # Delete existing index if it exists
            if storage_manager:
                index_name = f"doc_{document_id}"
                await storage_manager.delete_index(index_name, db=db)
            
            # Re-index the document
            results = await self.batch_index_documents(
                document_ids=[document_id],
                db=db,
                storage_manager=storage_manager,
                create_indices=True
            )
            
            if results["statistics"]["indexed_documents"] > 0:
                return {
                    "document_id": document_id,
                    "status": "success",
                    "chunks_processed": results["processed_documents"][0]["chunks_processed"] if results["processed_documents"] else 0,
                    "processing_time": results["statistics"]["processing_time_seconds"]
                }
            else:
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": results["failed_documents"][0]["error"] if results["failed_documents"] else "Unknown error"
                }
                
        except Exception as e:
            logger.error(f"Failed to re-index document {document_id}: {str(e)}")
            raise EmbeddingError(f"Re-indexing failed: {str(e)}")

    def get_batch_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about batch processing capabilities."""
        return {
            "model_name": getattr(self, 'model_name', 'unknown'),
            "device": self.device,
            "batch_size": self.batch_size,
            "max_sequence_length": 512,
            "embedding_dimension": 768,  # Typical for transformer models
            "memory_efficient": True,
            "supports_gpu": torch.cuda.is_available(),
            "gpu_memory_mb": torch.cuda.get_device_properties(0).total_memory // (1024**2) if torch.cuda.is_available() else 0
        }