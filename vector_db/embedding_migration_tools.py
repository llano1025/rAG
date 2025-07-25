# vector_db/embedding_migration_tools.py

from typing import List, Dict, Any, Optional, Callable
import logging
import asyncio
from datetime import datetime
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .enhanced_embedding_manager import EnhancedEmbeddingManager
from .embedding_manager import EmbeddingManager
from .embedding_model_registry import EmbeddingModelRegistry, get_embedding_model_registry, EmbeddingProvider
from database.models import Document, DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class MigrationPlan:
    """Plan for embedding model migration."""
    source_model_id: str
    target_model_id: str
    document_ids: List[int]
    estimated_time_minutes: float
    estimated_cost: Optional[float] = None
    backup_required: bool = True
    
class EmbeddingMigrationError(Exception):
    """Raised when embedding migration fails."""
    pass

class EmbeddingMigrationTools:
    """Tools for migrating between embedding models."""
    
    def __init__(self, storage_manager=None, registry: Optional[EmbeddingModelRegistry] = None):
        self.storage_manager = storage_manager
        self.registry = registry or get_embedding_model_registry()
        
    async def create_migration_plan(
        self,
        source_model_id: str,
        target_model_id: str,
        document_ids: List[int],
        db: Session
    ) -> MigrationPlan:
        """Create a migration plan for switching embedding models."""
        try:
            # Validate models exist
            source_model = self.registry.get_model(source_model_id)
            target_model = self.registry.get_model(target_model_id)
            
            if not source_model:
                raise EmbeddingMigrationError(f"Source model {source_model_id} not found in registry")
            if not target_model:
                raise EmbeddingMigrationError(f"Target model {target_model_id} not found in registry")
            
            # Get documents and calculate statistics
            documents = db.query(Document).filter(
                Document.id.in_(document_ids),
                Document.is_deleted == False
            ).all()
            
            if not documents:
                raise EmbeddingMigrationError("No valid documents found for migration")
            
            # Calculate total chunks to migrate
            total_chunks = 0
            for document in documents:
                chunk_count = db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document.id
                ).count()
                total_chunks += chunk_count
            
            # Estimate processing time based on model performance
            source_metrics = self.registry.get_performance_metrics(source_model_id)
            target_metrics = self.registry.get_performance_metrics(target_model_id)
            
            # Use average processing time or fallback to model tier estimates
            processing_time_per_chunk = 100  # Default 100ms per chunk
            if target_metrics and target_metrics.average_latency_ms > 0:
                processing_time_per_chunk = target_metrics.average_latency_ms
            elif target_model.performance_tier == "fast":
                processing_time_per_chunk = 50
            elif target_model.performance_tier == "quality":
                processing_time_per_chunk = 200
            
            estimated_time_minutes = (total_chunks * processing_time_per_chunk) / (1000 * 60)
            
            # Estimate cost for API-based models
            estimated_cost = None
            if target_model.api_cost_per_1k_tokens:
                # Rough estimate: 100 tokens per chunk on average
                estimated_tokens = total_chunks * 100
                estimated_cost = (estimated_tokens / 1000) * target_model.api_cost_per_1k_tokens
            
            plan = MigrationPlan(
                source_model_id=source_model_id,
                target_model_id=target_model_id,
                document_ids=document_ids,
                estimated_time_minutes=estimated_time_minutes,
                estimated_cost=estimated_cost,
                backup_required=True
            )
            
            logger.info(f"Created migration plan: {len(documents)} documents, {total_chunks} chunks, ~{estimated_time_minutes:.1f} minutes")
            return plan
            
        except Exception as e:
            raise EmbeddingMigrationError(f"Failed to create migration plan: {str(e)}")

    async def execute_migration(
        self,
        plan: MigrationPlan,
        db: Session,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        backup_before_migration: bool = True
    ) -> Dict[str, Any]:
        """Execute the embedding migration plan."""
        try:
            start_time = datetime.utcnow()
            logger.info(f"Starting migration from {plan.source_model_id} to {plan.target_model_id}")
            
            # Get target model and create manager
            target_model = self.registry.get_model(plan.target_model_id)
            if not target_model:
                raise EmbeddingMigrationError(f"Target model {plan.target_model_id} not found")
            
            # Create target embedding manager
            target_manager = await self._create_embedding_manager(target_model)
            
            # Get documents to migrate
            documents = db.query(Document).filter(
                Document.id.in_(plan.document_ids),
                Document.is_deleted == False
            ).all()
            
            migration_results = {
                "plan": plan,
                "migrated_documents": [],
                "failed_documents": [],
                "backup_info": None,
                "statistics": {}
            }
            
            # Create backup if requested
            if backup_before_migration and self.storage_manager:
                if progress_callback:
                    progress_callback(0, len(documents), "Creating backup...")
                
                backup_info = await self._create_migration_backup(
                    documents, plan.source_model_id, db
                )
                migration_results["backup_info"] = backup_info
            
            # Migrate each document
            migrated_count = 0
            failed_count = 0
            total_chunks_migrated = 0
            
            for i, document in enumerate(documents):
                try:
                    if progress_callback:
                        progress_callback(
                            i, len(documents), 
                            f"Migrating document {document.filename}"
                        )
                    
                    # Get document chunks
                    chunks = db.query(DocumentChunk).filter(
                        DocumentChunk.document_id == document.id
                    ).all()
                    
                    if not chunks:
                        logger.warning(f"No chunks found for document {document.id}")
                        continue
                    
                    # Delete old index
                    if self.storage_manager:
                        old_index_name = f"doc_{document.id}"
                        await self.storage_manager.delete_index(old_index_name, db=db)
                    
                    # Generate new embeddings
                    content_texts = [chunk.text for chunk in chunks]
                    context_texts = []
                    
                    for chunk in chunks:
                        context_parts = []
                        if chunk.context_before:
                            context_parts.append(chunk.context_before)
                        context_parts.append(chunk.text)
                        if chunk.context_after:
                            context_parts.append(chunk.context_after)
                        context_texts.append(" ".join(context_parts))
                    
                    # Track processing time for metrics
                    processing_start = datetime.utcnow()
                    
                    content_embeddings = await target_manager.generate_embeddings(content_texts)
                    context_embeddings = await target_manager.generate_embeddings(context_texts)
                    
                    processing_end = datetime.utcnow()
                    processing_time_ms = (processing_end - processing_start).total_seconds() * 1000
                    
                    # Update performance metrics
                    self.registry.update_performance_metrics(
                        plan.target_model_id,
                        processing_time_ms / len(chunks),  # Average per chunk
                        success=True
                    )
                    
                    # Store new embeddings
                    if self.storage_manager:
                        new_index_name = f"doc_{document.id}"
                        chunk_ids = [chunk.chunk_id for chunk in chunks]
                        
                        # Create new index with correct dimension
                        await self.storage_manager.create_index(
                            index_name=new_index_name,
                            embedding_dimension=target_manager.get_embedding_dimension(),
                            faiss_index_type="HNSW",
                            user_id=document.user_id,
                            document_id=document.id,
                            db=db
                        )
                        
                        # Prepare metadata
                        metadata_list = []
                        for chunk in chunks:
                            metadata_list.append({
                                "chunk_id": chunk.chunk_id,
                                "document_id": document.id,
                                "chunk_index": chunk.chunk_index,
                                "text_length": chunk.text_length,
                                "page_number": chunk.page_number,
                                "section_title": chunk.section_title,
                                "embedding_model": target_model.model_name,
                                "migration_timestamp": datetime.utcnow().isoformat()
                            })
                        
                        # Add vectors
                        await self.storage_manager.add_vectors(
                            index_name=new_index_name,
                            content_vectors=content_embeddings,
                            context_vectors=context_embeddings,
                            metadata_list=metadata_list,
                            chunk_ids=chunk_ids
                        )
                    
                    # Update document status
                    document.status = "completed"
                    document.processed_at = datetime.utcnow()
                    
                    migrated_count += 1
                    total_chunks_migrated += len(chunks)
                    
                    migration_results["migrated_documents"].append({
                        "document_id": document.id,
                        "filename": document.filename,
                        "chunks_migrated": len(chunks),
                        "processing_time_ms": processing_time_ms
                    })
                    
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Failed to migrate document {document.id}: {str(e)}"
                    logger.error(error_msg)
                    
                    # Update performance metrics for failure
                    self.registry.update_performance_metrics(
                        plan.target_model_id,
                        0,  # No processing time for failures
                        success=False
                    )
                    
                    migration_results["failed_documents"].append({
                        "document_id": document.id,
                        "filename": document.filename,
                        "error": str(e)
                    })
                    
                    # Mark document as failed
                    document.status = "failed"
                    document.processing_error = str(e)
            
            # Commit database changes
            db.commit()
            
            # Calculate final statistics
            end_time = datetime.utcnow()
            total_time = (end_time - start_time).total_seconds()
            
            migration_results["statistics"] = {
                "total_documents": len(documents),
                "migrated_documents": migrated_count,
                "failed_documents": failed_count,
                "total_chunks_migrated": total_chunks_migrated,
                "migration_time_seconds": total_time,
                "migration_time_minutes": total_time / 60,
                "chunks_per_second": total_chunks_migrated / total_time if total_time > 0 else 0,
                "source_model": plan.source_model_id,
                "target_model": plan.target_model_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            if progress_callback:
                progress_callback(len(documents), len(documents), "Migration completed")
            
            logger.info(f"Migration completed: {migrated_count}/{len(documents)} documents migrated in {total_time:.2f}s")
            return migration_results
            
        except Exception as e:
            db.rollback()
            logger.error(f"Migration failed: {str(e)}")
            raise EmbeddingMigrationError(f"Migration execution failed: {str(e)}")

    async def _create_embedding_manager(self, model_metadata) -> EnhancedEmbeddingManager:
        """Create appropriate embedding manager for a model."""
        if model_metadata.provider == EmbeddingProvider.HUGGINGFACE:
            return EnhancedEmbeddingManager.create_huggingface_manager(
                model_name=model_metadata.model_name
            )
        elif model_metadata.provider == EmbeddingProvider.OLLAMA:
            return EnhancedEmbeddingManager.create_ollama_manager(
                model_name=model_metadata.model_name
            )
        elif model_metadata.provider == EmbeddingProvider.OPENAI:
            from config import get_settings
            settings = get_settings()
            if not settings.OPENAI_API_KEY:
                raise EmbeddingMigrationError("OpenAI API key not configured")
            return EnhancedEmbeddingManager.create_openai_manager(
                api_key=settings.OPENAI_API_KEY,
                model_name=model_metadata.model_name
            )
        else:
            raise EmbeddingMigrationError(f"Unsupported provider: {model_metadata.provider}")

    async def _create_migration_backup(
        self,
        documents: List[Document],
        source_model_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Create backup before migration."""
        try:
            backup_info = {
                "backup_id": f"migration_backup_{source_model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "source_model_id": source_model_id,
                "document_count": len(documents),
                "created_at": datetime.utcnow().isoformat(),
                "indices_backed_up": []
            }
            
            # For each document, backup its vector index
            for document in documents:
                if self.storage_manager:
                    index_name = f"doc_{document.id}"
                    backup_success = await self.storage_manager.backup_index(
                        index_name, backup_info["backup_id"], db=db
                    )
                    
                    if backup_success:
                        backup_info["indices_backed_up"].append(index_name)
            
            logger.info(f"Created migration backup: {backup_info['backup_id']}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Failed to create migration backup: {str(e)}")
            raise EmbeddingMigrationError(f"Backup creation failed: {str(e)}")

    async def rollback_migration(
        self,
        backup_id: str,
        document_ids: List[int],
        db: Session
    ) -> Dict[str, Any]:
        """Rollback a migration using backup."""
        try:
            logger.info(f"Starting migration rollback for backup {backup_id}")
            
            rollback_results = {
                "backup_id": backup_id,
                "restored_documents": [],
                "failed_documents": [],
                "statistics": {}
            }
            
            start_time = datetime.utcnow()
            restored_count = 0
            failed_count = 0
            
            for document_id in document_ids:
                try:
                    if self.storage_manager:
                        index_name = f"doc_{document_id}"
                        restore_success = await self.storage_manager.restore_index(
                            index_name, backup_id, db=db
                        )
                        
                        if restore_success:
                            restored_count += 1
                            rollback_results["restored_documents"].append({
                                "document_id": document_id,
                                "index_name": index_name
                            })
                        else:
                            failed_count += 1
                            rollback_results["failed_documents"].append({
                                "document_id": document_id,
                                "error": "Failed to restore index"
                            })
                
                except Exception as e:
                    failed_count += 1
                    rollback_results["failed_documents"].append({
                        "document_id": document_id,
                        "error": str(e)
                    })
            
            end_time = datetime.utcnow()
            total_time = (end_time - start_time).total_seconds()
            
            rollback_results["statistics"] = {
                "total_documents": len(document_ids),
                "restored_documents": restored_count,
                "failed_documents": failed_count,
                "rollback_time_seconds": total_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            logger.info(f"Rollback completed: {restored_count}/{len(document_ids)} documents restored")
            return rollback_results
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            raise EmbeddingMigrationError(f"Rollback failed: {str(e)}")

    async def validate_migration(
        self,
        document_ids: List[int],
        target_model_id: str,
        db: Session,
        sample_size: int = 10
    ) -> Dict[str, Any]:
        """Validate migration by testing embeddings and search."""
        try:
            target_model = self.registry.get_model(target_model_id)
            if not target_model:
                raise EmbeddingMigrationError(f"Target model {target_model_id} not found")
            
            validation_results = {
                "model_id": target_model_id,
                "validation_tests": [],
                "overall_status": "passed",
                "statistics": {}
            }
            
            # Sample documents for validation
            documents = db.query(Document).filter(
                Document.id.in_(document_ids[:sample_size]),
                Document.is_deleted == False
            ).all()
            
            target_manager = await self._create_embedding_manager(target_model)
            
            passed_tests = 0
            total_tests = 0
            
            for document in documents:
                try:
                    # Test 1: Check if embeddings exist and have correct dimensions
                    if self.storage_manager:
                        index_name = f"doc_{document.id}"
                        index_info = await self.storage_manager.get_index_info(index_name, db=db)
                        
                        expected_dim = target_manager.get_embedding_dimension()
                        actual_dim = index_info.get("embedding_dimension", 0)
                        
                        test_result = {
                            "document_id": document.id,
                            "test_type": "dimension_check",
                            "expected_dimension": expected_dim,
                            "actual_dimension": actual_dim,
                            "passed": expected_dim == actual_dim
                        }
                        
                        validation_results["validation_tests"].append(test_result)
                        total_tests += 1
                        if test_result["passed"]:
                            passed_tests += 1
                    
                    # Test 2: Search functionality
                    sample_chunks = db.query(DocumentChunk).filter(
                        DocumentChunk.document_id == document.id
                    ).limit(3).all()
                    
                    if sample_chunks:
                        sample_text = sample_chunks[0].text[:100]  # Use first 100 chars as query
                        
                        # Perform search using the new embeddings
                        if self.storage_manager:
                            query_embedding, _ = await target_manager.generate_query_embeddings(sample_text)
                            
                            search_results = await self.storage_manager.search_similar(
                                index_name=f"doc_{document.id}",
                                query_vector=query_embedding,
                                top_k=5,
                                db=db
                            )
                            
                            test_result = {
                                "document_id": document.id,
                                "test_type": "search_functionality",
                                "query_text": sample_text,
                                "results_count": len(search_results),
                                "passed": len(search_results) > 0
                            }
                            
                            validation_results["validation_tests"].append(test_result)
                            total_tests += 1
                            if test_result["passed"]:
                                passed_tests += 1
                    
                except Exception as e:
                    test_result = {
                        "document_id": document.id,
                        "test_type": "error",
                        "error": str(e),
                        "passed": False
                    }
                    validation_results["validation_tests"].append(test_result)
                    total_tests += 1
            
            # Calculate overall status
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            validation_results["overall_status"] = "passed" if success_rate >= 0.8 else "failed"
            
            validation_results["statistics"] = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "documents_validated": len(documents)
            }
            
            logger.info(f"Migration validation completed: {passed_tests}/{total_tests} tests passed")
            return validation_results
            
        except Exception as e:
            logger.error(f"Migration validation failed: {str(e)}")
            raise EmbeddingMigrationError(f"Validation failed: {str(e)}")

    async def get_migration_status(self, document_ids: List[int], db: Session) -> Dict[str, Any]:
        """Get status of documents and their embedding models."""
        try:
            documents = db.query(Document).filter(
                Document.id.in_(document_ids)
            ).all()
            
            status_info = {
                "documents": [],
                "model_distribution": {},
                "total_documents": len(documents),
                "last_updated": datetime.utcnow().isoformat()
            }
            
            for document in documents:
                doc_info = {
                    "document_id": document.id,
                    "filename": document.filename,
                    "status": document.status,
                    "processed_at": document.processed_at.isoformat() if document.processed_at else None,
                    "embedding_model": "unknown",
                    "chunks_count": 0
                }
                
                # Get chunk count
                chunk_count = db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document.id
                ).count()
                doc_info["chunks_count"] = chunk_count
                
                # Try to determine embedding model from metadata
                if self.storage_manager:
                    try:
                        index_name = f"doc_{document.id}"
                        index_info = await self.storage_manager.get_index_info(index_name, db=db)
                        if index_info and "metadata" in index_info:
                            doc_info["embedding_model"] = index_info["metadata"].get("embedding_model", "unknown")
                    except:
                        pass
                
                status_info["documents"].append(doc_info)
                
                # Update model distribution
                model = doc_info["embedding_model"]
                status_info["model_distribution"][model] = status_info["model_distribution"].get(model, 0) + 1
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {str(e)}")
            raise EmbeddingMigrationError(f"Status check failed: {str(e)}")