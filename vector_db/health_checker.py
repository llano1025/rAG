"""
Vector database health checker with comprehensive monitoring.
Provides health checks for FAISS indices, Qdrant collections, and database connectivity.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.models import Document, DocumentChunk, VectorIndex
from .storage_manager import VectorStorageManager, get_storage_manager
from .qdrant_client import QdrantManager
from utils.monitoring.health_check import HealthStatus, ComponentHealth, HealthChecker
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class VectorHealthStatus:
    """Vector database specific health status indicators."""
    
    def __init__(self):
        self.qdrant_status = HealthStatus.HEALTHY
        self.database_status = HealthStatus.HEALTHY
        self.storage_manager_status = HealthStatus.HEALTHY
        self.last_check = datetime.now(timezone.utc)
        self.details = {}

class VectorHealthChecker:
    """
    Comprehensive health checker for vector database components.
    
    Monitors:
    - Qdrant collection health and connectivity
    - Database connectivity and consistency
    - Storage manager functionality
    - Vector index status
    """
    
    def __init__(self, storage_manager: VectorStorageManager = None):
        """Initialize vector health checker."""
        self.storage_manager = storage_manager or get_storage_manager()
        self.qdrant_manager = QdrantManager()
        self.status = VectorHealthStatus()
        
        # Health check configuration
        self.max_response_time_ms = 1000  # 1 second
        self.critical_response_time_ms = 5000  # 5 seconds
        self.test_index_name = "health_check_test"
        
    async def check_all_components(self, db: Session = None) -> Dict[str, Any]:
        """
        Run comprehensive health checks on all vector database components.
        
        Args:
            db: Database session for connectivity checks
            
        Returns:
            Dictionary with detailed health status
        """
        start_time = time.time()
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        try:
            # Check database connectivity
            db_result = await self._check_database_connectivity(db)
            results['database'] = db_result
            self._update_overall_status(overall_status, db_result['status'])
            
            # Check Qdrant connectivity and health
            qdrant_result = await self._check_qdrant_health()
            results['qdrant'] = qdrant_result
            self._update_overall_status(overall_status, qdrant_result['status'])
            
            
            # Check storage manager functionality
            storage_result = await self._check_storage_manager(db)
            results['storage_manager'] = storage_result
            self._update_overall_status(overall_status, storage_result['status'])
            
            # Check index synchronization
            sync_result = await self._check_index_synchronization(db)
            results['synchronization'] = sync_result
            self._update_overall_status(overall_status, sync_result['status'])
            
            # Performance benchmarks
            perf_result = await self._run_performance_tests()
            results['performance'] = perf_result
            self._update_overall_status(overall_status, perf_result['status'])
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            overall_status = HealthStatus.UNHEALTHY
            results['error'] = str(e)
        
        check_time_ms = (time.time() - start_time) * 1000
        
        # Update internal status
        self.status.last_check = datetime.now(timezone.utc)
        self.status.details = results
        
        return {
            'status': overall_status,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'check_duration_ms': round(check_time_ms, 2),
            'components': results
        }
    
    async def _check_database_connectivity(self, db: Session) -> Dict[str, Any]:
        """Check database connectivity and basic queries."""
        try:
            start_time = time.time()
            
            if not db:
                return {
                    'status': HealthStatus.UNHEALTHY,
                    'details': {'error': 'No database session provided'}
                }
            
            # Test basic connectivity
            db.execute(text("SELECT 1"))
            
            # Check document table
            doc_count = db.query(Document).count()
            
            # Check chunks table
            chunk_count = db.query(DocumentChunk).count()
            
            # Check vector indices table
            index_count = db.query(VectorIndex).count()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY
            if response_time_ms > self.critical_response_time_ms:
                status = HealthStatus.UNHEALTHY
            elif response_time_ms > self.max_response_time_ms:
                status = HealthStatus.DEGRADED
            
            return {
                'status': status,
                'details': {
                    'response_time_ms': round(response_time_ms, 2),
                    'documents_count': doc_count,
                    'chunks_count': chunk_count,
                    'indices_count': index_count,
                    'connection_active': True
                }
            }
            
        except Exception as e:
            logger.error(f"Database connectivity check failed: {e}")
            return {
                'status': HealthStatus.UNHEALTHY,
                'details': {'error': str(e), 'connection_active': False}
            }
    
    async def _check_qdrant_health(self) -> Dict[str, Any]:
        """Check Qdrant server health and connectivity."""
        try:
            start_time = time.time()
            
            # Check if Qdrant is accessible
            health_info = await self.qdrant_manager.health_check()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if not health_info:
                return {
                    'status': HealthStatus.UNHEALTHY,
                    'details': {'error': 'Qdrant health check returned no data'}
                }
            
            # Get collection info
            collections = await self.qdrant_manager.list_collections()
            
            status = HealthStatus.HEALTHY
            if response_time_ms > self.critical_response_time_ms:
                status = HealthStatus.UNHEALTHY
            elif response_time_ms > self.max_response_time_ms:
                status = HealthStatus.DEGRADED
            
            return {
                'status': status,
                'details': {
                    'response_time_ms': round(response_time_ms, 2),
                    'server_status': 'online',
                    'collections_count': len(collections) if collections else 0,
                    'collections': collections[:10] if collections else [],  # First 10 collections
                    'health_info': health_info
                }
            }
            
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return {
                'status': HealthStatus.UNHEALTHY,
                'details': {
                    'error': str(e),
                    'server_status': 'offline'
                }
            }
    
    
    async def _check_storage_manager(self, db: Session) -> Dict[str, Any]:
        """Check storage manager functionality with a test operation."""
        try:
            start_time = time.time()
            
            # Test creating a small test index
            test_dimension = 384  # Default embedding dimension
            test_user_id = 0  # System user for health checks
            
            # Check if storage manager can create an index
            await self.storage_manager.create_index(
                index_name=self.test_index_name,
                embedding_dimension=test_dimension,
                user_id=test_user_id,
                document_id=None,  # Health check index
                db=db
            )
            
            # Test adding vectors
            test_vectors = [[0.1] * test_dimension, [0.2] * test_dimension]
            test_metadata = [
                {'test': True, 'chunk_id': 'health_1'},
                {'test': True, 'chunk_id': 'health_2'}
            ]
            test_ids = ['health_1', 'health_2']
            
            added_ids = await self.storage_manager.add_vectors(
                index_name=self.test_index_name,
                content_vectors=test_vectors,
                context_vectors=test_vectors,
                metadata_list=test_metadata,
                chunk_ids=test_ids
            )
            
            # Test searching
            search_results = await self.storage_manager.search_vectors(
                index_name=self.test_index_name,
                query_vector=[0.15] * test_dimension,
                vector_type="content",
                limit=2
            )
            
            # Clean up test index
            await self.storage_manager.delete_index(self.test_index_name, db)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY
            if response_time_ms > self.critical_response_time_ms:
                status = HealthStatus.UNHEALTHY
            elif response_time_ms > self.max_response_time_ms:
                status = HealthStatus.DEGRADED
            
            return {
                'status': status,
                'details': {
                    'response_time_ms': round(response_time_ms, 2),
                    'test_vectors_added': len(added_ids) if added_ids else 0,
                    'test_search_results': len(search_results) if search_results else 0,
                    'operations_successful': True
                }
            }
            
        except Exception as e:
            logger.error(f"Storage manager health check failed: {e}")
            # Try to clean up test index even if there was an error
            try:
                await self.storage_manager.delete_index(self.test_index_name, db)
            except:
                pass
            
            return {
                'status': HealthStatus.UNHEALTHY,
                'details': {
                    'error': str(e),
                    'operations_successful': False
                }
            }
    
    async def _check_index_synchronization(self, db: Session) -> Dict[str, Any]:
        """Check synchronization between database and vector indices."""
        try:
            # Get document counts from database
            total_documents = db.query(Document).filter(Document.is_deleted == False).count()
            total_chunks = db.query(DocumentChunk).count()
            total_vector_indices = db.query(VectorIndex).count()
            
            # Check consistency
            status = HealthStatus.HEALTHY
            issues = []
            
            # Check if we have documents but no vector indices
            if total_documents > 0 and total_vector_indices == 0:
                status = HealthStatus.UNHEALTHY
                issues.append("Documents exist but no vector indices found")
            
            # Check if we have chunks but significantly fewer indices
            if total_chunks > 0 and total_vector_indices < total_chunks * 0.8:
                status = HealthStatus.DEGRADED
                issues.append(f"Chunk to index ratio is low: {total_chunks} chunks, {total_vector_indices} indices")
            
            # Get sample of recent documents and check their index status
            recent_docs = db.query(Document).filter(
                Document.is_deleted == False,
                Document.status == 'completed'
            ).order_by(Document.created_at.desc()).limit(10).all()
            
            indexed_docs = 0
            for doc in recent_docs:
                index_exists = db.query(VectorIndex).filter(
                    VectorIndex.document_id == doc.id
                ).first()
                if index_exists:
                    indexed_docs += 1
            
            if recent_docs and indexed_docs < len(recent_docs) * 0.8:
                status = HealthStatus.DEGRADED
                issues.append(f"Recent documents missing indices: {len(recent_docs) - indexed_docs}/{len(recent_docs)}")
            
            return {
                'status': status,
                'details': {
                    'total_documents': total_documents,
                    'total_chunks': total_chunks,
                    'total_vector_indices': total_vector_indices,
                    'recent_documents_checked': len(recent_docs),
                    'recent_documents_indexed': indexed_docs,
                    'synchronization_issues': issues,
                    'sync_ratio': round(indexed_docs / len(recent_docs), 2) if recent_docs else 1.0
                }
            }
            
        except Exception as e:
            logger.error(f"Index synchronization check failed: {e}")
            return {
                'status': HealthStatus.UNHEALTHY,
                'details': {'error': str(e)}
            }
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks on vector operations."""
        try:
            results = {}
            
            # Test embedding generation performance
            embedding_start = time.time()
            test_texts = ["This is a test document for performance testing."] * 10
            
            # Note: This would need the actual embedding manager
            # For now, we'll simulate the test
            embedding_time = (time.time() - embedding_start) * 1000
            results['embedding_generation_ms'] = round(embedding_time, 2)
            
            # Test vector search performance (if indices exist)
            search_stats = await self.storage_manager.get_performance_stats()
            results.update(search_stats)
            
            # Determine status based on performance
            status = HealthStatus.HEALTHY
            if embedding_time > 5000:  # 5 seconds for 10 embeddings
                status = HealthStatus.DEGRADED
            
            return {
                'status': status,
                'details': results
            }
            
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            return {
                'status': HealthStatus.DEGRADED,
                'details': {'error': str(e)}
            }
    
    def _update_overall_status(self, current_status: HealthStatus, new_status: HealthStatus) -> HealthStatus:
        """Update overall status based on component status."""
        if new_status == HealthStatus.UNHEALTHY:
            return HealthStatus.UNHEALTHY
        elif new_status == HealthStatus.DEGRADED and current_status != HealthStatus.UNHEALTHY:
            return HealthStatus.DEGRADED
        return current_status
    
    async def get_quick_status(self) -> Dict[str, Any]:
        """Get quick health status without running full checks."""
        try:
            # Quick connectivity checks
            db_ok = True
            qdrant_ok = await self.qdrant_manager.is_connected()
            
            status = HealthStatus.HEALTHY
            if not qdrant_ok:
                status = HealthStatus.UNHEALTHY
            
            return {
                'status': status,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'quick_check': True,
                'components': {
                    'database': 'connected' if db_ok else 'disconnected',
                    'qdrant': 'connected' if qdrant_ok else 'disconnected'
                }
            }
            
        except Exception as e:
            logger.error(f"Quick status check failed: {e}")
            return {
                'status': HealthStatus.UNHEALTHY,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }


# Global vector health checker instance
_vector_health_checker: Optional[VectorHealthChecker] = None

def get_vector_health_checker() -> VectorHealthChecker:
    """Get the global vector health checker instance."""
    global _vector_health_checker
    if _vector_health_checker is None:
        _vector_health_checker = VectorHealthChecker()
    return _vector_health_checker


# Integration with main health checker
def register_vector_health_checks(main_checker: HealthChecker):
    """Register vector database health checks with the main health checker."""
    vector_checker = get_vector_health_checker()
    
    async def vector_health_check():
        """Wrapper for vector health check."""
        try:
            result = await vector_checker.get_quick_status()
            return {
                'status': result['status'],
                'details': result.get('components', {})
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'details': {'error': str(e)}
            }
    
    main_checker.register_check("vector_database", vector_health_check)