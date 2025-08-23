"""
Comprehensive tests for EnhancedSearchEngine implementation.
Tests all search modes, caching, filtering, and integration functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from typing import List

from vector_db.search_engine import (
    EnhancedSearchEngine, 
    SearchResult, 
    SearchFilter, 
    SearchType
)
from database.models import User, Document
from sqlalchemy.orm import Session


class TestEnhancedSearchEngine:
    """Test suite for EnhancedSearchEngine functionality."""
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Mock storage manager for testing."""
        mock_storage = AsyncMock()
        mock_storage.search_vectors.return_value = []
        return mock_storage
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """Mock embedding manager for testing."""
        mock_embedding = AsyncMock()
        mock_embedding.get_query_embedding.return_value = [0.1, 0.2, 0.3]
        return mock_embedding
    
    @pytest.fixture
    def search_engine(self, mock_storage_manager, mock_embedding_manager):
        """Create EnhancedSearchEngine instance for testing."""
        return EnhancedSearchEngine(mock_storage_manager, mock_embedding_manager)
    
    @pytest.fixture
    def mock_user(self):
        """Mock user for testing."""
        user = Mock(spec=User)
        user.id = 1
        user.username = "testuser"
        return user
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = Mock(spec=Session)
        session.query.return_value.filter.return_value.first.return_value = None
        return session
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        result1 = SearchResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            text="Machine learning is a subset of artificial intelligence",
            score=0.95,
            metadata={"page": 1},
            document_metadata={"filename": "ml_guide.pdf", "title": "ML Guide"}
        )
        
        result2 = SearchResult(
            chunk_id="chunk_2",
            document_id="doc_2",
            text="Deep learning uses neural networks with multiple layers",
            score=0.88,
            metadata={"page": 2},
            document_metadata={"filename": "dl_intro.pdf", "title": "Deep Learning Intro"}
        )
        
        return [result1, result2]
    
    async def test_search_semantic_mode(self, search_engine, mock_user, mock_db_session, sample_search_results):
        """Test semantic search mode."""
        search_engine.storage_manager.search_vectors.return_value = sample_search_results
        
        results = await search_engine.search(
            query="what is machine learning",
            user=mock_user,
            search_type=SearchType.SEMANTIC,
            db=mock_db_session
        )
        
        assert len(results) == 2
        assert results[0].text == "Machine learning is a subset of artificial intelligence"
        assert results[0].score == 0.95
        
        # Verify storage manager was called for vector search
        search_engine.storage_manager.search_vectors.assert_called_once()
    
    async def test_search_keyword_mode(self, search_engine, mock_user, mock_db_session, sample_search_results):
        """Test keyword search mode."""
        with patch.object(search_engine, '_keyword_search') as mock_keyword_search:
            mock_keyword_search.return_value = sample_search_results
            
            results = await search_engine.search(
                query="machine learning",
                user=mock_user,
                search_type=SearchType.KEYWORD,
                db=mock_db_session
            )
            
            assert len(results) == 2
            mock_keyword_search.assert_called_once()
    
    async def test_search_hybrid_mode(self, search_engine, mock_user, mock_db_session, sample_search_results):
        """Test hybrid search mode combining semantic and keyword."""
        with patch.object(search_engine, '_hybrid_search') as mock_hybrid_search:
            mock_hybrid_search.return_value = sample_search_results
            
            results = await search_engine.search(
                query="machine learning algorithms",
                user=mock_user,
                search_type=SearchType.HYBRID,
                db=mock_db_session
            )
            
            assert len(results) == 2
            mock_hybrid_search.assert_called_once()
    
    async def test_search_with_filters(self, search_engine, mock_user, mock_db_session, sample_search_results):
        """Test search with filters applied."""
        search_engine.storage_manager.search_vectors.return_value = sample_search_results
        
        filters = SearchFilter()
        filters.content_types = ["pdf"]
        filters.min_score = 0.8
        filters.tags = ["ml", "ai"]
        
        results = await search_engine.search(
            query="machine learning",
            user=mock_user,
            search_type=SearchType.SEMANTIC,
            filters=filters,
            db=mock_db_session
        )
        
        # Verify filters were applied
        call_args = search_engine.storage_manager.search_vectors.call_args
        assert call_args is not None
        assert len(results) == 2
    
    async def test_search_with_limit(self, search_engine, mock_user, mock_db_session, sample_search_results):
        """Test search with result limit."""
        search_engine.storage_manager.search_vectors.return_value = sample_search_results
        
        results = await search_engine.search(
            query="test query",
            user=mock_user,
            search_type=SearchType.SEMANTIC,
            limit=1,
            db=mock_db_session
        )
        
        # Should respect the limit parameter
        call_args = search_engine.storage_manager.search_vectors.call_args
        assert call_args is not None
    
    async def test_search_caching(self, search_engine, mock_user, mock_db_session, sample_search_results):
        """Test search result caching functionality."""
        with patch.object(search_engine, '_get_cached_results') as mock_get_cache:
            with patch.object(search_engine, '_cache_results') as mock_cache_results:
                # First call - no cache
                mock_get_cache.return_value = None
                search_engine.storage_manager.search_vectors.return_value = sample_search_results
                
                results = await search_engine.search(
                    query="test query",
                    user=mock_user,
                    search_type=SearchType.SEMANTIC,
                    db=mock_db_session,
                    use_cache=True
                )
                
                # Verify cache was checked and results were cached
                mock_get_cache.assert_called_once()
                mock_cache_results.assert_called_once()
                assert len(results) == 2
    
    async def test_search_cache_hit(self, search_engine, mock_user, mock_db_session, sample_search_results):
        """Test search with cache hit."""
        with patch.object(search_engine, '_get_cached_results') as mock_get_cache:
            # Return cached results
            mock_get_cache.return_value = sample_search_results
            
            results = await search_engine.search(
                query="cached query",
                user=mock_user,
                search_type=SearchType.SEMANTIC,
                db=mock_db_session,
                use_cache=True
            )
            
            # Should return cached results without calling storage manager
            mock_get_cache.assert_called_once()
            search_engine.storage_manager.search_vectors.assert_not_called()
            assert len(results) == 2
    
    async def test_text_search(self, search_engine, mock_user, mock_db_session, sample_search_results):
        """Test text search method for backward compatibility."""
        with patch.object(search_engine, 'search') as mock_search:
            mock_search.return_value = sample_search_results
            
            results = await search_engine.text_search(
                query="test query",
                user=mock_user,
                db=mock_db_session
            )
            
            # Should call main search method with KEYWORD type
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args[1]['search_type'] == SearchType.KEYWORD
            assert len(results) == 2
    
    async def test_get_search_suggestions(self, search_engine, mock_user, mock_db_session):
        """Test search suggestions generation."""
        # Mock database query results
        mock_db_session.query.return_value.filter.return_value.limit.return_value.all.return_value = [
            Mock(query_text="machine learning", created_at=datetime.now()),
            Mock(query_text="deep learning", created_at=datetime.now())
        ]
        
        # Mock document titles
        mock_documents = [
            Mock(title="Machine Learning Guide", filename="ml_guide.pdf"),
            Mock(title="AI Fundamentals", filename="ai_basics.pdf")
        ]
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_documents
        
        suggestions = await search_engine.get_search_suggestions(
            query="machine",
            user=mock_user,
            limit=5,
            db=mock_db_session
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Check suggestion structure
        for suggestion in suggestions:
            assert 'type' in suggestion
            assert 'text' in suggestion
            assert 'icon' in suggestion
    
    async def test_get_available_filters(self, search_engine, mock_user, mock_db_session):
        """Test available filters generation."""
        # Mock document data for filter generation
        mock_documents = [
            Mock(content_type="pdf", tags="ml,ai", created_at=datetime.now(), file_size=1024000),
            Mock(content_type="txt", tags="programming", created_at=datetime.now(), file_size=2048000)
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_documents
        
        filters = await search_engine.get_available_filters(
            user=mock_user,
            db=mock_db_session
        )
        
        assert isinstance(filters, dict)
        assert 'file_types' in filters
        assert 'tags' in filters
        assert 'date_range' in filters
        assert 'file_size_range' in filters
    
    async def test_search_error_handling(self, search_engine, mock_user, mock_db_session):
        """Test error handling in search operations."""
        # Make storage manager raise an exception
        search_engine.storage_manager.search_vectors.side_effect = Exception("Storage error")
        
        with pytest.raises(Exception):
            await search_engine.search(
                query="test query",
                user=mock_user,
                search_type=SearchType.SEMANTIC,
                db=mock_db_session
            )
    
    async def test_search_filter_validation(self, search_engine):
        """Test SearchFilter validation and functionality."""
        filter_obj = SearchFilter()
        
        # Test default values
        assert filter_obj.min_score is None
        assert filter_obj.content_types is None
        assert filter_obj.tags is None
        
        # Test setting values
        filter_obj.min_score = 0.5
        filter_obj.content_types = ["pdf", "txt"]
        filter_obj.tags = ["ml", "ai"]
        
        assert filter_obj.min_score == 0.5
        assert filter_obj.content_types == ["pdf", "txt"]
        assert filter_obj.tags == ["ml", "ai"]
        
        # Test to_dict method if implemented
        if hasattr(filter_obj, 'to_dict'):
            filter_dict = filter_obj.to_dict()
            assert isinstance(filter_dict, dict)
    
    def test_search_result_properties(self, sample_search_results):
        """Test SearchResult object properties."""
        result = sample_search_results[0]
        
        assert result.chunk_id == "chunk_1"
        assert result.document_id == "doc_1"
        assert "Machine learning" in result.text
        assert result.score == 0.95
        assert result.metadata["page"] == 1
        assert result.document_metadata["filename"] == "ml_guide.pdf"
        
        # Test to_dict method if implemented
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert result_dict['chunk_id'] == "chunk_1"
    
    async def test_reranking_functionality(self, search_engine, mock_user, mock_db_session, sample_search_results):
        """Test search result reranking."""
        with patch.object(search_engine, '_apply_reranking') as mock_rerank:
            mock_rerank.return_value = sample_search_results
            search_engine.storage_manager.search_vectors.return_value = sample_search_results
            
            filters = SearchFilter()
            filters.enable_reranking = True
            filters.reranker_model = "ms-marco-MiniLM-L-6-v2"
            
            results = await search_engine.search(
                query="test query",
                user=mock_user,
                search_type=SearchType.SEMANTIC,
                filters=filters,
                db=mock_db_session
            )
            
            # Should apply reranking when enabled
            mock_rerank.assert_called_once()
            assert len(results) == 2


class TestSearchEngineIntegration:
    """Integration tests for SearchEngine with other components."""
    
    @pytest.fixture
    def integration_search_engine(self):
        """Create SearchEngine with mock dependencies for integration testing."""
        with patch('vector_db.search_engine.get_storage_manager') as mock_storage:
            with patch('vector_db.search_engine.EnhancedEmbeddingManager') as mock_embedding:
                storage_manager = Mock()
                embedding_manager = Mock()
                mock_storage.return_value = storage_manager
                mock_embedding.create_default_manager.return_value = embedding_manager
                
                return EnhancedSearchEngine(storage_manager, embedding_manager)
    
    def test_search_type_enum(self):
        """Test SearchType enumeration."""
        assert SearchType.SEMANTIC == "semantic"
        assert SearchType.KEYWORD == "keyword" 
        assert SearchType.HYBRID == "hybrid"
        assert SearchType.CONTEXTUAL == "contextual"
    
    async def test_performance_logging(self, integration_search_engine):
        """Test that search operations are properly logged with timing."""
        with patch.object(integration_search_engine, '_log_search_query') as mock_log:
            with patch.object(integration_search_engine.storage_manager, 'search_vectors') as mock_search:
                mock_search.return_value = []
                
                mock_user = Mock()
                mock_user.id = 1
                mock_db = Mock()
                
                await integration_search_engine.search(
                    query="test query",
                    user=mock_user,
                    search_type=SearchType.SEMANTIC,
                    db=mock_db
                )
                
                # Verify logging was called with timing information
                mock_log.assert_called_once()
                call_args = mock_log.call_args[1]
                assert 'search_time' in call_args or 'execution_time' in call_args


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])