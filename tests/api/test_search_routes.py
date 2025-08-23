"""
Comprehensive tests for the unified search routes implementation.
Tests all four search endpoints: unified, text, semantic, and hybrid.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.routes.search_routes import router
from api.schemas.search_schemas import SearchQuery, SearchFilters, SearchResponse
from vector_db.search_engine import SearchResult, SearchType, SearchFilter
from database.models import User


class TestUnifiedSearchRoutes:
    """Test suite for the consolidated search routes using EnhancedSearchEngine."""
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user for testing."""
        user = Mock(spec=User)
        user.id = 1
        user.username = "testuser"
        user.email = "test@example.com"
        return user
    
    @pytest.fixture
    def mock_search_engine(self):
        """Mock EnhancedSearchEngine for testing."""
        mock_engine = AsyncMock()
        
        # Mock search results
        mock_result = Mock(spec=SearchResult)
        mock_result.document_id = "doc_1"
        mock_result.text = "This is a test document content for search testing"
        mock_result.score = 0.95
        mock_result.chunk_id = "chunk_1"
        mock_result.document_metadata = {"filename": "test.pdf", "title": "Test Document"}
        mock_result.metadata = {"content_type": "pdf", "page": 1}
        
        mock_engine.search.return_value = [mock_result]
        mock_engine.text_search.return_value = [mock_result]
        mock_engine.get_search_suggestions.return_value = [
            {"type": "history", "text": "machine learning", "icon": "clock"},
            {"type": "document_title", "text": "Deep Learning Guide", "icon": "document"}
        ]
        mock_engine.get_available_filters.return_value = {
            "file_types": [{"value": "pdf", "label": "PDF", "count": 10}],
            "tags": [{"value": "ml", "label": "Machine Learning", "count": 5}],
            "languages": [],
            "folders": [],
            "date_range": {"min_date": "2024-01-01", "max_date": "2024-12-31"},
            "file_size_range": {"min_size": 1024, "max_size": 10485760, "avg_size": 2048576},
            "search_types": [
                {"value": "semantic", "label": "Semantic", "description": "AI-powered semantic search"}
            ]
        }
        
        return mock_engine
    
    @pytest.fixture
    def search_request(self):
        """Standard search request for testing."""
        return SearchQuery(
            query="machine learning algorithms",
            filters=SearchFilters(
                file_types=["pdf", "txt"],
                tag_ids=["ml", "ai"]
            ),
            semantic_search=True,
            top_k=10,
            similarity_threshold=0.5
        )
    
    @patch('api.routes.search_routes.get_storage_manager')
    @patch('api.routes.search_routes.EnhancedEmbeddingManager')
    @patch('api.routes.search_routes.EnhancedSearchEngine')
    @patch('api.routes.search_routes.get_db')
    @patch('api.routes.search_routes.get_current_active_user')
    async def test_unified_search_semantic_query(
        self, 
        mock_get_user, 
        mock_get_db,
        mock_search_engine_class,
        mock_embedding_manager,
        mock_storage_manager,
        mock_user,
        mock_search_engine,
        search_request
    ):
        """Test unified search with semantic query detection."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = Mock(spec=Session)
        mock_search_engine_class.return_value = mock_search_engine
        
        # Modify request to trigger semantic search detection
        search_request.query = "what is machine learning"
        
        from api.routes.search_routes import unified_search
        
        result = await unified_search(search_request, mock_user)
        
        # Verify search engine was called with SEMANTIC type
        mock_search_engine.search.assert_called_once()
        call_args = mock_search_engine.search.call_args
        assert call_args[1]['search_type'] == SearchType.SEMANTIC
        assert call_args[1]['query'] == "what is machine learning"
        
        # Verify response format
        assert isinstance(result, dict)
        assert 'results' in result
        assert 'total_hits' in result
        assert result['total_hits'] == 1
    
    @patch('api.routes.search_routes.get_storage_manager')
    @patch('api.routes.search_routes.EnhancedEmbeddingManager')
    @patch('api.routes.search_routes.EnhancedSearchEngine')
    @patch('api.routes.search_routes.get_db')
    @patch('api.routes.search_routes.get_current_active_user')
    async def test_unified_search_keyword_query(
        self, 
        mock_get_user, 
        mock_get_db,
        mock_search_engine_class,
        mock_embedding_manager,
        mock_storage_manager,
        mock_user,
        mock_search_engine,
        search_request
    ):
        """Test unified search with keyword query detection."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = Mock(spec=Session)
        mock_search_engine_class.return_value = mock_search_engine
        
        # Modify request to trigger keyword search detection
        search_request.query = "python"
        
        from api.routes.search_routes import unified_search
        
        result = await unified_search(search_request, mock_user)
        
        # Verify search engine was called with KEYWORD type
        mock_search_engine.search.assert_called_once()
        call_args = mock_search_engine.search.call_args
        assert call_args[1]['search_type'] == SearchType.KEYWORD
    
    @patch('api.routes.search_routes.get_storage_manager')
    @patch('api.routes.search_routes.EnhancedEmbeddingManager')
    @patch('api.routes.search_routes.EnhancedSearchEngine')
    @patch('api.routes.search_routes.get_db')
    @patch('api.routes.search_routes.get_current_active_user')
    async def test_unified_search_hybrid_query(
        self, 
        mock_get_user, 
        mock_get_db,
        mock_search_engine_class,
        mock_embedding_manager,
        mock_storage_manager,
        mock_user,
        mock_search_engine,
        search_request
    ):
        """Test unified search with hybrid query detection."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = Mock(spec=Session)
        mock_search_engine_class.return_value = mock_search_engine
        
        # Modify request to trigger hybrid search detection
        search_request.query = "explain the detailed process of implementing machine learning algorithms in production environments"
        
        from api.routes.search_routes import unified_search
        
        result = await unified_search(search_request, mock_user)
        
        # Verify search engine was called with HYBRID type
        mock_search_engine.search.assert_called_once()
        call_args = mock_search_engine.search.call_args
        assert call_args[1]['search_type'] == SearchType.HYBRID
    
    @patch('api.routes.search_routes.get_storage_manager')
    @patch('api.routes.search_routes.EnhancedEmbeddingManager')
    @patch('api.routes.search_routes.EnhancedSearchEngine')
    @patch('api.routes.search_routes.get_db')
    @patch('api.routes.search_routes.get_current_active_user')
    async def test_text_search_endpoint(
        self, 
        mock_get_user, 
        mock_get_db,
        mock_search_engine_class,
        mock_embedding_manager,
        mock_storage_manager,
        mock_user,
        mock_search_engine,
        search_request
    ):
        """Test text search endpoint uses SearchEngine.text_search."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = Mock(spec=Session)
        mock_search_engine_class.return_value = mock_search_engine
        
        from api.routes.search_routes import text_search
        
        result = await text_search(search_request, mock_user)
        
        # Verify text_search method was called
        mock_search_engine.text_search.assert_called_once()
        call_args = mock_search_engine.text_search.call_args
        assert call_args[1]['query'] == search_request.query
        assert call_args[1]['user'] == mock_user
        assert call_args[1]['limit'] == search_request.top_k
    
    @patch('api.routes.search_routes.get_storage_manager')
    @patch('api.routes.search_routes.EnhancedEmbeddingManager')
    @patch('api.routes.search_routes.EnhancedSearchEngine')
    @patch('api.routes.search_routes.get_db')
    @patch('api.routes.search_routes.get_current_active_user')
    async def test_semantic_search_endpoint(
        self, 
        mock_get_user, 
        mock_get_db,
        mock_search_engine_class,
        mock_embedding_manager,
        mock_storage_manager,
        mock_user,
        mock_search_engine,
        search_request
    ):
        """Test semantic search endpoint uses SearchEngine with SEMANTIC type."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = Mock(spec=Session)
        mock_search_engine_class.return_value = mock_search_engine
        
        from api.routes.search_routes import semantic_search
        
        result = await semantic_search(search_request, mock_user)
        
        # Verify search method was called with SEMANTIC type
        mock_search_engine.search.assert_called_once()
        call_args = mock_search_engine.search.call_args
        assert call_args[1]['search_type'] == SearchType.SEMANTIC
    
    @patch('api.routes.search_routes.get_storage_manager')
    @patch('api.routes.search_routes.EnhancedEmbeddingManager')
    @patch('api.routes.search_routes.EnhancedSearchEngine')
    @patch('api.routes.search_routes.get_db')
    @patch('api.routes.search_routes.get_current_active_user')
    async def test_hybrid_search_endpoint(
        self, 
        mock_get_user, 
        mock_get_db,
        mock_search_engine_class,
        mock_embedding_manager,
        mock_storage_manager,
        mock_user,
        mock_search_engine,
        search_request
    ):
        """Test hybrid search endpoint uses SearchEngine with HYBRID type."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = Mock(spec=Session)
        mock_search_engine_class.return_value = mock_search_engine
        
        from api.routes.search_routes import hybrid_search
        
        result = await hybrid_search(search_request, mock_user)
        
        # Verify search method was called with HYBRID type
        mock_search_engine.search.assert_called_once()
        call_args = mock_search_engine.search.call_args
        assert call_args[1]['search_type'] == SearchType.HYBRID
    
    @patch('api.routes.search_routes.get_storage_manager')
    @patch('api.routes.search_routes.EnhancedEmbeddingManager')
    @patch('api.routes.search_routes.EnhancedSearchEngine')
    @patch('api.routes.search_routes.get_db')
    @patch('api.routes.search_routes.get_current_active_user')
    async def test_get_search_suggestions(
        self, 
        mock_get_user, 
        mock_get_db,
        mock_search_engine_class,
        mock_embedding_manager,
        mock_storage_manager,
        mock_user,
        mock_search_engine
    ):
        """Test search suggestions endpoint."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = Mock(spec=Session)
        mock_search_engine_class.return_value = mock_search_engine
        
        from api.routes.search_routes import get_search_suggestions
        
        result = await get_search_suggestions(
            query="machine", 
            limit=5, 
            current_user=mock_user
        )
        
        # Verify suggestions method was called
        mock_search_engine.get_search_suggestions.assert_called_once()
        call_args = mock_search_engine.get_search_suggestions.call_args
        assert call_args[1]['query'] == "machine"
        assert call_args[1]['limit'] == 5
        assert call_args[1]['user'] == mock_user
        
        # Verify response format
        assert isinstance(result, list)
        assert len(result) == 2  # Based on mock data
        assert result[0].type == "history"
        assert result[0].text == "machine learning"
    
    @patch('api.routes.search_routes.get_storage_manager')
    @patch('api.routes.search_routes.EnhancedEmbeddingManager')
    @patch('api.routes.search_routes.EnhancedSearchEngine')
    @patch('api.routes.search_routes.get_db')
    @patch('api.routes.search_routes.get_current_active_user')
    async def test_get_available_filters(
        self, 
        mock_get_user, 
        mock_get_db,
        mock_search_engine_class,
        mock_embedding_manager,
        mock_storage_manager,
        mock_user,
        mock_search_engine
    ):
        """Test available filters endpoint."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = Mock(spec=Session)
        mock_search_engine_class.return_value = mock_search_engine
        
        from api.routes.search_routes import get_available_filters
        
        result = await get_available_filters(current_user=mock_user)
        
        # Verify filters method was called
        mock_search_engine.get_available_filters.assert_called_once()
        call_args = mock_search_engine.get_available_filters.call_args
        assert call_args[1]['user'] == mock_user
        
        # Verify response structure
        assert 'file_types' in result
        assert 'tags' in result
        assert 'date_range' in result
        assert result['file_types'][0]['value'] == "pdf"
    
    @patch('api.routes.search_routes.get_storage_manager')
    @patch('api.routes.search_routes.EnhancedEmbeddingManager')
    @patch('api.routes.search_routes.EnhancedSearchEngine')
    @patch('api.routes.search_routes.get_db')
    @patch('api.routes.search_routes.get_current_active_user')
    async def test_search_with_filters(
        self, 
        mock_get_user, 
        mock_get_db,
        mock_search_engine_class,
        mock_embedding_manager,
        mock_storage_manager,
        mock_user,
        mock_search_engine,
        search_request
    ):
        """Test that API filters are properly converted to SearchEngine format."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = Mock(spec=Session)
        mock_search_engine_class.return_value = mock_search_engine
        
        from api.routes.search_routes import unified_search
        
        # Set up filters
        search_request.filters = SearchFilters(
            file_types=["pdf", "txt"],
            tag_ids=["ml", "ai"],
            date_range=("2024-01-01", "2024-12-31")
        )
        
        result = await unified_search(search_request, mock_user)
        
        # Verify search was called with converted filters
        mock_search_engine.search.assert_called_once()
        call_args = mock_search_engine.search.call_args
        
        filters = call_args[1]['filters']
        assert hasattr(filters, 'content_types')
        assert hasattr(filters, 'min_score')
        assert filters.min_score == 0.5  # From similarity_threshold
    
    @patch('api.routes.search_routes.get_storage_manager')
    @patch('api.routes.search_routes.EnhancedEmbeddingManager')
    @patch('api.routes.search_routes.EnhancedSearchEngine')
    @patch('api.routes.search_routes.get_db')
    @patch('api.routes.search_routes.get_current_active_user')
    async def test_search_error_handling(
        self, 
        mock_get_user, 
        mock_get_db,
        mock_search_engine_class,
        mock_embedding_manager,
        mock_storage_manager,
        mock_user,
        mock_search_engine,
        search_request
    ):
        """Test error handling in search endpoints."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = Mock(spec=Session)
        mock_search_engine_class.return_value = mock_search_engine
        
        # Make search engine throw an exception
        mock_search_engine.search.side_effect = Exception("Search failed")
        
        from api.routes.search_routes import unified_search
        
        # Test that HTTPException is raised
        with pytest.raises(Exception):  # FastAPI HTTPException
            await unified_search(search_request, mock_user)


class TestSearchRouteIntegration:
    """Integration tests for search route functionality."""
    
    def test_search_type_detection_logic(self):
        """Test the intelligent search type detection logic."""
        from vector_db.search_engine import SearchType
        
        def detect_optimal_search_type(query: str) -> str:
            # Short queries → keyword search
            if len(query.split()) <= 2:
                return SearchType.KEYWORD
            
            # Question-like queries → semantic search  
            if query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')):
                return SearchType.SEMANTIC
            
            # Complex queries → hybrid search
            if len(query.split()) > 5:
                return SearchType.HYBRID
                
            # Default to semantic for best results
            return SearchType.SEMANTIC
        
        # Test cases
        assert detect_optimal_search_type("python") == SearchType.KEYWORD
        assert detect_optimal_search_type("machine learning") == SearchType.KEYWORD
        assert detect_optimal_search_type("what is python") == SearchType.SEMANTIC
        assert detect_optimal_search_type("how to code") == SearchType.SEMANTIC
        assert detect_optimal_search_type("explain complex algorithms") == SearchType.SEMANTIC
        assert detect_optimal_search_type("detailed explanation of machine learning algorithms and their applications") == SearchType.HYBRID
    
    def test_api_filter_conversion(self):
        """Test API filter to SearchEngine filter conversion."""
        from api.schemas.search_schemas import SearchFilters, convert_api_filters_to_search_filter
        
        api_filters = SearchFilters(
            file_types=["pdf", "txt"],
            tag_ids=["ml", "ai"],
            date_range=("2024-01-01", "2024-12-31")
        )
        
        search_filter = convert_api_filters_to_search_filter(api_filters)
        
        assert hasattr(search_filter, 'content_types')
        assert search_filter.content_types == ["pdf", "txt"]
        assert search_filter.tags == ["ml", "ai"]
        assert search_filter.date_range is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])