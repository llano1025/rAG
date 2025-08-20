"""
Tests for API routes.
"""
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch


@pytest.mark.api
@pytest.mark.document
class TestDocumentRoutes:
    """Test document-related API routes."""
    
    def test_upload_document_unauthorized(self, test_client: TestClient):
        """Test document upload without authentication."""
        files = {"file": ("test.txt", b"test content", "text/plain")}
        response = test_client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 401
    
    def test_upload_document_authorized(self, test_client: TestClient, auth_headers: dict):
        """Test document upload with authentication."""
        files = {"file": ("test.txt", b"test content", "text/plain")}
        
        with patch("api.controllers.document_controller.process_upload") as mock_upload:
            mock_upload.return_value = {
                "id": "test-doc-id",
                "filename": "test.txt",
                "status": "processed"
            }
            
            response = test_client.post(
                "/api/v1/documents/upload",
                files=files,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["filename"] == "test.txt"
    
    def test_get_document_not_found(self, test_client: TestClient, auth_headers: dict):
        """Test getting non-existent document."""
        with patch("api.controllers.document_controller.get_document") as mock_get:
            mock_get.return_value = None
            
            response = test_client.get(
                "/api/v1/documents/nonexistent-id",
                headers=auth_headers
            )
            
            assert response.status_code == 404
    
    def test_delete_document_success(self, test_client: TestClient, auth_headers: dict):
        """Test successful document deletion."""
        with patch("api.controllers.document_controller.delete_document") as mock_delete:
            mock_delete.return_value = True
            
            response = test_client.delete(
                "/api/v1/documents/test-doc-id",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data


@pytest.mark.api
@pytest.mark.search
class TestSearchRoutes:
    """Test search-related API routes."""
    
    def test_search_documents_unauthorized(self, test_client: TestClient):
        """Test search without authentication."""
        search_data = {"query": "test query"}
        response = test_client.post("/api/v1/search/", json=search_data)
        
        assert response.status_code == 401
    
    def test_search_documents_authorized(self, test_client: TestClient, auth_headers: dict):
        """Test search with authentication."""
        search_data = {
            "query": "test query",
            "filters": {},
            "page": 1,
            "page_size": 10
        }
        
        with patch("api.controllers.search_controller.search_documents") as mock_search:
            mock_search.return_value = {
                "results": [],
                "total": 0,
                "page": 1,
                "page_size": 10
            }
            
            response = test_client.post(
                "/api/v1/search/",
                json=search_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "total" in data
    
    def test_semantic_search(self, test_client: TestClient, auth_headers: dict):
        """Test semantic search endpoint."""
        search_data = {
            "query_text": "test semantic query",
            "top_k": 5,
            "threshold": 0.7
        }
        
        with patch("api.controllers.search_controller.similarity_search") as mock_search:
            mock_search.return_value = {
                "results": [],
                "total": 0
            }
            
            response = test_client.post(
                "/api/v1/search/semantic",
                json=search_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
    
    def test_get_search_suggestions(self, test_client: TestClient, auth_headers: dict):
        """Test search suggestions endpoint."""
        with patch("api.controllers.search_controller.get_search_suggestions") as mock_suggestions:
            mock_suggestions.return_value = ["suggestion1", "suggestion2"]
            
            response = test_client.post(
                "/api/v1/search/suggest?query=test&limit=5",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)


@pytest.mark.api
@pytest.mark.vector
class TestVectorRoutes:
    """Test vector-related API routes."""
    
    @pytest.mark.asyncio
    async def test_vector_operations_unauthorized(self, async_client: AsyncClient):
        """Test vector operations without authentication."""
        response = await async_client.get("/api/v1/vectors/health")
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_vector_health_check(self, async_client: AsyncClient, auth_headers: dict):
        """Test vector database health check."""
        with patch("api.controllers.vector_controller.health_check") as mock_health:
            mock_health.return_value = {"status": "healthy"}
            
            response = await async_client.get(
                "/api/v1/vectors/health",
                headers=auth_headers
            )
            
            assert response.status_code == 200


@pytest.mark.api
@pytest.mark.auth
class TestAuthenticationFlow:
    """Test authentication-related functionality."""
    
    def test_protected_endpoint_requires_auth(self, test_client: TestClient):
        """Test that protected endpoints require authentication."""
        endpoints = [
            "/api/v1/documents/upload",
            "/api/v1/search/",
            "/api/v1/vectors/health"
        ]
        
        for endpoint in endpoints:
            if endpoint == "/api/v1/search/":
                response = test_client.post(endpoint, json={"query": "test"})
            else:
                response = test_client.get(endpoint)
            
            assert response.status_code == 401
    
    def test_invalid_token_rejected(self, test_client: TestClient):
        """Test that invalid tokens are rejected."""
        invalid_headers = {"Authorization": "Bearer invalid-token"}
        
        with patch("api.middleware.auth.AuthMiddleware.verify_token") as mock_verify:
            mock_verify.side_effect = Exception("Invalid token")
            
            response = test_client.get(
                "/api/v1/documents/test-id",
                headers=invalid_headers
            )
            
            assert response.status_code == 401