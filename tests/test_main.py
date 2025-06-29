"""
Tests for the main application module.
"""
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


@pytest.mark.api
class TestMainEndpoints:
    """Test main application endpoints."""
    
    def test_root_endpoint(self, test_client: TestClient):
        """Test the root endpoint returns API information."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "RAG Document Processing API"
        assert data["version"] == "1.0.0"
    
    def test_health_endpoint(self, test_client: TestClient):
        """Test the health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_metrics_endpoint(self, test_client: TestClient):
        """Test the metrics endpoint."""
        response = test_client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        # Should return metrics data or a message
        assert isinstance(data, dict)
    
    @pytest.mark.asyncio
    async def test_root_endpoint_async(self, async_client: AsyncClient):
        """Test the root endpoint using async client."""
        response = await async_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "RAG Document Processing API"
    
    @pytest.mark.asyncio
    async def test_health_endpoint_async(self, async_client: AsyncClient):
        """Test the health endpoint using async client."""
        response = await async_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


@pytest.mark.api
class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self, test_client: TestClient):
        """Test that CORS headers are properly set."""
        response = test_client.options("/", headers={"Origin": "http://localhost:3000"})
        
        # Should allow CORS preflight
        assert response.status_code in [200, 204]
    
    def test_cors_actual_request(self, test_client: TestClient):
        """Test actual request with CORS headers."""
        response = test_client.get("/", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        # CORS headers should be present in response
        assert "access-control-allow-origin" in [h.lower() for h in response.headers.keys()]


@pytest.mark.unit
class TestApplicationConfiguration:
    """Test application configuration and setup."""
    
    def test_app_creation(self, test_app):
        """Test that the FastAPI app is created correctly."""
        assert test_app.title == "RAG Document Processing API"
        assert test_app.version == "1.0.0"
        assert test_app.docs_url == "/docs"
        assert test_app.redoc_url == "/redoc"
    
    def test_app_state_initialization(self, test_app):
        """Test that app state is properly initialized."""
        # These should be set by the test fixtures
        assert hasattr(test_app.state, "redis")
        assert hasattr(test_app.state, "health")
        assert hasattr(test_app.state, "auth")
        assert hasattr(test_app.state, "settings")
    
    def test_router_inclusion(self, test_app):
        """Test that all routers are properly included."""
        routes = [route.path for route in test_app.routes]
        
        # Check that main API routes are included
        api_routes = [route for route in routes if route.startswith("/api/v1")]
        assert len(api_routes) > 0
        
        # Check for main endpoints
        assert "/" in routes
        assert "/health" in routes
        assert "/metrics" in routes


@pytest.mark.integration
class TestApplicationLifespan:
    """Test application startup and shutdown behavior."""
    
    @pytest.mark.asyncio
    async def test_startup_sequence(self, test_app):
        """Test that startup sequence completes successfully."""
        # In a real test, we might test actual startup behavior
        # For now, we verify that the app with mocked dependencies works
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
    
    def test_middleware_stack(self, test_app):
        """Test that middleware is properly configured."""
        middleware_types = [type(middleware.cls).__name__ for middleware in test_app.user_middleware]
        
        # Should include our custom middleware
        expected_middleware = ["CORSMiddleware"]  # At minimum, CORS should be present
        for middleware in expected_middleware:
            assert any(middleware in mw_type for mw_type in middleware_types)


@pytest.mark.api
class TestErrorHandling:
    """Test error handling in main endpoints."""
    
    def test_404_error(self, test_client: TestClient):
        """Test 404 error for non-existent endpoint."""
        response = test_client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, test_client: TestClient):
        """Test 405 error for wrong HTTP method."""
        response = test_client.post("/")  # Root only supports GET
        assert response.status_code == 405
    
    @pytest.mark.asyncio
    async def test_health_endpoint_error_handling(self, async_client: AsyncClient):
        """Test health endpoint when health check fails."""
        # This would require mocking the health check to fail
        # For now, just verify the endpoint responds
        response = await async_client.get("/health")
        assert response.status_code in [200, 503]  # Either healthy or service unavailable