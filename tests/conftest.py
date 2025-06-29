"""
Test configuration and fixtures for the RAG application.
"""
import asyncio
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock
import tempfile
import os
from typing import AsyncGenerator, Generator

# Import application modules
from main import create_app
from config import TestSettings
from utils.caching.redis_manager import RedisManager
from utils.monitoring.health_check import HealthCheck
from api.middleware.auth import AuthMiddleware

# Test settings
@pytest.fixture(scope="session")
def test_settings():
    """Test settings configuration."""
    return TestSettings()

# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Mock Redis Manager
@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager for testing."""
    mock_redis = AsyncMock(spec=RedisManager)
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.exists.return_value = False
    mock_redis.connect.return_value = None
    mock_redis.disconnect.return_value = None
    return mock_redis

# Mock Health Check
@pytest.fixture
def mock_health_check():
    """Mock health check for testing."""
    mock_health = AsyncMock(spec=HealthCheck)
    mock_health.check_all.return_value = {"status": "healthy", "services": {}}
    mock_health.get_metrics.return_value = {"requests": 0, "errors": 0}
    mock_health.initialize.return_value = None
    mock_health.cleanup.return_value = None
    return mock_health

# Mock Auth Middleware
@pytest.fixture
def mock_auth_middleware():
    """Mock authentication middleware for testing."""
    mock_auth = AsyncMock(spec=AuthMiddleware)
    mock_auth.create_access_token.return_value = "test-token"
    mock_auth.verify_token.return_value = {"user_id": "test-user", "exp": 9999999999}
    mock_auth.get_current_user.return_value = {"id": "test-user", "is_admin": False}
    return mock_auth

# Test directories
@pytest.fixture
def temp_upload_dir():
    """Create temporary upload directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def temp_log_dir():
    """Create temporary log directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

# FastAPI App with mocked dependencies
@pytest_asyncio.fixture
async def test_app(
    test_settings,
    mock_redis_manager,
    mock_health_check,
    mock_auth_middleware,
    temp_upload_dir,
    temp_log_dir
):
    """Create test FastAPI application with mocked dependencies."""
    app = create_app()
    
    # Override app state with mocks
    app.state.redis = mock_redis_manager
    app.state.health = mock_health_check
    app.state.auth = mock_auth_middleware
    app.state.settings = test_settings
    
    # Update settings for test environment
    test_settings.UPLOAD_DIRECTORY = temp_upload_dir
    test_settings.TEMP_DIRECTORY = temp_log_dir
    
    yield app

# Sync test client
@pytest.fixture
def test_client(test_app) -> Generator[TestClient, None, None]:
    """Create synchronous test client."""
    with TestClient(test_app) as client:
        yield client

# Async test client
@pytest_asyncio.fixture
async def async_client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Create asynchronous test client."""
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client

# Mock file upload
@pytest.fixture
def mock_upload_file():
    """Create mock upload file for testing."""
    def _create_mock_file(filename: str = "test.txt", content: bytes = b"test content"):
        mock_file = MagicMock()
        mock_file.filename = filename
        mock_file.content_type = "text/plain"
        mock_file.file.read.return_value = content
        mock_file.size = len(content)
        return mock_file
    return _create_mock_file

# Authentication headers
@pytest.fixture
def auth_headers():
    """Authentication headers for protected endpoints."""
    return {"Authorization": "Bearer test-token"}

# Mock user data
@pytest.fixture
def test_user():
    """Test user data."""
    return {
        "id": "test-user-id",
        "username": "testuser",
        "email": "test@example.com",
        "is_admin": False,
        "created_at": "2023-01-01T00:00:00Z"
    }

@pytest.fixture
def admin_user():
    """Test admin user data."""
    return {
        "id": "admin-user-id",
        "username": "adminuser",
        "email": "admin@example.com",
        "is_admin": True,
        "created_at": "2023-01-01T00:00:00Z"
    }

# Mock document data
@pytest.fixture
def test_document():
    """Test document data."""
    return {
        "id": "test-doc-id",
        "filename": "test.pdf",
        "content_type": "application/pdf",
        "size": 1024,
        "user_id": "test-user-id",
        "created_at": "2023-01-01T00:00:00Z",
        "metadata": {
            "title": "Test Document",
            "author": "Test Author"
        }
    }

# Mock search results
@pytest.fixture
def test_search_results():
    """Test search results data."""
    return {
        "results": [
            {
                "document_id": "doc-1",
                "title": "Test Document 1",
                "content": "This is test content for document 1",
                "score": 0.95,
                "metadata": {"author": "Author 1"}
            },
            {
                "document_id": "doc-2", 
                "title": "Test Document 2",
                "content": "This is test content for document 2",
                "score": 0.87,
                "metadata": {"author": "Author 2"}
            }
        ],
        "total": 2,
        "page": 1,
        "page_size": 10
    }

# Database fixtures (if needed)
@pytest_asyncio.fixture
async def db_session():
    """Mock database session."""
    # This would be implemented if using a real database
    mock_session = AsyncMock()
    yield mock_session

# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    # Cleanup logic would go here if needed
    pass

# Pytest markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "auth: Authentication tests")
    config.addinivalue_line("markers", "document: Document processing tests")
    config.addinivalue_line("markers", "search: Search functionality tests")
    config.addinivalue_line("markers", "vector: Vector operations tests")
    config.addinivalue_line("markers", "llm: LLM integration tests")

# Skip conditions for external dependencies
def pytest_runtest_setup(item):
    """Setup conditions for test execution."""
    # Skip Redis tests if Redis is not available
    if "redis" in item.keywords and not os.getenv("REDIS_AVAILABLE"):
        pytest.skip("Redis not available")
    
    # Skip Qdrant tests if Qdrant is not available
    if "qdrant" in item.keywords and not os.getenv("QDRANT_AVAILABLE"):
        pytest.skip("Qdrant not available")
    
    # Skip LLM tests if API keys are not available
    if "llm" in item.keywords and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("LLM API keys not available")