"""
Unified exception system for the RAG application.
Provides comprehensive exception hierarchy with standardized error codes and context.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid


class BaseRAGException(Exception):
    """Base exception class for all RAG application exceptions."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 500,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code or self._get_default_error_code()
        self.status_code = status_code
        self.context = context or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)
    
    def _get_default_error_code(self) -> str:
        """Override in subclasses to provide default error code."""
        return "UNKNOWN_ERROR"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format for API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
            "context": self.context,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat()
        }


class AuthenticationException(BaseRAGException):
    """Base class for authentication-related exceptions."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("status_code", 401)
        super().__init__(message, **kwargs)


class AuthenticationFailedException(AuthenticationException):
    """Raised when user authentication fails."""
    
    def _get_default_error_code(self) -> str:
        return "AUTH_001"


class InvalidCredentialsException(AuthenticationException):
    """Raised when user provides invalid credentials."""
    
    def _get_default_error_code(self) -> str:
        return "AUTH_002"


class AccountLockedException(AuthenticationException):
    """Raised when user account is locked."""
    
    def __init__(self, message: str, locked_until: Optional[datetime] = None, **kwargs):
        kwargs.setdefault("status_code", 423)
        if locked_until:
            kwargs.setdefault("context", {})["locked_until"] = locked_until.isoformat()
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "AUTH_003"


class TokenExpiredException(AuthenticationException):
    """Raised when JWT token has expired."""
    
    def _get_default_error_code(self) -> str:
        return "AUTH_004"


class InvalidTokenException(AuthenticationException):
    """Raised when JWT token is invalid."""
    
    def _get_default_error_code(self) -> str:
        return "AUTH_005"


class AuthorizationException(BaseRAGException):
    """Base class for authorization-related exceptions."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("status_code", 403)
        super().__init__(message, **kwargs)


class InsufficientPermissionsException(AuthorizationException):
    """Raised when user lacks required permissions."""
    
    def __init__(self, message: str, required_permissions: Optional[List[str]] = None, **kwargs):
        if required_permissions:
            kwargs.setdefault("context", {})["required_permissions"] = required_permissions
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "AUTH_006"


class ValidationException(BaseRAGException):
    """Base class for validation-related exceptions."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("status_code", 422)
        super().__init__(message, **kwargs)


class InvalidInputException(ValidationException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field_errors: Optional[Dict[str, str]] = None, **kwargs):
        if field_errors:
            kwargs.setdefault("context", {})["field_errors"] = field_errors
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "VALIDATION_001"


class MissingFieldException(ValidationException):
    """Raised when required fields are missing."""
    
    def __init__(self, message: str, missing_fields: Optional[List[str]] = None, **kwargs):
        if missing_fields:
            kwargs.setdefault("context", {})["missing_fields"] = missing_fields
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "VALIDATION_002"


class DocumentException(BaseRAGException):
    """Base class for document-related exceptions."""
    
    def __init__(self, message: str, document_id: Optional[str] = None, **kwargs):
        if document_id:
            kwargs.setdefault("context", {})["document_id"] = document_id
        super().__init__(message, **kwargs)


class DocumentNotFoundException(DocumentException):
    """Raised when a document is not found."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("status_code", 404)
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "DOC_001"


class DocumentProcessingException(DocumentException):
    """Raised when document processing fails."""
    
    def __init__(self, message: str, processing_stage: Optional[str] = None, **kwargs):
        kwargs.setdefault("status_code", 422)
        if processing_stage:
            kwargs.setdefault("context", {})["processing_stage"] = processing_stage
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "DOC_002"


class InvalidFileTypeException(DocumentException):
    """Raised when file type is not supported."""
    
    def __init__(self, message: str, file_type: Optional[str] = None, supported_types: Optional[List[str]] = None, **kwargs):
        kwargs.setdefault("status_code", 415)
        context = kwargs.setdefault("context", {})
        if file_type:
            context["file_type"] = file_type
        if supported_types:
            context["supported_types"] = supported_types
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "DOC_003"


class FileTooLargeException(DocumentException):
    """Raised when file exceeds size limits."""
    
    def __init__(self, message: str, file_size: Optional[int] = None, max_size: Optional[int] = None, **kwargs):
        kwargs.setdefault("status_code", 413)
        context = kwargs.setdefault("context", {})
        if file_size:
            context["file_size"] = file_size
        if max_size:
            context["max_size"] = max_size
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "DOC_004"


class DocumentAccessDeniedException(DocumentException):
    """Raised when user cannot access document."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("status_code", 403)
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "DOC_005"


class SearchException(BaseRAGException):
    """Base class for search-related exceptions."""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        if query:
            kwargs.setdefault("context", {})["query"] = query
        super().__init__(message, **kwargs)


class InvalidSearchQueryException(SearchException):
    """Raised when search query is invalid."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("status_code", 400)
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "SEARCH_001"


class SearchEngineUnavailableException(SearchException):
    """Raised when search engine is unavailable."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("status_code", 503)
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "SEARCH_002"


class SearchTimeoutException(SearchException):
    """Raised when search operation times out."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        kwargs.setdefault("status_code", 408)
        if timeout_seconds:
            kwargs.setdefault("context", {})["timeout_seconds"] = timeout_seconds
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "SEARCH_003"


class VectorException(BaseRAGException):
    """Base class for vector database-related exceptions."""
    
    def __init__(self, message: str, vector_id: Optional[str] = None, **kwargs):
        if vector_id:
            kwargs.setdefault("context", {})["vector_id"] = vector_id
        super().__init__(message, **kwargs)


class VectorNotFoundException(VectorException):
    """Raised when vector is not found."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("status_code", 404)
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "VECTOR_001"


class EmbeddingGenerationException(VectorException):
    """Raised when embedding generation fails."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        kwargs.setdefault("status_code", 422)
        if model_name:
            kwargs.setdefault("context", {})["model_name"] = model_name
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "VECTOR_002"


class VectorDatabaseException(VectorException):
    """Raised when vector database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        kwargs.setdefault("status_code", 503)
        if operation:
            kwargs.setdefault("context", {})["operation"] = operation
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "VECTOR_003"


class InvalidVectorDimensionException(VectorException):
    """Raised when vector dimensions are invalid."""
    
    def __init__(self, message: str, expected_dim: Optional[int] = None, actual_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault("status_code", 400)
        context = kwargs.setdefault("context", {})
        if expected_dim:
            context["expected_dimension"] = expected_dim
        if actual_dim:
            context["actual_dimension"] = actual_dim
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "VECTOR_004"


class LLMException(BaseRAGException):
    """Base class for LLM-related exceptions."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        if model_name:
            kwargs.setdefault("context", {})["model_name"] = model_name
        super().__init__(message, **kwargs)


class LLMUnavailableException(LLMException):
    """Raised when LLM service is unavailable."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("status_code", 503)
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "LLM_001"


class LLMTimeoutException(LLMException):
    """Raised when LLM request times out."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        kwargs.setdefault("status_code", 408)
        if timeout_seconds:
            kwargs.setdefault("context", {})["timeout_seconds"] = timeout_seconds
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "LLM_002"


class LLMRateLimitException(LLMException):
    """Raised when LLM rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        kwargs.setdefault("status_code", 429)
        if retry_after:
            kwargs.setdefault("context", {})["retry_after"] = retry_after
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "LLM_003"


class InvalidPromptException(LLMException):
    """Raised when prompt is invalid or too long."""
    
    def __init__(self, message: str, prompt_length: Optional[int] = None, max_length: Optional[int] = None, **kwargs):
        kwargs.setdefault("status_code", 400)
        context = kwargs.setdefault("context", {})
        if prompt_length:
            context["prompt_length"] = prompt_length
        if max_length:
            context["max_length"] = max_length
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "LLM_004"


class ChatException(BaseRAGException):
    """Base class for chat-related exceptions."""
    
    def __init__(self, message: str, session_id: Optional[str] = None, **kwargs):
        if session_id:
            kwargs.setdefault("context", {})["session_id"] = session_id
        super().__init__(message, **kwargs)


class ChatSessionNotFoundException(ChatException):
    """Raised when chat session is not found."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("status_code", 404)
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "CHAT_001"


class ChatSessionExpiredException(ChatException):
    """Raised when chat session has expired."""
    
    def __init__(self, message: str, expired_at: Optional[datetime] = None, **kwargs):
        kwargs.setdefault("status_code", 410)
        if expired_at:
            kwargs.setdefault("context", {})["expired_at"] = expired_at.isoformat()
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "CHAT_002"


class LibraryException(BaseRAGException):
    """Base class for library management exceptions."""
    
    def __init__(self, message: str, library_id: Optional[str] = None, **kwargs):
        if library_id:
            kwargs.setdefault("context", {})["library_id"] = library_id
        super().__init__(message, **kwargs)


class FolderNotFoundException(LibraryException):
    """Raised when folder is not found."""
    
    def __init__(self, message: str, folder_id: Optional[str] = None, **kwargs):
        kwargs.setdefault("status_code", 404)
        if folder_id:
            kwargs.setdefault("context", {})["folder_id"] = folder_id
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "LIBRARY_001"


class TagNotFoundException(LibraryException):
    """Raised when tag is not found."""
    
    def __init__(self, message: str, tag_name: Optional[str] = None, **kwargs):
        kwargs.setdefault("status_code", 404)
        if tag_name:
            kwargs.setdefault("context", {})["tag_name"] = tag_name
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "LIBRARY_002"


class DuplicateResourceException(BaseRAGException):
    """Raised when attempting to create a resource that already exists."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None, **kwargs):
        kwargs.setdefault("status_code", 409)
        context = kwargs.setdefault("context", {})
        if resource_type:
            context["resource_type"] = resource_type
        if resource_id:
            context["resource_id"] = resource_id
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "DUPLICATE_001"


class ExternalServiceException(BaseRAGException):
    """Base class for external service exceptions."""
    
    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        kwargs.setdefault("status_code", 502)
        if service_name:
            kwargs.setdefault("context", {})["service_name"] = service_name
        super().__init__(message, **kwargs)


class DatabaseConnectionException(ExternalServiceException):
    """Raised when database connection fails."""
    
    def _get_default_error_code(self) -> str:
        return "DATABASE_001"


class RedisConnectionException(ExternalServiceException):
    """Raised when Redis connection fails."""
    
    def _get_default_error_code(self) -> str:
        return "REDIS_001"


class ConfigurationException(BaseRAGException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        kwargs.setdefault("status_code", 500)
        if config_key:
            kwargs.setdefault("context", {})["config_key"] = config_key
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "CONFIG_001"


class RateLimitException(BaseRAGException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, limit: Optional[int] = None, retry_after: Optional[int] = None, **kwargs):
        kwargs.setdefault("status_code", 429)
        context = kwargs.setdefault("context", {})
        if limit:
            context["limit"] = limit
        if retry_after:
            context["retry_after"] = retry_after
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "RATE_LIMIT_001"


class MaintenanceModeException(BaseRAGException):
    """Raised when system is in maintenance mode."""
    
    def __init__(self, message: str, estimated_completion: Optional[datetime] = None, **kwargs):
        kwargs.setdefault("status_code", 503)
        if estimated_completion:
            kwargs.setdefault("context", {})["estimated_completion"] = estimated_completion.isoformat()
        super().__init__(message, **kwargs)
    
    def _get_default_error_code(self) -> str:
        return "MAINTENANCE_001"


def map_exception_to_http_status(exception: BaseRAGException) -> int:
    """Map exception to appropriate HTTP status code."""
    return exception.status_code


def get_exception_hierarchy() -> Dict[str, List[str]]:
    """Get the complete exception hierarchy for documentation."""
    return {
        "BaseRAGException": [
            "AuthenticationException",
            "AuthorizationException", 
            "ValidationException",
            "DocumentException",
            "SearchException",
            "VectorException",
            "LLMException",
            "ChatException",
            "LibraryException",
            "DuplicateResourceException",
            "ExternalServiceException",
            "ConfigurationException",
            "RateLimitException",
            "MaintenanceModeException"
        ],
        "AuthenticationException": [
            "AuthenticationFailedException",
            "InvalidCredentialsException",
            "AccountLockedException",
            "TokenExpiredException",
            "InvalidTokenException"
        ],
        "AuthorizationException": [
            "InsufficientPermissionsException"
        ],
        "ValidationException": [
            "InvalidInputException",
            "MissingFieldException"
        ],
        "DocumentException": [
            "DocumentNotFoundException",
            "DocumentProcessingException",
            "InvalidFileTypeException",
            "FileTooLargeException",
            "DocumentAccessDeniedException"
        ],
        "SearchException": [
            "InvalidSearchQueryException",
            "SearchEngineUnavailableException",
            "SearchTimeoutException"
        ],
        "VectorException": [
            "VectorNotFoundException",
            "EmbeddingGenerationException",
            "VectorDatabaseException",
            "InvalidVectorDimensionException"
        ],
        "LLMException": [
            "LLMUnavailableException",
            "LLMTimeoutException",
            "LLMRateLimitException",
            "InvalidPromptException"
        ],
        "ChatException": [
            "ChatSessionNotFoundException",
            "ChatSessionExpiredException"
        ],
        "LibraryException": [
            "FolderNotFoundException",
            "TagNotFoundException"
        ],
        "ExternalServiceException": [
            "DatabaseConnectionException",
            "RedisConnectionException"
        ]
    }