"""
Centralized error codes for the RAG application.
Provides a single source of truth for all error codes and their descriptions.
"""

from enum import Enum
from typing import Dict, Tuple


class ErrorCode(Enum):
    """Enumeration of all standardized error codes in the system."""
    
    # Authentication Errors (AUTH_xxx)
    AUTH_001 = ("AUTH_001", "Authentication failed")
    AUTH_002 = ("AUTH_002", "Invalid credentials provided")
    AUTH_003 = ("AUTH_003", "Account is locked")
    AUTH_004 = ("AUTH_004", "JWT token has expired")
    AUTH_005 = ("AUTH_005", "Invalid JWT token")
    AUTH_006 = ("AUTH_006", "Insufficient permissions")
    AUTH_007 = ("AUTH_007", "User not found")
    AUTH_008 = ("AUTH_008", "Email already registered")
    AUTH_009 = ("AUTH_009", "Username already taken")
    AUTH_010 = ("AUTH_010", "Invalid password format")
    AUTH_011 = ("AUTH_011", "Password reset required")
    AUTH_012 = ("AUTH_012", "Account not verified")
    AUTH_013 = ("AUTH_013", "Session expired")
    AUTH_014 = ("AUTH_014", "Multi-factor authentication required")
    AUTH_015 = ("AUTH_015", "Invalid API key")
    
    # Validation Errors (VALIDATION_xxx)
    VALIDATION_001 = ("VALIDATION_001", "Invalid input provided")
    VALIDATION_002 = ("VALIDATION_002", "Required field missing")
    VALIDATION_003 = ("VALIDATION_003", "Field format invalid")
    VALIDATION_004 = ("VALIDATION_004", "Field length exceeded")
    VALIDATION_005 = ("VALIDATION_005", "Invalid field value")
    VALIDATION_006 = ("VALIDATION_006", "Field value out of range")
    VALIDATION_007 = ("VALIDATION_007", "Invalid email format")
    VALIDATION_008 = ("VALIDATION_008", "Invalid URL format")
    VALIDATION_009 = ("VALIDATION_009", "Invalid date format")
    VALIDATION_010 = ("VALIDATION_010", "Invalid JSON format")
    
    # Document Errors (DOC_xxx)
    DOC_001 = ("DOC_001", "Document not found")
    DOC_002 = ("DOC_002", "Document processing failed")
    DOC_003 = ("DOC_003", "Unsupported file type")
    DOC_004 = ("DOC_004", "File size exceeds limit")
    DOC_005 = ("DOC_005", "Document access denied")
    DOC_006 = ("DOC_006", "Document already exists")
    DOC_007 = ("DOC_007", "Document is corrupted")
    DOC_008 = ("DOC_008", "Document upload failed")
    DOC_009 = ("DOC_009", "Document extraction failed")
    DOC_010 = ("DOC_010", "Document metadata invalid")
    DOC_011 = ("DOC_011", "Document version conflict")
    DOC_012 = ("DOC_012", "Document is locked")
    DOC_013 = ("DOC_013", "OCR processing failed")
    DOC_014 = ("DOC_014", "Document format not supported")
    DOC_015 = ("DOC_015", "Document content empty")
    
    # Search Errors (SEARCH_xxx)
    SEARCH_001 = ("SEARCH_001", "Invalid search query")
    SEARCH_002 = ("SEARCH_002", "Search engine unavailable")
    SEARCH_003 = ("SEARCH_003", "Search timeout")
    SEARCH_004 = ("SEARCH_004", "Search index not found")
    SEARCH_005 = ("SEARCH_005", "Search query too complex")
    SEARCH_006 = ("SEARCH_006", "Search results limit exceeded")
    SEARCH_007 = ("SEARCH_007", "Invalid search filters")
    SEARCH_008 = ("SEARCH_008", "Search syntax error")
    SEARCH_009 = ("SEARCH_009", "Search index corrupted")
    SEARCH_010 = ("SEARCH_010", "Search permission denied")
    
    # Vector Database Errors (VECTOR_xxx)
    VECTOR_001 = ("VECTOR_001", "Vector not found")
    VECTOR_002 = ("VECTOR_002", "Embedding generation failed")
    VECTOR_003 = ("VECTOR_003", "Vector database operation failed")
    VECTOR_004 = ("VECTOR_004", "Invalid vector dimensions")
    VECTOR_005 = ("VECTOR_005", "Vector index not available")
    VECTOR_006 = ("VECTOR_006", "Vector similarity threshold invalid")
    VECTOR_007 = ("VECTOR_007", "Vector batch operation failed")
    VECTOR_008 = ("VECTOR_008", "Vector collection not found")
    VECTOR_009 = ("VECTOR_009", "Vector metadata invalid")
    VECTOR_010 = ("VECTOR_010", "Vector embedding model unavailable")
    
    # LLM Errors (LLM_xxx)
    LLM_001 = ("LLM_001", "LLM service unavailable")
    LLM_002 = ("LLM_002", "LLM request timeout")
    LLM_003 = ("LLM_003", "LLM rate limit exceeded")
    LLM_004 = ("LLM_004", "Invalid prompt")
    LLM_005 = ("LLM_005", "LLM model not found")
    LLM_006 = ("LLM_006", "LLM context length exceeded")
    LLM_007 = ("LLM_007", "LLM API key invalid")
    LLM_008 = ("LLM_008", "LLM response format invalid")
    LLM_009 = ("LLM_009", "LLM content filter triggered")
    LLM_010 = ("LLM_010", "LLM provider configuration invalid")
    
    # Chat Errors (CHAT_xxx)
    CHAT_001 = ("CHAT_001", "Chat session not found")
    CHAT_002 = ("CHAT_002", "Chat session expired")
    CHAT_003 = ("CHAT_003", "Chat message too long")
    CHAT_004 = ("CHAT_004", "Chat history unavailable")
    CHAT_005 = ("CHAT_005", "Chat participant limit exceeded")
    CHAT_006 = ("CHAT_006", "Chat session locked")
    CHAT_007 = ("CHAT_007", "Chat message invalid")
    CHAT_008 = ("CHAT_008", "Chat context lost")
    CHAT_009 = ("CHAT_009", "Chat streaming error")
    CHAT_010 = ("CHAT_010", "Chat configuration invalid")
    
    # Library Management Errors (LIBRARY_xxx)
    LIBRARY_001 = ("LIBRARY_001", "Folder not found")
    LIBRARY_002 = ("LIBRARY_002", "Tag not found")
    LIBRARY_003 = ("LIBRARY_003", "Library access denied")
    LIBRARY_004 = ("LIBRARY_004", "Folder already exists")
    LIBRARY_005 = ("LIBRARY_005", "Tag already exists")
    LIBRARY_006 = ("LIBRARY_006", "Library quota exceeded")
    LIBRARY_007 = ("LIBRARY_007", "Invalid folder structure")
    LIBRARY_008 = ("LIBRARY_008", "Folder not empty")
    LIBRARY_009 = ("LIBRARY_009", "Tag limit exceeded")
    LIBRARY_010 = ("LIBRARY_010", "Library synchronization failed")
    
    # External Service Errors (SERVICE_xxx)
    SERVICE_001 = ("SERVICE_001", "Database connection failed")
    SERVICE_002 = ("SERVICE_002", "Redis connection failed")
    SERVICE_003 = ("SERVICE_003", "External API unavailable")
    SERVICE_004 = ("SERVICE_004", "Service timeout")
    SERVICE_005 = ("SERVICE_005", "Service authentication failed")
    SERVICE_006 = ("SERVICE_006", "Service quota exceeded")
    SERVICE_007 = ("SERVICE_007", "Service configuration invalid")
    SERVICE_008 = ("SERVICE_008", "Service dependency unavailable")
    SERVICE_009 = ("SERVICE_009", "Service version incompatible")
    SERVICE_010 = ("SERVICE_010", "Service maintenance mode")
    
    # Configuration Errors (CONFIG_xxx)
    CONFIG_001 = ("CONFIG_001", "Configuration invalid")
    CONFIG_002 = ("CONFIG_002", "Configuration missing")
    CONFIG_003 = ("CONFIG_003", "Configuration format invalid")
    CONFIG_004 = ("CONFIG_004", "Environment variable missing")
    CONFIG_005 = ("CONFIG_005", "Configuration value out of range")
    CONFIG_006 = ("CONFIG_006", "Configuration dependency missing")
    CONFIG_007 = ("CONFIG_007", "Configuration encryption failed")
    CONFIG_008 = ("CONFIG_008", "Configuration access denied")
    CONFIG_009 = ("CONFIG_009", "Configuration backup failed")
    CONFIG_010 = ("CONFIG_010", "Configuration validation failed")
    
    # Rate Limiting Errors (RATE_xxx)
    RATE_001 = ("RATE_001", "Rate limit exceeded")
    RATE_002 = ("RATE_002", "Request quota exceeded")
    RATE_003 = ("RATE_003", "Concurrent request limit exceeded")
    RATE_004 = ("RATE_004", "API key quota exceeded")
    RATE_005 = ("RATE_005", "User request limit exceeded")
    
    # Duplicate Resource Errors (DUPLICATE_xxx)
    DUPLICATE_001 = ("DUPLICATE_001", "Resource already exists")
    DUPLICATE_002 = ("DUPLICATE_002", "Duplicate entry")
    DUPLICATE_003 = ("DUPLICATE_003", "Conflict with existing resource")
    
    # System Errors (SYSTEM_xxx)
    SYSTEM_001 = ("SYSTEM_001", "Internal server error")
    SYSTEM_002 = ("SYSTEM_002", "Service unavailable")
    SYSTEM_003 = ("SYSTEM_003", "Maintenance mode active")
    SYSTEM_004 = ("SYSTEM_004", "Resource exhausted")
    SYSTEM_005 = ("SYSTEM_005", "Operation not supported")
    SYSTEM_006 = ("SYSTEM_006", "Service degraded")
    SYSTEM_007 = ("SYSTEM_007", "Emergency maintenance")
    SYSTEM_008 = ("SYSTEM_008", "Service overloaded")
    SYSTEM_009 = ("SYSTEM_009", "Critical system failure")
    SYSTEM_010 = ("SYSTEM_010", "Service restart required")
    
    # File Operation Errors (FILE_xxx)
    FILE_001 = ("FILE_001", "File not found")
    FILE_002 = ("FILE_002", "File access denied")
    FILE_003 = ("FILE_003", "File already exists")
    FILE_004 = ("FILE_004", "File corrupted")
    FILE_005 = ("FILE_005", "File operation failed")
    FILE_006 = ("FILE_006", "File format invalid")
    FILE_007 = ("FILE_007", "File size invalid")
    FILE_008 = ("FILE_008", "File path invalid")
    FILE_009 = ("FILE_009", "File permissions invalid")
    FILE_010 = ("FILE_010", "File storage full")
    
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description
    
    def __str__(self) -> str:
        return self.code


class ErrorCodeManager:
    """Manager for error codes with additional functionality."""
    
    @staticmethod
    def get_error_info(error_code: str) -> Tuple[str, str]:
        """Get error code and description by code string."""
        for error in ErrorCode:
            if error.code == error_code:
                return error.code, error.description
        return error_code, "Unknown error"
    
    @staticmethod
    def get_all_codes() -> Dict[str, str]:
        """Get all error codes with their descriptions."""
        return {error.code: error.description for error in ErrorCode}
    
    @staticmethod
    def get_codes_by_category(category: str) -> Dict[str, str]:
        """Get error codes for a specific category (e.g., 'AUTH', 'DOC')."""
        category_upper = category.upper()
        return {
            error.code: error.description 
            for error in ErrorCode 
            if error.code.startswith(f"{category_upper}_")
        }
    
    @staticmethod
    def validate_error_code(error_code: str) -> bool:
        """Validate if an error code exists in the system."""
        return any(error.code == error_code for error in ErrorCode)
    
    @staticmethod
    def get_http_status_mapping() -> Dict[str, int]:
        """Get recommended HTTP status codes for error categories."""
        return {
            "AUTH_001": 401, "AUTH_002": 401, "AUTH_003": 423, "AUTH_004": 401, "AUTH_005": 401,
            "AUTH_006": 403, "AUTH_007": 404, "AUTH_008": 409, "AUTH_009": 409, "AUTH_010": 400,
            "AUTH_011": 403, "AUTH_012": 403, "AUTH_013": 401, "AUTH_014": 401, "AUTH_015": 401,
            
            "VALIDATION_001": 422, "VALIDATION_002": 422, "VALIDATION_003": 422, "VALIDATION_004": 422,
            "VALIDATION_005": 422, "VALIDATION_006": 422, "VALIDATION_007": 422, "VALIDATION_008": 422,
            "VALIDATION_009": 422, "VALIDATION_010": 422,
            
            "DOC_001": 404, "DOC_002": 422, "DOC_003": 415, "DOC_004": 413, "DOC_005": 403,
            "DOC_006": 409, "DOC_007": 422, "DOC_008": 422, "DOC_009": 422, "DOC_010": 422,
            "DOC_011": 409, "DOC_012": 423, "DOC_013": 422, "DOC_014": 415, "DOC_015": 422,
            
            "SEARCH_001": 400, "SEARCH_002": 503, "SEARCH_003": 408, "SEARCH_004": 404,
            "SEARCH_005": 400, "SEARCH_006": 400, "SEARCH_007": 400, "SEARCH_008": 400,
            "SEARCH_009": 503, "SEARCH_010": 403,
            
            "VECTOR_001": 404, "VECTOR_002": 422, "VECTOR_003": 503, "VECTOR_004": 400,
            "VECTOR_005": 503, "VECTOR_006": 400, "VECTOR_007": 422, "VECTOR_008": 404,
            "VECTOR_009": 422, "VECTOR_010": 503,
            
            "LLM_001": 503, "LLM_002": 408, "LLM_003": 429, "LLM_004": 400,
            "LLM_005": 404, "LLM_006": 400, "LLM_007": 401, "LLM_008": 422,
            "LLM_009": 400, "LLM_010": 500,
            
            "CHAT_001": 404, "CHAT_002": 410, "CHAT_003": 400, "CHAT_004": 503,
            "CHAT_005": 400, "CHAT_006": 423, "CHAT_007": 400, "CHAT_008": 500,
            "CHAT_009": 500, "CHAT_010": 500,
            
            "LIBRARY_001": 404, "LIBRARY_002": 404, "LIBRARY_003": 403, "LIBRARY_004": 409,
            "LIBRARY_005": 409, "LIBRARY_006": 413, "LIBRARY_007": 400, "LIBRARY_008": 400,
            "LIBRARY_009": 400, "LIBRARY_010": 503,
            
            "SERVICE_001": 503, "SERVICE_002": 503, "SERVICE_003": 503, "SERVICE_004": 408,
            "SERVICE_005": 502, "SERVICE_006": 503, "SERVICE_007": 500, "SERVICE_008": 503,
            "SERVICE_009": 503, "SERVICE_010": 503,
            
            "CONFIG_001": 500, "CONFIG_002": 500, "CONFIG_003": 500, "CONFIG_004": 500,
            "CONFIG_005": 500, "CONFIG_006": 500, "CONFIG_007": 500, "CONFIG_008": 500,
            "CONFIG_009": 500, "CONFIG_010": 500,
            
            "RATE_001": 429, "RATE_002": 429, "RATE_003": 429, "RATE_004": 429, "RATE_005": 429,
            
            "DUPLICATE_001": 409, "DUPLICATE_002": 409, "DUPLICATE_003": 409,
            
            "SYSTEM_001": 500, "SYSTEM_002": 503, "SYSTEM_003": 503, "SYSTEM_004": 503,
            "SYSTEM_005": 501, "SYSTEM_006": 503, "SYSTEM_007": 503, "SYSTEM_008": 503,
            "SYSTEM_009": 500, "SYSTEM_010": 503,
            
            "FILE_001": 404, "FILE_002": 403, "FILE_003": 409, "FILE_004": 422,
            "FILE_005": 500, "FILE_006": 415, "FILE_007": 413, "FILE_008": 400,
            "FILE_009": 403, "FILE_010": 507
        }
    
    @staticmethod
    def get_error_categories() -> Dict[str, str]:
        """Get all error categories with descriptions."""
        return {
            "AUTH": "Authentication and authorization errors",
            "VALIDATION": "Input validation errors",
            "DOC": "Document processing errors",
            "SEARCH": "Search operation errors",
            "VECTOR": "Vector database errors",
            "LLM": "Large Language Model errors",
            "CHAT": "Chat system errors",
            "LIBRARY": "Library management errors",
            "SERVICE": "External service errors",
            "CONFIG": "Configuration errors",
            "RATE": "Rate limiting errors",
            "DUPLICATE": "Duplicate resource errors",
            "SYSTEM": "System-level errors",
            "FILE": "File operation errors"
        }


def get_error_code_for_exception(exception_class_name: str) -> str:
    """Map exception class names to error codes."""
    mapping = {
        "AuthenticationFailedException": ErrorCode.AUTH_001.code,
        "InvalidCredentialsException": ErrorCode.AUTH_002.code,
        "AccountLockedException": ErrorCode.AUTH_003.code,
        "TokenExpiredException": ErrorCode.AUTH_004.code,
        "InvalidTokenException": ErrorCode.AUTH_005.code,
        "InsufficientPermissionsException": ErrorCode.AUTH_006.code,
        "InvalidInputException": ErrorCode.VALIDATION_001.code,
        "MissingFieldException": ErrorCode.VALIDATION_002.code,
        "DocumentNotFoundException": ErrorCode.DOC_001.code,
        "DocumentProcessingException": ErrorCode.DOC_002.code,
        "InvalidFileTypeException": ErrorCode.DOC_003.code,
        "FileTooLargeException": ErrorCode.DOC_004.code,
        "DocumentAccessDeniedException": ErrorCode.DOC_005.code,
        "InvalidSearchQueryException": ErrorCode.SEARCH_001.code,
        "SearchEngineUnavailableException": ErrorCode.SEARCH_002.code,
        "SearchTimeoutException": ErrorCode.SEARCH_003.code,
        "VectorNotFoundException": ErrorCode.VECTOR_001.code,
        "EmbeddingGenerationException": ErrorCode.VECTOR_002.code,
        "VectorDatabaseException": ErrorCode.VECTOR_003.code,
        "InvalidVectorDimensionException": ErrorCode.VECTOR_004.code,
        "LLMUnavailableException": ErrorCode.LLM_001.code,
        "LLMTimeoutException": ErrorCode.LLM_002.code,
        "LLMRateLimitException": ErrorCode.LLM_003.code,
        "InvalidPromptException": ErrorCode.LLM_004.code,
        "ChatSessionNotFoundException": ErrorCode.CHAT_001.code,
        "ChatSessionExpiredException": ErrorCode.CHAT_002.code,
        "FolderNotFoundException": ErrorCode.LIBRARY_001.code,
        "TagNotFoundException": ErrorCode.LIBRARY_002.code,
        "DuplicateResourceException": ErrorCode.DUPLICATE_001.code,
        "DatabaseConnectionException": ErrorCode.SERVICE_001.code,
        "RedisConnectionException": ErrorCode.SERVICE_002.code,
        "ConfigurationException": ErrorCode.CONFIG_001.code,
        "RateLimitException": ErrorCode.RATE_001.code,
        "MaintenanceModeException": ErrorCode.SYSTEM_003.code
    }
    return mapping.get(exception_class_name, ErrorCode.SYSTEM_001.code)