"""
Standard response schemas for the RAG API.
Provides consistent response formats for all API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Generic, TypeVar, Union
from pydantic import BaseModel, Field
import uuid

T = TypeVar('T')


class ResponseMetadata(BaseModel):
    """Metadata included in all API responses."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request correlation ID")
    request_id: Optional[str] = Field(None, description="Original request identifier")
    api_version: str = Field(default="v1", description="API version")
    response_time_ms: Optional[float] = Field(None, description="Response processing time in milliseconds")


class StandardResponse(BaseModel, Generic[T]):
    """Standard response wrapper for all successful API responses."""
    
    success: bool = Field(True, description="Indicates successful operation")
    data: T = Field(..., description="Response payload")
    message: Optional[str] = Field(None, description="Optional success message")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    error_code: str = Field(..., description="Standardized error code")
    message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error (for validation errors)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional error context")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    success: bool = Field(False, description="Indicates failed operation")
    error: ErrorDetail = Field(..., description="Error details")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


class ValidationErrorResponse(BaseModel):
    """Response format for validation errors with multiple field errors."""
    
    success: bool = Field(False, description="Indicates failed operation")
    error: ErrorDetail = Field(..., description="Primary error details")
    field_errors: List[ErrorDetail] = Field(default_factory=list, description="Field-specific validation errors")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


class PaginatedResponse(BaseModel, Generic[T]):
    """Response wrapper for paginated data."""
    
    success: bool = Field(True, description="Indicates successful operation")
    data: List[T] = Field(..., description="List of items")
    pagination: "PaginationInfo" = Field(..., description="Pagination information")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


class PaginationInfo(BaseModel):
    """Pagination information for list responses."""
    
    current_page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")
    next_page: Optional[int] = Field(None, description="Next page number")
    previous_page: Optional[int] = Field(None, description="Previous page number")


class OperationResponse(BaseModel):
    """Response for operations that don't return specific data."""
    
    success: bool = Field(True, description="Indicates successful operation")
    message: str = Field(..., description="Operation result message")
    operation_id: Optional[str] = Field(None, description="Operation identifier")
    affected_count: Optional[int] = Field(None, description="Number of items affected")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


class BatchOperationResponse(BaseModel):
    """Response for batch operations."""
    
    success: bool = Field(True, description="Indicates overall operation success")
    message: str = Field(..., description="Overall operation message")
    batch_id: str = Field(..., description="Batch operation identifier")
    total_items: int = Field(..., description="Total items in batch")
    successful_items: int = Field(..., description="Successfully processed items")
    failed_items: int = Field(..., description="Failed items")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[ErrorDetail] = Field(default_factory=list, description="Item-specific errors")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


class HealthCheckResponse(BaseModel):
    """Health check response format."""
    
    status: str = Field(..., description="Overall health status (healthy, degraded, unhealthy)")
    version: str = Field(..., description="API version")
    uptime_seconds: int = Field(..., description="Service uptime in seconds")
    services: Dict[str, "ServiceStatus"] = Field(..., description="Individual service statuses")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


class ServiceStatus(BaseModel):
    """Status of an individual service component."""
    
    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status (up, down, degraded)")
    response_time_ms: Optional[float] = Field(None, description="Service response time")
    last_checked: datetime = Field(..., description="Last health check time")
    error_message: Optional[str] = Field(None, description="Error message if service is down")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional service details")


class AsyncOperationResponse(BaseModel):
    """Response for long-running async operations."""
    
    success: bool = Field(True, description="Indicates operation was started successfully")
    message: str = Field(..., description="Operation status message")
    operation_id: str = Field(..., description="Unique operation identifier")
    status: str = Field(..., description="Operation status (pending, running, completed, failed)")
    progress_percent: Optional[float] = Field(None, description="Operation progress percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    result_url: Optional[str] = Field(None, description="URL to fetch operation result")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


class FileUploadResponse(BaseModel):
    """Response for file upload operations."""
    
    success: bool = Field(True, description="Indicates successful upload")
    message: str = Field(..., description="Upload status message")
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File MIME type")
    upload_time: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    processing_status: Optional[str] = Field(None, description="File processing status")
    download_url: Optional[str] = Field(None, description="File download URL")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


class SearchResponse(BaseModel, Generic[T]):
    """Response format for search operations."""
    
    success: bool = Field(True, description="Indicates successful search")
    query: str = Field(..., description="Original search query")
    results: List[T] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of matching results")
    search_time_ms: float = Field(..., description="Search execution time")
    facets: Optional[Dict[str, Any]] = Field(None, description="Search facets/filters")
    suggestions: Optional[List[str]] = Field(None, description="Query suggestions")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    """Response for authentication operations."""
    
    success: bool = Field(True, description="Indicates successful authentication")
    message: str = Field(..., description="Authentication status message")
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    user_id: str = Field(..., description="Authenticated user ID")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


class MetricsResponse(BaseModel):
    """Response for metrics and analytics data."""
    
    success: bool = Field(True, description="Indicates successful metrics retrieval")
    metrics: Dict[str, Any] = Field(..., description="Metrics data")
    time_range: Dict[str, str] = Field(..., description="Time range for metrics")
    aggregation: Optional[str] = Field(None, description="Aggregation method used")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata, description="Response metadata")
    
    class Config:
        from_attributes = True


def create_success_response(
    data: Any,
    message: Optional[str] = None,
    correlation_id: Optional[str] = None,
    response_time_ms: Optional[float] = None
) -> StandardResponse:
    """Helper function to create standardized success responses."""
    metadata = ResponseMetadata()
    if correlation_id:
        metadata.correlation_id = correlation_id
    if response_time_ms:
        metadata.response_time_ms = response_time_ms
    
    return StandardResponse(
        data=data,
        message=message,
        metadata=metadata
    )


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    field: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> ErrorResponse:
    """Helper function to create standardized error responses."""
    metadata = ResponseMetadata()
    if correlation_id:
        metadata.correlation_id = correlation_id
    
    error_detail = ErrorDetail(
        error_code=error_code,
        message=message,
        field=field,
        context=context
    )
    
    return ErrorResponse(
        error=error_detail,
        metadata=metadata
    )


def create_validation_error_response(
    error_code: str,
    message: str,
    field_errors: Optional[List[Dict[str, str]]] = None,
    correlation_id: Optional[str] = None
) -> ValidationErrorResponse:
    """Helper function to create validation error responses."""
    metadata = ResponseMetadata()
    if correlation_id:
        metadata.correlation_id = correlation_id
    
    error_detail = ErrorDetail(
        error_code=error_code,
        message=message
    )
    
    field_error_list = []
    if field_errors:
        for field_error in field_errors:
            field_error_list.append(
                ErrorDetail(
                    error_code=field_error.get("code", "VALIDATION_ERROR"),
                    message=field_error.get("message", "Validation failed"),
                    field=field_error.get("field")
                )
            )
    
    return ValidationErrorResponse(
        error=error_detail,
        field_errors=field_error_list,
        metadata=metadata
    )


def create_paginated_response(
    data: List[Any],
    current_page: int,
    page_size: int,
    total_items: int,
    correlation_id: Optional[str] = None
) -> PaginatedResponse:
    """Helper function to create paginated responses."""
    total_pages = (total_items + page_size - 1) // page_size
    has_next = current_page < total_pages
    has_previous = current_page > 1
    
    pagination_info = PaginationInfo(
        current_page=current_page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        has_next=has_next,
        has_previous=has_previous,
        next_page=current_page + 1 if has_next else None,
        previous_page=current_page - 1 if has_previous else None
    )
    
    metadata = ResponseMetadata()
    if correlation_id:
        metadata.correlation_id = correlation_id
    
    return PaginatedResponse(
        data=data,
        pagination=pagination_info,
        metadata=metadata
    )


# Update forward references
PaginatedResponse.model_rebuild()
HealthCheckResponse.model_rebuild()