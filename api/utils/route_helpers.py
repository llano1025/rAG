"""
Route helper utilities for standardized response handling.
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic
from fastapi import HTTPException
from api.schemas.responses import (
    StandardResponse,
    ErrorResponse,
    PaginatedResponse,
    AuthResponse,
    create_success_response,
    create_error_response,
    create_paginated_response
)

T = TypeVar('T')


def success_response(
    data: Any,
    message: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> StandardResponse:
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        message: Optional success message
        correlation_id: Optional correlation ID
        
    Returns:
        StandardResponse with consistent format
    """
    return create_success_response(
        data=data,
        message=message,
        correlation_id=correlation_id
    )


def paginated_response(
    items: List[T],
    page: int,
    size: int,
    total: int,
    correlation_id: Optional[str] = None
) -> PaginatedResponse[T]:
    """
    Create a standardized paginated response.
    
    Args:
        items: List of items for current page
        page: Current page number (1-based)
        size: Page size
        total: Total number of items
        correlation_id: Optional correlation ID
        
    Returns:
        PaginatedResponse with consistent format
    """
    return create_paginated_response(
        data=items,
        current_page=page,
        page_size=size,
        total_items=total,
        correlation_id=correlation_id
    )


def auth_response(
    access_token: str,
    user_id: str,
    permissions: List[str],
    expires_in: int = 3600,
    refresh_token: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> AuthResponse:
    """
    Create a standardized authentication response.
    
    Args:
        access_token: JWT access token
        user_id: Authenticated user ID
        permissions: User permissions
        expires_in: Token expiration time in seconds
        refresh_token: Optional refresh token
        correlation_id: Optional correlation ID
        
    Returns:
        AuthResponse with consistent format
    """
    return AuthResponse(
        success=True,
        message="Authentication successful",
        access_token=access_token,
        expires_in=expires_in,
        refresh_token=refresh_token,
        user_id=user_id,
        permissions=permissions
    )


class RouteResponseValidator:
    """Validator for route responses to ensure they follow standard formats."""
    
    @staticmethod
    def validate_response_model(response_model, actual_response):
        """
        Validate that actual response matches the declared response model.
        
        Args:
            response_model: Expected response model class
            actual_response: Actual response object
            
        Raises:
            HTTPException: If response doesn't match expected format
        """
        if response_model == StandardResponse:
            if not hasattr(actual_response, 'success') or not hasattr(actual_response, 'data'):
                raise HTTPException(
                    status_code=500,
                    detail="Response does not match StandardResponse format"
                )
        
        elif response_model == PaginatedResponse:
            if not hasattr(actual_response, 'pagination'):
                raise HTTPException(
                    status_code=500,
                    detail="Response does not match PaginatedResponse format"
                )
        
        elif response_model == AuthResponse:
            required_fields = ['access_token', 'user_id', 'permissions']
            if not all(hasattr(actual_response, field) for field in required_fields):
                raise HTTPException(
                    status_code=500,
                    detail="Response does not match AuthResponse format"
                )


def route_response_decorator(response_model):
    """
    Decorator to enforce response model validation.
    
    Usage:
        @route_response_decorator(StandardResponse)
        @router.get("/example")
        async def example_endpoint():
            return success_response({"key": "value"})
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            RouteResponseValidator.validate_response_model(response_model, result)
            return result
        return wrapper
    return decorator


# Common response patterns
class ResponsePatterns:
    """Common response patterns for different types of operations."""
    
    @staticmethod
    def created(data: Any, message: str = "Resource created successfully") -> StandardResponse:
        """Standard response for resource creation."""
        return success_response(data=data, message=message)
    
    @staticmethod
    def updated(data: Any, message: str = "Resource updated successfully") -> StandardResponse:
        """Standard response for resource updates."""
        return success_response(data=data, message=message)
    
    @staticmethod
    def deleted(message: str = "Resource deleted successfully") -> StandardResponse:
        """Standard response for resource deletion."""
        return success_response(data={"deleted": True}, message=message)
    
    @staticmethod
    def list_response(items: List[T], message: str = "Resources retrieved successfully") -> StandardResponse:
        """Standard response for list operations."""
        return success_response(
            data={"items": items, "count": len(items)}, 
            message=message
        )
    
    @staticmethod
    def not_found(resource_type: str = "Resource") -> StandardResponse:
        """Standard response for not found errors."""
        return create_error_response(
            error_code="RESOURCE_NOT_FOUND",
            message=f"{resource_type} not found"
        )


# Response model type hints for better IDE support
ResponseType = StandardResponse[Any]
PaginatedResponseType = PaginatedResponse[Any]
AuthResponseType = AuthResponse