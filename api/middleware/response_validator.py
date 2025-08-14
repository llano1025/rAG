"""
Response validation middleware to ensure all API responses follow standard formats.
"""

import logging
from typing import Any, Dict
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from api.schemas.responses import StandardResponse, ErrorResponse, PaginatedResponse

logger = logging.getLogger(__name__)


class ResponseValidationMiddleware:
    """Middleware to validate that all API responses follow standard formats."""
    
    def __init__(self, app, validate_responses: bool = True):
        self.app = app
        self.validate_responses = validate_responses
        
    async def __call__(self, scope, receive, send):
        """ASGI middleware implementation."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        if not self.validate_responses:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        
        # Skip validation for certain paths
        skip_paths = ["/docs", "/openapi.json", "/redoc", "/health", "/favicon.ico"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            await self.app(scope, receive, send)
            return
        
        response_body = b""
        response_status = 200
        response_headers = []
        
        async def send_wrapper(message):
            nonlocal response_body, response_status, response_headers
            
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = message.get("headers", [])
                
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
                
                # Only validate when response is complete
                if not message.get("more_body", False):
                    if self._should_validate_response(request, response_status, response_headers):
                        try:
                            await self._validate_response_format(
                                request, response_body, response_status
                            )
                        except Exception as e:
                            logger.error(f"Response validation failed: {e}")
                            # Log but don't break the response
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    def _should_validate_response(self, request: Request, status_code: int, headers) -> bool:
        """Determine if response should be validated."""
        # Only validate JSON responses
        content_type = None
        for header_name, header_value in headers:
            if header_name == b"content-type":
                content_type = header_value.decode()
                break
        
        if not content_type or "application/json" not in content_type:
            return False
            
        # Only validate API endpoints
        if not request.url.path.startswith("/api/"):
            return False
            
        return True
    
    async def _validate_response_format(self, request: Request, body: bytes, status_code: int):
        """Validate that response follows standard format."""
        if not body:
            return
            
        try:
            import json
            response_data = json.loads(body.decode())
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON response from {request.url.path}")
            return
        
        # Check for standard response format
        if not isinstance(response_data, dict):
            logger.warning(f"Response is not a dictionary: {request.url.path}")
            return
        
        # Validate success responses (2xx status codes)
        if 200 <= status_code < 300:
            self._validate_success_response(request, response_data)
        
        # Validate error responses (4xx, 5xx status codes)
        elif status_code >= 400:
            self._validate_error_response(request, response_data)
    
    def _validate_success_response(self, request: Request, data: Dict[str, Any]):
        """Validate success response format."""
        required_fields = ["success", "data", "metadata"]
        
        # Check for standard response wrapper
        if all(field in data for field in required_fields):
            # StandardResponse format
            if data.get("success") is not True:
                logger.warning(f"Success response has success=False: {request.url.path}")
            
            # Validate metadata
            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                logger.warning(f"Metadata is not a dict: {request.url.path}")
            
            required_metadata = ["timestamp", "correlation_id", "api_version"]
            missing_metadata = [field for field in required_metadata if field not in metadata]
            if missing_metadata:
                logger.warning(f"Missing metadata fields {missing_metadata}: {request.url.path}")
        
        # Check for paginated response
        elif "pagination" in data:
            # PaginatedResponse format
            pagination = data.get("pagination", {})
            required_pagination = ["current_page", "page_size", "total_items", "total_pages"]
            missing_pagination = [field for field in required_pagination if field not in pagination]
            if missing_pagination:
                logger.warning(f"Missing pagination fields {missing_pagination}: {request.url.path}")
        
        else:
            # Legacy format - log warning
            logger.warning(f"Response not using standard format: {request.url.path}")
    
    def _validate_error_response(self, request: Request, data: Dict[str, Any]):
        """Validate error response format."""
        # Check for standard error format
        if "error" in data and "metadata" in data:
            error = data.get("error", {})
            required_error_fields = ["error_code", "message"]
            missing_error_fields = [field for field in required_error_fields if field not in error]
            if missing_error_fields:
                logger.warning(f"Missing error fields {missing_error_fields}: {request.url.path}")
        else:
            logger.warning(f"Error response not using standard format: {request.url.path}")


def create_response_validation_middleware(validate_responses: bool = True):
    """Factory function to create response validation middleware."""
    def middleware_factory(app):
        return ResponseValidationMiddleware(app, validate_responses)
    return middleware_factory