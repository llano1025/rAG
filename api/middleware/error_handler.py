from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Dict, Any, Optional, List, Union
import logging
import traceback
import sys
from datetime import datetime, timezone
from pydantic import ValidationError
import asyncio
from contextlib import asynccontextmanager

from utils.exceptions import BaseRAGException
from utils.error_codes import ErrorCodeManager, get_error_code_for_exception
from api.schemas.responses import (
    ErrorResponse, 
    ValidationErrorResponse, 
    ResponseMetadata,
    create_error_response,
    create_validation_error_response
)

logger = logging.getLogger(__name__)

# Using unified exception system and response schemas

class ErrorLogger:
    """Handle error logging with different severity levels."""
    
    def __init__(self, logger_instance: logging.Logger):
        self.logger = logger_instance
    
    def log_error(
        self,
        message: str,
        error: Optional[Exception] = None,
        severity: str = "error",
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log error with appropriate severity level."""
        log_data = extra or {}
        if error:
            log_data["error_type"] = error.__class__.__name__
            log_data["error_message"] = str(error)

        if severity == "critical":
            self.logger.critical(message, extra=log_data, exc_info=error)
        elif severity == "error":
            self.logger.error(message, extra=log_data, exc_info=error)
        elif severity == "warning":
            self.logger.warning(message, extra=log_data)
        else:
            self.logger.error(message, extra=log_data, exc_info=error)

class ErrorHandler:
    def __init__(
        self,
        app,
        error_logging: bool = True,
        include_traceback: bool = False,
        error_handlers: Optional[Dict] = None
    ):
        self.app = app
        self.error_logging = error_logging
        self.include_traceback = include_traceback
        self.error_logger = ErrorLogger(logger)
        self.custom_error_handlers = error_handlers or {}

    async def __call__(self, scope, receive, send):
        """ASGI middleware implementation."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        
        async def send_wrapper(message):
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            response = await self.handle_exception(request, exc)
            await response(scope, receive, send)

    async def handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle any exception that occurs during request processing."""
        if isinstance(exc, BaseRAGException):
            return await self.handle_rag_exception(request, exc)
        elif isinstance(exc, StarletteHTTPException):
            return await self.handle_http_exception(request, exc)
        elif isinstance(exc, RequestValidationError):
            return await self.handle_validation_exception(request, exc)
        elif isinstance(exc, ValidationError):
            return await self.handle_pydantic_validation_error(request, exc)
        else:
            return await self.handle_general_exception(request, exc)

    def register_error_handler(self, exception_class: type, handler: callable):
        """Register a custom error handler for a specific exception type."""
        self.custom_error_handlers[exception_class] = handler

    async def handle_rag_exception(
        self,
        request: Request,
        exc: BaseRAGException
    ) -> JSONResponse:
        """Handle RAG-specific exceptions using the unified system."""
        correlation_id = getattr(request.state, "correlation_id", exc.correlation_id)
        
        # Log the error
        if self.error_logging:
            severity = "error" if exc.status_code >= 500 else "warning"
            if exc.status_code >= 500:
                severity = "critical"
            
            self.error_logger.log_error(
                f"{exc.error_code}: {exc.message}",
                error=exc,
                severity=severity,
                extra={
                    "correlation_id": correlation_id,
                    "path": request.url.path,
                    "method": request.method,
                    "context": exc.context
                }
            )
        
        # Create error response using our standard format
        error_response = create_error_response(
            error_code=exc.error_code,
            message=exc.message,
            context=exc.context,
            correlation_id=correlation_id
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump()
        )

    async def handle_http_exception(
        self,
        request: Request,
        exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions using standard format."""
        correlation_id = getattr(request.state, "correlation_id", None)
        
        # Map HTTP status to error code
        error_code = f"HTTP_{exc.status_code}"
        if exc.status_code == 404:
            error_code = "RESOURCE_NOT_FOUND"
        elif exc.status_code == 403:
            error_code = "ACCESS_DENIED"
        elif exc.status_code == 401:
            error_code = "UNAUTHORIZED"
        elif exc.status_code >= 500:
            error_code = "INTERNAL_ERROR"

        if self.error_logging:
            self.error_logger.log_error(
                f"HTTP Error {exc.status_code}: {exc.detail}",
                error=exc,
                severity="error" if exc.status_code >= 500 else "warning",
                extra={"correlation_id": correlation_id}
            )

        error_response = create_error_response(
            error_code=error_code,
            message=str(exc.detail),
            context={"path": str(request.url.path), "method": request.method},
            correlation_id=correlation_id
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump()
        )

    async def handle_validation_exception(
        self,
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle FastAPI validation errors using standard format."""
        correlation_id = getattr(request.state, "correlation_id", None)
        
        # Format field errors for our standard response
        field_errors = []
        for error in exc.errors():
            field_errors.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "code": "VALIDATION_001",
                "message": error["msg"]
            })

        if self.error_logging:
            self.error_logger.log_error(
                "Request Validation Error",
                error=exc,
                severity="warning",
                extra={
                    "errors": exc.errors(),
                    "correlation_id": correlation_id
                }
            )

        validation_response = create_validation_error_response(
            error_code="VALIDATION_001",
            message="Request validation failed",
            field_errors=field_errors,
            correlation_id=correlation_id
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=validation_response.model_dump()
        )

    async def handle_pydantic_validation_error(
        self,
        request: Request,
        exc: ValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors using standard format."""
        correlation_id = getattr(request.state, "correlation_id", None)
        
        # Format field errors for our standard response
        field_errors = []
        for error in exc.errors():
            field_errors.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "code": "VALIDATION_002", 
                "message": error["msg"]
            })

        if self.error_logging:
            self.error_logger.log_error(
                "Pydantic Validation Error",
                error=exc,
                severity="warning",
                extra={
                    "errors": exc.errors(),
                    "correlation_id": correlation_id
                }
            )

        validation_response = create_validation_error_response(
            error_code="VALIDATION_002",
            message="Data validation failed",
            field_errors=field_errors,
            correlation_id=correlation_id
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=validation_response.model_dump()
        )

    async def handle_general_exception(
        self,
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle general exceptions using standard format."""
        # Check for custom error handler
        if exc.__class__ in self.custom_error_handlers:
            return await self.custom_error_handlers[exc.__class__](request, exc)

        correlation_id = getattr(request.state, "correlation_id", None)
        
        # Try to map exception class to error code
        error_code = get_error_code_for_exception(exc.__class__.__name__)
        
        context = {
            "exception_type": exc.__class__.__name__,
            "path": str(request.url.path),
            "method": request.method
        }
        
        # Include traceback in development mode
        if self.include_traceback:
            context["traceback"] = traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__
            )

        if self.error_logging:
            self.error_logger.log_error(
                "Unhandled Exception",
                error=exc,
                severity="critical",
                extra={
                    "correlation_id": correlation_id,
                    "path": request.url.path,
                    "method": request.method
                }
            )

        error_response = create_error_response(
            error_code=error_code,
            message="An unexpected error occurred",
            context=context,
            correlation_id=correlation_id
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump()
        )

# Validation error formatting now handled in standard response creation

@asynccontextmanager
async def error_handling_context(request: Request, error_handler: ErrorHandler):
    """Context manager for handling errors in specific routes or blocks of code."""
    try:
        yield
    except Exception as e:
        await error_handler.handle_general_exception(request, e)

# Enhanced error handler for unified RAG exception system
# No legacy fallback patterns - all errors use standardized responses