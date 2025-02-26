from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Dict, Any, Optional, List, Union
import logging
import traceback
import sys
from datetime import datetime
from pydantic import ValidationError
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class CustomHTTPException(StarletteHTTPException):
    """Custom HTTP Exception with additional fields."""
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.extra = extra or {}

class ErrorDetails:
    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.error_code = error_code
        self.details = details or {}
        self.request_id = request_id
        self.path = path
        self.method = method
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error details to dictionary format."""
        error_dict = {
            "status_code": self.status_code,
            "message": self.message,
            "error_type": self.error_type,
            "timestamp": self.timestamp
        }

        if self.error_code:
            error_dict["error_code"] = self.error_code
        if self.details:
            error_dict["details"] = self.details
        if self.request_id:
            error_dict["request_id"] = self.request_id
        if self.path:
            error_dict["path"] = self.path
        if self.method:
            error_dict["method"] = self.method

        return error_dict

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
        app: FastAPI,
        error_logging: bool = True,
        include_traceback: bool = False,
        error_handlers: Optional[Dict] = None
    ):
        self.app = app
        self.error_logging = error_logging
        self.include_traceback = include_traceback
        self.error_logger = ErrorLogger(logger)
        self.custom_error_handlers = error_handlers or {}
        self.setup_exception_handlers()

    def setup_exception_handlers(self):
        """Setup exception handlers for different types of errors."""
        @self.app.exception_handler(StarletteHTTPException)
        async def http_exception_handler(request: Request, exc: StarletteHTTPException):
            return await self.handle_http_exception(request, exc)

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            return await self.handle_validation_exception(request, exc)

        @self.app.exception_handler(ValidationError)
        async def pydantic_validation_handler(request: Request, exc: ValidationError):
            return await self.handle_pydantic_validation_error(request, exc)

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            return await self.handle_general_exception(request, exc)

    def register_error_handler(self, exception_class: type, handler: callable):
        """Register a custom error handler for a specific exception type."""
        self.custom_error_handlers[exception_class] = handler

    async def handle_http_exception(
        self,
        request: Request,
        exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions."""
        error_code = getattr(exc, 'error_code', None)
        extra = getattr(exc, 'extra', {})
        
        error_details = ErrorDetails(
            status_code=exc.status_code,
            message=str(exc.detail),
            error_type="http_error",
            error_code=error_code,
            details=extra,
            request_id=getattr(request.state, "request_id", None),
            path=str(request.url.path),
            method=request.method
        )

        if self.error_logging:
            self.error_logger.log_error(
                f"HTTP Error {exc.status_code}: {exc.detail}",
                error=exc,
                severity="error" if exc.status_code >= 500 else "warning",
                extra={"request_id": error_details.request_id}
            )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_details.to_dict()
        )

    async def handle_validation_exception(
        self,
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle FastAPI validation errors."""
        error_details = ErrorDetails(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Validation error",
            error_type="validation_error",
            details={"errors": self._format_validation_errors(exc.errors())},
            request_id=getattr(request.state, "request_id", None),
            path=str(request.url.path),
            method=request.method
        )

        if self.error_logging:
            self.error_logger.log_error(
                "Validation Error",
                error=exc,
                severity="warning",
                extra={
                    "errors": exc.errors(),
                    "request_id": error_details.request_id
                }
            )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_details.to_dict()
        )

    async def handle_pydantic_validation_error(
        self,
        request: Request,
        exc: ValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        error_details = ErrorDetails(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Data validation error",
            error_type="pydantic_validation_error",
            details={"errors": self._format_validation_errors(exc.errors())},
            request_id=getattr(request.state, "request_id", None),
            path=str(request.url.path),
            method=request.method
        )

        if self.error_logging:
            self.error_logger.log_error(
                "Pydantic Validation Error",
                error=exc,
                severity="warning",
                extra={
                    "errors": exc.errors(),
                    "request_id": error_details.request_id
                }
            )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_details.to_dict()
        )

    async def handle_general_exception(
        self,
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle general exceptions."""
        # Check for custom error handler
        if exc.__class__ in self.custom_error_handlers:
            return await self.custom_error_handlers[exc.__class__](request, exc)

        error_details = ErrorDetails(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Internal server error",
            error_type="server_error",
            request_id=getattr(request.state, "request_id", None),
            path=str(request.url.path),
            method=request.method
        )

        if self.error_logging:
            self.error_logger.log_error(
                "Unhandled Exception",
                error=exc,
                severity="critical",
                extra={
                    "request_id": error_details.request_id,
                    "path": request.url.path,
                    "method": request.method
                }
            )

        # Include traceback in development mode
        if self.include_traceback:
            error_details.details["traceback"] = traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__
            )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_details.to_dict()
        )

    def _format_validation_errors(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format validation errors into a more readable structure."""
        formatted_errors = []
        for error in errors:
            formatted_error = {
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            }
            if "ctx" in error:
                formatted_error["context"] = error["ctx"]
            formatted_errors.append(formatted_error)
        return formatted_errors

@asynccontextmanager
async def error_handling_context(request: Request, error_handler: ErrorHandler):
    """Context manager for handling errors in specific routes or blocks of code."""
    try:
        yield
    except Exception as e:
        await error_handler.handle_general_exception(request, e)

# Usage example:
"""
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, ValidationError
import uvicorn

app = FastAPI()
error_handler = ErrorHandler(
    app,
    error_logging=True,
    include_traceback=app.debug
)

# Custom error example
class DatabaseError(Exception):
    pass

# Register custom error handler
@error_handler.register_error_handler(DatabaseError)
async def handle_database_error(request: Request, exc: DatabaseError):
    error_details = ErrorDetails(
        status_code=503,
        message="Database error occurred",
        error_type="database_error",
        request_id=getattr(request.state, "request_id", None)
    )
    return JSONResponse(
        status_code=503,
        content=error_details.to_dict()
    )

# Example model
class User(BaseModel):
    username: str
    email: str
    age: int

@app.get("/test-http-error")
async def test_http_error():
    raise CustomHTTPException(
        status_code=400,
        detail="Test error",
        error_code="TEST_ERROR",
        extra={"additional": "info"}
    )

@app.post("/test-validation")
async def test_validation(user: User):
    return {"user": user.dict()}

@app.get("/test-database-error")
async def test_database_error():
    raise DatabaseError("Database connection failed")

@app.get("/test-context-manager")
async def test_context_manager(request: Request):
    async with error_handling_context(request, error_handler):
        # Your code that might raise an exception
        result = perform_risky_operation()
        return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""