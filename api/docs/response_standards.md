# API Response Standards

This document outlines the standardized response formats for all API endpoints in the RAG system.

## Overview

All API responses must follow consistent formats to ensure:
- Predictable client integration
- Consistent error handling  
- Proper correlation tracking
- Standardized metadata

## Response Types

### 1. Standard Success Response

All successful operations return a `StandardResponse` format:

```json
{
  "success": true,
  "data": {
    // Response payload
  },
  "message": "Operation completed successfully",
  "metadata": {
    "timestamp": "2025-08-14T10:30:00Z",
    "correlation_id": "req_123456789",
    "api_version": "v1",
    "response_time_ms": 150.5
  }
}
```

**Usage in Controllers:**
```python
from api.schemas.responses import create_success_response

return create_success_response(
    data={"user_id": 123, "username": "john_doe"},
    message="User created successfully"
)
```

### 2. Paginated Response

For list endpoints with pagination:

```json
{
  "success": true,
  "data": [
    // Array of items
  ],
  "pagination": {
    "current_page": 1,
    "page_size": 20,
    "total_items": 150,
    "total_pages": 8,
    "has_next": true,
    "has_previous": false,
    "next_page": 2,
    "previous_page": null
  },
  "metadata": {
    "timestamp": "2025-08-14T10:30:00Z",
    "correlation_id": "req_123456789",
    "api_version": "v1"
  }
}
```

**Usage in Controllers:**
```python
from api.schemas.responses import create_paginated_response

return create_paginated_response(
    data=documents,
    current_page=page,
    page_size=size,
    total_items=total_count
)
```

### 3. Authentication Response

For authentication endpoints:

```json
{
  "success": true,
  "message": "Authentication successful",
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_here",
  "user_id": "123",
  "permissions": ["read:documents", "write:documents"],
  "metadata": {
    "timestamp": "2025-08-14T10:30:00Z",
    "correlation_id": "req_123456789",
    "api_version": "v1"
  }
}
```

### 4. Error Response

All errors return a standardized format:

```json
{
  "success": false,
  "error": {
    "error_code": "DOC_001",
    "message": "Document not found",
    "field": null,
    "context": {
      "document_id": "123"
    }
  },
  "metadata": {
    "timestamp": "2025-08-14T10:30:00Z",
    "correlation_id": "req_123456789",
    "api_version": "v1"
  }
}
```

## Route Implementation Standards

### 1. Response Model Declaration

All routes must declare their response model:

```python
from api.schemas.responses import StandardResponse, PaginatedResponse, AuthResponse

@router.get("/users", response_model=PaginatedResponse)
async def list_users():
    # Implementation
    pass

@router.post("/users", response_model=StandardResponse)
async def create_user():
    # Implementation
    pass

@router.post("/auth/login", response_model=AuthResponse)
async def login():
    # Implementation
    pass
```

### 2. Controller Response Format

Controllers must return properly formatted responses:

```python
# ✅ Correct - Using helper functions
return create_success_response(
    data={"id": 123, "name": "example"},
    message="Operation successful"
)

# ❌ Incorrect - Raw dictionary
return {"id": 123, "name": "example"}
```

### 3. Error Handling

Use unified exception system instead of HTTPException:

```python
# ✅ Correct - Using unified exceptions
from utils.exceptions import DocumentNotFoundException

raise DocumentNotFoundException(
    message="Document not found",
    document_id=str(doc_id)
)

# ❌ Incorrect - Direct HTTPException
raise HTTPException(
    status_code=404,
    detail="Document not found"
)
```

## Helper Utilities

### Route Helpers

Use the route helpers for common patterns:

```python
from api.utils.route_helpers import ResponsePatterns

# Common response patterns
return ResponsePatterns.created(data=new_user)
return ResponsePatterns.updated(data=updated_user)
return ResponsePatterns.deleted()
return ResponsePatterns.list_response(items=users)
```

### Response Validation

The system includes automatic response validation:

```python
from api.utils.route_helpers import route_response_decorator

@route_response_decorator(StandardResponse)
@router.get("/example")
async def example():
    return success_response({"key": "value"})
```

## Migration Guide

### Updating Existing Routes

1. **Update Imports**:
   ```python
   from api.schemas.responses import StandardResponse, create_success_response
   ```

2. **Update Response Models**:
   ```python
   @router.get("/example", response_model=StandardResponse)
   ```

3. **Update Return Statements**:
   ```python
   # Before
   return {"data": result}
   
   # After  
   return create_success_response(data=result, message="Success")
   ```

4. **Update Error Handling**:
   ```python
   # Before
   raise HTTPException(status_code=404, detail="Not found")
   
   # After
   raise DocumentNotFoundException(message="Document not found")
   ```

## Validation Rules

The response validation middleware checks:

1. **Structure Compliance**: All responses follow declared format
2. **Required Fields**: Responses include all required fields
3. **Metadata Presence**: All responses include proper metadata
4. **Error Format**: Error responses follow standard error schema

## Common Patterns

### List Endpoints
```python
@router.get("/documents", response_model=PaginatedResponse)
async def list_documents(page: int = 1, size: int = 20):
    documents = get_documents(page, size)
    total = count_documents()
    
    return create_paginated_response(
        data=documents,
        current_page=page,
        page_size=size,
        total_items=total
    )
```

### Create Endpoints
```python
@router.post("/documents", response_model=StandardResponse)
async def create_document(doc_data: DocumentCreate):
    document = create_document_service(doc_data)
    
    return ResponsePatterns.created(
        data=document,
        message="Document created successfully"
    )
```

### Update Endpoints
```python
@router.put("/documents/{doc_id}", response_model=StandardResponse)
async def update_document(doc_id: int, doc_data: DocumentUpdate):
    document = update_document_service(doc_id, doc_data)
    
    return ResponsePatterns.updated(
        data=document,
        message="Document updated successfully"
    )
```

### Delete Endpoints
```python
@router.delete("/documents/{doc_id}", response_model=StandardResponse)
async def delete_document(doc_id: int):
    delete_document_service(doc_id)
    
    return ResponsePatterns.deleted(
        message="Document deleted successfully"
    )
```

## Best Practices

1. **Consistent Messaging**: Use clear, consistent success messages
2. **Meaningful Data**: Include relevant data in response payload
3. **Proper Correlation**: Always include correlation IDs for tracking
4. **Error Context**: Provide helpful context in error responses
5. **Performance Info**: Include timing information where relevant

## Testing Response Formats

All responses can be validated using the response validation middleware:

```python
# Enable validation in development/testing
app.add_middleware(ResponseValidationMiddleware, validate_responses=True)
```

This ensures all endpoints comply with the standard formats before deployment.