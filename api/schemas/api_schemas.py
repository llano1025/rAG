from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, HttpUrl
import uuid

class APIKeyBase(BaseModel):
    name: str = Field(..., description="Name/description of the API key")
    permissions: List[str] = Field(..., description="List of permitted operations")
    expiration_date: Optional[datetime] = Field(None, description="Optional expiration date")

class APIKeyCreate(APIKeyBase):
    @validator('permissions')
    def validate_permissions(cls, v):
        valid_permissions = {
            'read:documents', 'write:documents', 'delete:documents',
            'read:vectors', 'write:vectors', 'delete:vectors',
            'manage:folders', 'manage:tags', 'read:analytics'
        }
        if not all(perm in valid_permissions for perm in v):
            raise ValueError(f"Invalid permissions. Valid values are: {valid_permissions}")
        return v

class APIKey(APIKeyBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key: str
    created_at: datetime
    created_by: str
    last_used: Optional[datetime] = None
    is_active: bool = True
    rate_limit: Optional[int] = Field(
        None, 
        description="Rate limit in requests per minute"
    )

    class Config:
        from_attributes = True

class APIKeyResponse(APIKey):
    masked_key: str = Field(..., description="Partially masked API key")
    total_requests: int
    requests_today: int

class APIUsageLog(BaseModel):
    id: str
    api_key_id: str
    timestamp: datetime
    endpoint: str
    method: str
    response_time: float
    status_code: int
    ip_address: str
    request_size: int
    response_size: int
    error_details: Optional[dict] = None

class APIUsageStats(BaseModel):
    total_requests: int
    requests_by_endpoint: dict
    average_response_time: float
    error_rate: float
    usage_by_key: dict
    usage_by_ip: dict
    daily_usage: List[dict]
    peak_usage_time: datetime
    bandwidth_used: int

class APIRateLimit(BaseModel):
    api_key_id: str
    calls_remaining: int
    reset_at: datetime
    total_limit: int
    window_seconds: int

class APIError(BaseModel):
    code: str = Field(..., description="Error code identifying the error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class APIWebhook(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: HttpUrl = Field(..., description="Webhook destination URL")
    events: List[str] = Field(..., description="List of events to subscribe to")
    is_active: bool = True
    secret_key: str = Field(..., description="Secret key for webhook signature")
    retry_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_retries": 3,
            "initial_delay": 5,
            "max_delay": 300
        }
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    @validator('events')
    def validate_events(cls, v):
        valid_events = {
            'document.created', 'document.updated', 'document.deleted',
            'vector.created', 'vector.updated', 'vector.deleted',
            'search.performed', 'error.occurred'
        }
        if not all(event in valid_events for event in v):
            raise ValueError(f"Invalid events. Valid values are: {valid_events}")
        return v

class WebhookDeliveryLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    webhook_id: str
    event_type: str
    payload: Dict[str, Any]
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    delivery_duration: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool
    retry_count: int = 0
    error_message: Optional[str] = None

class APISettings(BaseModel):
    rate_limiting_enabled: bool = True
    default_rate_limit: int = Field(60, description="Default requests per minute")
    max_request_size_bytes: int = Field(10_485_760, description="10MB default max request size")
    webhook_timeout_seconds: int = Field(30, description="Webhook delivery timeout")
    api_version: str = Field("v1", description="Current API version")
    maintenance_mode: bool = False
    cors_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "allowed_origins": ["*"],
            "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
            "allowed_headers": ["*"],
            "max_age": 86400
        }
    )

class BatchOperation(BaseModel):
    operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operations: List[Dict[str, Any]] = Field(..., description="List of operations to perform")
    total_operations: int
    completed_operations: int = 0
    failed_operations: int = 0
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = Field("pending", description="Status of batch operation")
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        from_attributes = True

class APIMetrics(BaseModel):
    uptime_seconds: int
    total_requests_handled: int
    current_requests_per_second: float
    average_response_time_ms: float
    error_rate_percent: float
    active_connections: int
    memory_usage_bytes: int
    cpu_usage_percent: float
    disk_usage_percent: float
    cache_hit_ratio: float
    active_api_keys: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class APIHealthCheck(BaseModel):
    status: str = Field(..., description="overall health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="Status of individual services"
    )
    dependencies: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="Status of external dependencies"
    )
    performance_metrics: APIMetrics

class APICredentials(BaseModel):
    client_id: str = Field(..., description="OAuth2 client ID")
    client_secret: str = Field(..., description="OAuth2 client secret")
    scope: List[str] = Field(..., description="OAuth2 scopes")
    redirect_uris: List[HttpUrl] = Field(..., description="OAuth2 redirect URIs")
    grant_types: List[str] = Field(
        default=["authorization_code", "refresh_token"],
        description="Allowed OAuth2 grant types"
    )

    @validator('grant_types')
    def validate_grant_types(cls, v):
        valid_grants = {"authorization_code", "client_credentials", "refresh_token"}
        if not all(grant in valid_grants for grant in v):
            raise ValueError(f"Invalid grant types. Valid values are: {valid_grants}")
        return v

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: str