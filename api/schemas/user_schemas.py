from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, validator

class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: str = Field(..., min_length=1, max_length=100)
    role: str = Field(..., description="User role (admin, editor, viewer)")
    is_active: bool = True

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(...)

    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    role: Optional[str] = None
    is_active: Optional[bool] = None

class UserPermissions(BaseModel):
    can_create_documents: bool = False
    can_edit_documents: bool = False
    can_delete_documents: bool = False
    can_manage_folders: bool = False
    can_manage_tags: bool = False
    can_manage_users: bool = False
    can_view_analytics: bool = False
    can_manage_api_keys: bool = False
    folder_restrictions: List[str] = Field(default_factory=list)
    tag_restrictions: List[str] = Field(default_factory=list)

class User(UserBase):
    id: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    permissions: UserPermissions
    metadata: dict = Field(default_factory=dict)

    class Config:
        from_attributes = True

class UserActivityLog(BaseModel):
    id: str
    user_id: str
    activity_type: str
    timestamp: datetime
    details: dict
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class UserSession(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Name for the API key")
    description: Optional[str] = Field(None, max_length=500, description="Optional description")
    expires_days: Optional[int] = Field(None, gt=0, le=365, description="Days until expiration (max 365)")

class APIKeyResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    key_prefix: str
    is_active: bool
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True