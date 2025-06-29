"""
SQLAlchemy database models for authentication and user management.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Table, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import secrets
from datetime import datetime, timedelta

from .base import Base, TimestampMixin, SoftDeleteMixin

# Association table for many-to-many relationship between users and roles
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)

# Association table for many-to-many relationship between roles and permissions
role_permissions = Table(
    'role_permissions', 
    Base.metadata,
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True),
    Column('permission_id', Integer, ForeignKey('permissions.id'), primary_key=True)
)

class UserRoleEnum(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

class PermissionEnum(str, Enum):
    """Permission enumeration."""
    # Document permissions
    READ_DOCUMENTS = "read_documents"
    WRITE_DOCUMENTS = "write_documents"
    DELETE_DOCUMENTS = "delete_documents"
    
    # Folder permissions
    READ_FOLDERS = "read_folders"
    WRITE_FOLDERS = "write_folders"
    DELETE_FOLDERS = "delete_folders"
    
    # Tag permissions
    READ_TAGS = "read_tags"
    WRITE_TAGS = "write_tags"
    DELETE_TAGS = "delete_tags"
    
    # User management permissions
    READ_USERS = "read_users"
    WRITE_USERS = "write_users"
    DELETE_USERS = "delete_users"
    
    # Analytics permissions
    READ_ANALYTICS = "read_analytics"
    
    # API key permissions
    READ_API_KEYS = "read_api_keys"
    WRITE_API_KEYS = "write_api_keys"
    DELETE_API_KEYS = "delete_api_keys"
    
    # System permissions
    SYSTEM_ADMIN = "system_admin"

class User(Base, TimestampMixin, SoftDeleteMixin):
    """User model for authentication and user management."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    
    # User status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Profile information
    profile_picture = Column(String(255), nullable=True)
    bio = Column(Text, nullable=True)
    phone = Column(String(20), nullable=True)
    timezone = Column(String(50), default="UTC", nullable=False)
    
    # Authentication metadata
    last_login = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Email verification
    email_verification_token = Column(String(255), nullable=True)
    email_verification_expires = Column(DateTime(timezone=True), nullable=True)
    
    # Password reset
    reset_password_token = Column(String(255), nullable=True)
    reset_password_expires = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    roles = relationship("UserRole", secondary=user_roles, back_populates="users")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    activity_logs = relationship("UserActivityLog", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', username='{self.username}')>"
    
    def has_permission(self, permission: PermissionEnum) -> bool:
        """Check if user has a specific permission."""
        if self.is_superuser:
            return True
            
        for role in self.roles:
            if role.has_permission(permission):
                return True
        return False
    
    def has_role(self, role_name: UserRoleEnum) -> bool:
        """Check if user has a specific role."""
        return any(role.name == role_name for role in self.roles)
    
    def is_account_locked(self) -> bool:
        """Check if account is locked due to failed login attempts."""
        return self.locked_until and self.locked_until > datetime.utcnow()

class UserRole(Base, TimestampMixin):
    """User role model for role-based access control."""
    
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(SQLEnum(UserRoleEnum), unique=True, nullable=False)
    display_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
    
    def __repr__(self):
        return f"<UserRole(id={self.id}, name='{self.name}')>"
    
    def has_permission(self, permission: PermissionEnum) -> bool:
        """Check if role has a specific permission."""
        return any(perm.name == permission for perm in self.permissions)

class Permission(Base, TimestampMixin):
    """Permission model for granular access control."""
    
    __tablename__ = "permissions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(SQLEnum(PermissionEnum), unique=True, nullable=False)
    display_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    resource = Column(String(50), nullable=False)  # documents, folders, users, etc.
    action = Column(String(50), nullable=False)    # read, write, delete, etc.
    
    # Relationships
    roles = relationship("UserRole", secondary=role_permissions, back_populates="permissions")
    
    def __repr__(self):
        return f"<Permission(id={self.id}, name='{self.name}')>"

class UserSession(Base, TimestampMixin):
    """User session model for session management."""
    
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token = Column(String(255), unique=True, nullable=True, index=True)
    
    # Session metadata
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    device_info = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    
    # Session validity
    expires_at = Column(DateTime(timezone=True), nullable=False)
    refresh_expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Session tracking
    last_activity = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    login_method = Column(String(50), default="password", nullable=False)  # password, api_key, sso
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, expires_at='{self.expires_at}')>"
    
    def is_valid(self) -> bool:
        """Check if session is still valid."""
        return self.is_active and self.expires_at > datetime.utcnow()
    
    def is_refresh_valid(self) -> bool:
        """Check if refresh token is still valid."""
        return (self.refresh_token and 
                self.refresh_expires_at and 
                self.refresh_expires_at > datetime.utcnow())
    
    @classmethod
    def generate_session_token(cls) -> str:
        """Generate a secure session token."""
        return secrets.token_urlsafe(32)
    
    @classmethod
    def generate_refresh_token(cls) -> str:
        """Generate a secure refresh token."""
        return secrets.token_urlsafe(32)

class APIKey(Base, TimestampMixin, SoftDeleteMixin):
    """API key model for programmatic access."""
    
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # API key details
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    key_prefix = Column(String(10), nullable=False)  # First 8 chars for identification
    
    # API key status
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Usage tracking
    last_used = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)
    rate_limit = Column(Integer, default=1000, nullable=False)  # requests per hour
    
    # Permissions (JSON field in production would be better)
    allowed_endpoints = Column(Text, nullable=True)  # JSON string of allowed endpoints
    ip_whitelist = Column(Text, nullable=True)       # JSON string of allowed IPs
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.name}', user_id={self.user_id})>"
    
    def is_valid(self) -> bool:
        """Check if API key is still valid."""
        if not self.is_active or self.is_deleted:
            return False
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        return True
    
    @classmethod
    def generate_key(cls) -> tuple[str, str]:
        """Generate API key and return (full_key, hash)."""
        key = f"rag_{secrets.token_urlsafe(32)}"
        key_hash = hash(key)  # In production, use proper hashing like bcrypt
        return key, str(key_hash)
    
    def get_prefix(self, full_key: str) -> str:
        """Get key prefix for identification."""
        return full_key[:8] if len(full_key) >= 8 else full_key

class UserActivityLog(Base, TimestampMixin):
    """User activity log for audit trail."""
    
    __tablename__ = "user_activity_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Activity details
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(100), nullable=True)
    details = Column(Text, nullable=True)  # JSON string for additional details
    
    # Request metadata
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    request_id = Column(String(100), nullable=True)
    
    # Status
    status = Column(String(20), default="success", nullable=False)  # success, failed, error
    error_message = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="activity_logs")
    
    def __repr__(self):
        return f"<UserActivityLog(id={self.id}, user_id={self.user_id}, action='{self.action}')>"