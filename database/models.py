"""
SQLAlchemy database models for authentication and user management.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Table, Enum as SQLEnum, Float
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
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    vector_indices = relationship("VectorIndex", back_populates="user", cascade="all, delete-orphan")
    search_queries = relationship("SearchQuery", back_populates="user", cascade="all, delete-orphan")
    
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
    
    @property
    def is_admin(self) -> bool:
        """Check if user is admin (alias for is_superuser)."""
        return self.is_superuser or self.has_role(UserRoleEnum.ADMIN)

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


# Document and Vector Database Models

class DocumentStatusEnum(str, Enum):
    """Document processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

class Document(Base, TimestampMixin, SoftDeleteMixin):
    """Document model for storing document metadata and content."""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Document identification
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=True)  # Path to stored file
    file_hash = Column(String(64), nullable=False, index=True)  # SHA-256 hash
    
    # Document metadata
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    content_type = Column(String(100), nullable=False)  # MIME type
    file_size = Column(Integer, nullable=False)  # Size in bytes
    
    # Processing status
    status = Column(SQLEnum(DocumentStatusEnum), default=DocumentStatusEnum.PENDING, nullable=False)
    processing_error = Column(Text, nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Document content
    extracted_text = Column(Text, nullable=True)
    document_metadata = Column(Text, nullable=True)  # JSON string for additional metadata
    language = Column(String(10), nullable=True)  # Language code (e.g., 'en', 'es')
    
    # Organization
    folder_path = Column(String(500), nullable=True)
    tags = Column(Text, nullable=True)  # JSON array of tags
    
    # Version control
    version = Column(Integer, default=1, nullable=False)
    parent_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    # Access control
    is_public = Column(Boolean, default=False, nullable=False)
    access_permissions = Column(Text, nullable=True)  # JSON for fine-grained permissions
    
    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    versions = relationship("Document", backref="parent_document", remote_side=[id])
    vector_indices = relationship("VectorIndex", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"
    
    def get_tag_list(self) -> list:
        """Get tags as a list."""
        if self.tags:
            import json
            return json.loads(self.tags)
        return []
    
    def set_tag_list(self, tags: list):
        """Set tags from a list."""
        import json
        self.tags = json.dumps(tags)
    
    def can_access(self, user: User) -> bool:
        """Check if user can access this document."""
        # Owner can always access
        if self.user_id == user.id:
            return True
        
        # Public documents are accessible to all authenticated users
        if self.is_public:
            return True
        
        # Admins can access all documents
        if user.is_superuser or user.has_role("admin"):
            return True
        
        # Check fine-grained permissions from access_permissions JSON
        if self.access_permissions:
            import json
            try:
                permissions = json.loads(self.access_permissions)
                user_permissions = permissions.get('users', [])
                if str(user.id) in user_permissions:
                    return True
                
                # Check role-based permissions
                role_permissions = permissions.get('roles', [])
                for role in user.roles:
                    if role.name.value in role_permissions:
                        return True
            except (json.JSONDecodeError, KeyError):
                pass
        
        return False
    
    def get_metadata_dict(self) -> dict:
        """Get document metadata as dictionary."""
        if self.document_metadata:
            import json
            try:
                return json.loads(self.document_metadata)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_metadata(self, metadata: dict):
        """Set document metadata from dictionary."""
        import json
        self.document_metadata = json.dumps(metadata)
    
    def set_tags(self, tags: list):
        """Set document tags from list (alias for set_tag_list)."""
        self.set_tag_list(tags)

class DocumentChunk(Base, TimestampMixin):
    """Document chunk model for storing text chunks and their embeddings."""
    
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Chunk identification
    chunk_index = Column(Integer, nullable=False)  # Index within document
    chunk_id = Column(String(100), nullable=False, index=True)  # Unique chunk identifier
    
    # Chunk content
    text = Column(Text, nullable=False)
    text_length = Column(Integer, nullable=False)
    
    # Chunk positioning
    start_char = Column(Integer, nullable=True)  # Start position in original text
    end_char = Column(Integer, nullable=True)    # End position in original text
    page_number = Column(Integer, nullable=True)  # Page number if applicable
    
    # Context information
    context_before = Column(Text, nullable=True)  # Text before this chunk
    context_after = Column(Text, nullable=True)   # Text after this chunk
    section_title = Column(String(500), nullable=True)  # Section/chapter title
    
    # Embedding metadata
    embedding_model = Column(String(100), nullable=True)
    embedding_version = Column(String(50), nullable=True)
    content_embedding_id = Column(String(100), nullable=True, index=True)  # Qdrant point ID
    context_embedding_id = Column(String(100), nullable=True, index=True)   # Qdrant point ID
    
    # Chunk metadata
    document_metadata = Column(Text, nullable=True)  # JSON string for additional metadata
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"
    
    def get_context_text(self) -> str:
        """Get full context including before, chunk, and after text."""
        parts = []
        if self.context_before:
            parts.append(self.context_before)
        parts.append(self.text)
        if self.context_after:
            parts.append(self.context_after)
        return " ".join(parts)

class VectorIndex(Base, TimestampMixin):
    """Vector index model for managing FAISS and Qdrant indices."""
    
    __tablename__ = "vector_indices"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)  # Null for global indices
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Index identification
    index_name = Column(String(100), nullable=False, index=True)
    index_type = Column(String(50), nullable=False)  # 'content', 'context', 'combined'
    
    # Index configuration
    embedding_model = Column(String(100), nullable=False)
    embedding_dimension = Column(Integer, nullable=False)
    similarity_metric = Column(String(20), default="cosine", nullable=False)
    
    # FAISS configuration
    faiss_index_type = Column(String(50), nullable=False)  # 'FLAT', 'HNSW', 'IVF', etc.
    faiss_index_path = Column(String(500), nullable=True)  # Path to saved FAISS index
    faiss_index_params = Column(Text, nullable=True)  # JSON for FAISS parameters
    
    # Qdrant configuration
    qdrant_collection_name = Column(String(100), nullable=False)
    qdrant_config = Column(Text, nullable=True)  # JSON for Qdrant configuration
    
    # Index statistics
    total_vectors = Column(Integer, default=0, nullable=False)
    index_size_bytes = Column(Integer, default=0, nullable=False)
    last_updated = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Index status
    is_active = Column(Boolean, default=True, nullable=False)
    build_status = Column(String(20), default="building", nullable=False)  # building, ready, error
    error_message = Column(Text, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="vector_indices")
    user = relationship("User", back_populates="vector_indices")
    
    def __repr__(self):
        return f"<VectorIndex(id={self.id}, name='{self.index_name}', type='{self.index_type}')>"
    
    def get_faiss_params(self) -> dict:
        """Get FAISS parameters as dictionary."""
        if self.faiss_index_params:
            import json
            return json.loads(self.faiss_index_params)
        return {}
    
    def set_faiss_params(self, params: dict):
        """Set FAISS parameters from dictionary."""
        import json
        self.faiss_index_params = json.dumps(params)
    
    def get_qdrant_config(self) -> dict:
        """Get Qdrant configuration as dictionary."""
        if self.qdrant_config:
            import json
            return json.loads(self.qdrant_config)
        return {}
    
    def set_qdrant_config(self, config: dict):
        """Set Qdrant configuration from dictionary."""
        import json
        self.qdrant_config = json.dumps(config)

class SearchQuery(Base, TimestampMixin):
    """Search query model for tracking search analytics and caching."""
    
    __tablename__ = "search_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Null for anonymous
    
    # Query details
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), nullable=False, index=True)  # Hash for deduplication
    query_type = Column(String(50), default="semantic", nullable=False)  # semantic, keyword, hybrid
    
    # Search parameters
    search_filters = Column(Text, nullable=True)  # JSON for search filters
    max_results = Column(Integer, default=10, nullable=False)
    similarity_threshold = Column(Float, nullable=True)
    
    # Results metadata
    results_count = Column(Integer, default=0, nullable=False)
    search_time_ms = Column(Integer, nullable=True)
    
    # User context
    session_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Query results (cached)
    cached_results = Column(Text, nullable=True)  # JSON for cached search results
    cache_expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="search_queries")
    
    def __repr__(self):
        return f"<SearchQuery(id={self.id}, query='{self.query_text[:50]}...', user_id={self.user_id})>"
    
    def is_cache_valid(self) -> bool:
        """Check if cached results are still valid."""
        if not self.cached_results or not self.cache_expires_at:
            return False
        return self.cache_expires_at > datetime.utcnow()
    
    def get_cached_results(self) -> list:
        """Get cached results as list."""
        if self.is_cache_valid() and self.cached_results:
            import json
            return json.loads(self.cached_results)
        return []
    
    def set_cached_results(self, results: list, cache_duration_hours: int = 24):
        """Set cached results with expiration."""
        import json
        from datetime import timedelta
        self.cached_results = json.dumps(results)
        self.cache_expires_at = datetime.utcnow() + timedelta(hours=cache_duration_hours)
    
    @classmethod
    def create_query_hash(cls, query_text: str, filters: dict = None) -> str:
        """Create hash for query deduplication."""
        import hashlib
        import json
        
        hash_input = query_text
        if filters:
            hash_input += json.dumps(filters, sort_keys=True)
        
        return hashlib.sha256(hash_input.encode()).hexdigest()