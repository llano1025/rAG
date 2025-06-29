"""
Authentication controller for user management and security operations.
"""

from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List
import secrets
import hashlib

from database.connection import get_db
from database.models import User, UserSession, APIKey, UserRole, UserActivityLog, PermissionEnum
from api.schemas.user_schemas import UserCreate, UserUpdate, User as UserSchema
from utils.security.encryption import EncryptionManager
from utils.security.audit_logger import AuditLogger
from config import get_settings

settings = get_settings()

class AuthController:
    def __init__(
        self,
        encryption: EncryptionManager,
        audit_logger: AuditLogger
    ):
        self.encryption = encryption
        self.audit_logger = audit_logger
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    async def process_credentials(
        self,
        credentials: dict
    ) -> dict:
        """Process and verify user credentials."""
        try:
            # Encrypt sensitive credential data
            encrypted_creds = self.encryption.encrypt_data(credentials)
            
            # Process authentication
            auth_result = await self._verify_credentials(encrypted_creds)
            
            # Log authentication attempt
            await self.audit_logger.log(
                action="authentication",
                resource_type="user",
                user_id=auth_result.get("user_id"),
                details={"status": auth_result.get("status")}
            )
            
            return auth_result
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))

    async def _verify_credentials(self, encrypted_creds: bytes) -> dict:
        """Internal method to verify credentials."""
        # Decrypt credentials
        credentials = self.encryption.decrypt_data(encrypted_creds)
        email = credentials.get("email")
        password = credentials.get("password")
        
        if not email or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email and password required"
            )
        
        # This would need database session - simplified for now
        return {"status": "verified", "user_id": None}

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            email: str = payload.get("sub")
            if email is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def get_user_by_email(self, email: str, db: Session) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email, User.is_deleted == False).first()

    async def get_user_by_id(self, user_id: int, db: Session) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id, User.is_deleted == False).first()

    async def create_user(self, user_data: UserCreate, db: Session) -> User:
        """Create a new user."""
        # Check if user already exists
        existing_user = await self.get_user_by_email(user_data.email, db)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Check username availability
        existing_username = db.query(User).filter(User.username == user_data.username).first()
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Create user
        hashed_password = self.hash_password(user_data.password)
        db_user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            is_active=True,
            is_verified=False,  # Require email verification
        )
        
        # Assign default user role
        default_role = db.query(UserRole).filter(UserRole.name == "user").first()
        if default_role:
            db_user.roles = [default_role]
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        # Log user creation
        await self.audit_logger.log(
            action="user_created",
            resource_type="user",
            resource_id=str(db_user.id),
            user_id=db_user.id,
            details={"email": user_data.email, "username": user_data.username}
        )
        
        return db_user

    async def authenticate_user(self, email: str, password: str, db: Session) -> Optional[User]:
        """Authenticate user with email and password."""
        user = await self.get_user_by_email(email, db)
        if not user:
            return None
        
        # Check if account is locked
        if user.is_account_locked():
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account locked until {user.locked_until}"
            )
        
        if not self.verify_password(password, user.hashed_password):
            # Increment failed login attempts
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
            db.commit()
            
            # Log failed authentication
            await self.audit_logger.log(
                action="authentication_failed",
                resource_type="user",
                resource_id=str(user.id),
                user_id=user.id,
                details={"reason": "invalid_password", "attempts": user.failed_login_attempts}
            )
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user

    async def update_user(self, user_id: int, user_data: UserUpdate, db: Session) -> User:
        """Update user information."""
        user = await self.get_user_by_id(user_id, db)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update fields
        update_data = user_data.dict(exclude_unset=True)
        
        # Handle password update
        if "password" in update_data:
            update_data["hashed_password"] = self.hash_password(update_data.pop("password"))
            update_data["password_changed_at"] = datetime.utcnow()
        
        for field, value in update_data.items():
            setattr(user, field, value)
        
        db.commit()
        db.refresh(user)
        
        # Log user update
        await self.audit_logger.log(
            action="user_updated",
            resource_type="user",
            resource_id=str(user.id),
            user_id=user_id,
            details={"updated_fields": list(update_data.keys())}
        )
        
        return user

    async def delete_user(self, user_id: int, db: Session) -> bool:
        """Soft delete a user."""
        user = await self.get_user_by_id(user_id, db)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Soft delete
        user.is_deleted = True
        user.deleted_at = datetime.utcnow()
        user.is_active = False
        
        db.commit()
        
        # Log user deletion
        await self.audit_logger.log(
            action="user_deleted",
            resource_type="user",
            resource_id=str(user.id),
            user_id=user_id,
            details={"email": user.email}
        )
        
        return True

    async def list_users(self, db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """List all active users."""
        return db.query(User).filter(User.is_deleted == False).offset(skip).limit(limit).all()

    async def update_last_login(self, user_id: int, db: Session) -> None:
        """Update user's last login timestamp."""
        user = await self.get_user_by_id(user_id, db)
        if user:
            user.last_login = datetime.utcnow()
            db.commit()

    async def create_session(
        self, 
        user: User, 
        ip_address: str = None, 
        user_agent: str = None,
        db: Session = None
    ) -> UserSession:
        """Create a new user session."""
        session = UserSession(
            user_id=user.id,
            session_token=UserSession.generate_session_token(),
            refresh_token=UserSession.generate_refresh_token(),
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.utcnow() + timedelta(hours=24),
            refresh_expires_at=datetime.utcnow() + timedelta(days=30),
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return session

    async def validate_session(self, session_token: str, db: Session) -> Optional[UserSession]:
        """Validate and return active session."""
        session = db.query(UserSession).filter(
            UserSession.session_token == session_token,
            UserSession.is_active == True
        ).first()
        
        if session and session.is_valid():
            # Update last activity
            session.last_activity = datetime.utcnow()
            db.commit()
            return session
        
        return None

    async def invalidate_session(self, session_token: str, db: Session) -> bool:
        """Invalidate a user session."""
        session = db.query(UserSession).filter(
            UserSession.session_token == session_token
        ).first()
        
        if session:
            session.is_active = False
            db.commit()
            return True
        
        return False

    # API Key Management
    async def create_api_key(
        self, 
        user_id: int, 
        name: str, 
        description: str = None,
        expires_days: int = None,
        db: Session = None
    ) -> tuple[str, APIKey]:
        """Create new API key for user."""
        user = await self.get_user_by_id(user_id, db)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Generate API key
        full_key, key_hash = APIKey.generate_key()
        
        api_key = APIKey(
            user_id=user_id,
            name=name,
            description=description,
            key_hash=key_hash,
            key_prefix=full_key[:8],
            expires_at=datetime.utcnow() + timedelta(days=expires_days) if expires_days else None
        )
        
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        
        # Log API key creation
        await self.audit_logger.log(
            action="api_key_created",
            resource_type="api_key",
            resource_id=str(api_key.id),
            user_id=user_id,
            details={"name": name, "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None}
        )
        
        return full_key, api_key

    async def validate_api_key(self, api_key: str, db: Session) -> Optional[User]:
        """Validate API key and return associated user."""
        key_hash = str(hash(api_key))
        
        api_key_obj = db.query(APIKey).filter(
            APIKey.key_hash == key_hash,
            APIKey.is_deleted == False
        ).first()
        
        if api_key_obj and api_key_obj.is_valid():
            # Update usage tracking
            api_key_obj.last_used = datetime.utcnow()
            api_key_obj.usage_count += 1
            db.commit()
            
            return api_key_obj.user
        
        return None

    async def revoke_api_key(self, api_key_id: int, user_id: int, db: Session) -> bool:
        """Revoke an API key."""
        api_key = db.query(APIKey).filter(
            APIKey.id == api_key_id,
            APIKey.user_id == user_id,
            APIKey.is_deleted == False
        ).first()
        
        if api_key:
            api_key.is_active = False
            api_key.is_deleted = True
            api_key.deleted_at = datetime.utcnow()
            db.commit()
            
            # Log API key revocation
            await self.audit_logger.log(
                action="api_key_revoked",
                resource_type="api_key",
                resource_id=str(api_key.id),
                user_id=user_id,
                details={"name": api_key.name}
            )
            
            return True
        
        return False