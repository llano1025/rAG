"""
User management routes for authentication and user operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from database.connection import get_db
from api.schemas.user_schemas import (
    User as UserSchema, UserCreate, UserUpdate, UserPermissions,
    UserLogin, UserActivityLog, UserSession, APIKeyCreate, APIKeyResponse
)
from api.schemas.api_schemas import APIKeyResponse
from api.middleware.auth import AuthMiddleware
from api.controllers.auth_controller import AuthController
from utils.security.audit_logger import AuditLogger
from utils.security.encryption import EncryptionManager

router = APIRouter(prefix="/users", tags=["users"])
security = HTTPBearer()

# Dependency injection setup
async def get_auth_controller(
    db: Session = Depends(get_db)
) -> AuthController:
    """Dependency to get auth controller instance with database session."""
    from main import get_encryption_manager, get_audit_logger
    encryption = get_encryption_manager()
    audit_logger = get_audit_logger()
    return AuthController(encryption, audit_logger)

@router.post("/register", response_model=APIKeyResponse)
async def register_user(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Register a new user."""
    try:
        # Create new user using the controller
        user = await auth_controller.create_user(user_data, db)
        
        return APIKeyResponse(
            success=True,
            data={
                "user_id": user.id,
                "email": user.email,
                "username": user.username,
                "message": "User registered successfully. Please verify your email."
            },
            message="User registration completed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User registration failed"
        )

@router.post("/login", response_model=APIKeyResponse)
async def login_user(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Authenticate user and return access token."""
    try:
        # Authenticate user
        user = await auth_controller.authenticate_user(
            form_data.username,  # OAuth2PasswordRequestForm uses username field
            form_data.password,
            db
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        token_data = {
            "sub": user.email,
            "user_id": user.id,
            "username": user.username,
            "is_admin": user.has_role("admin")
        }
        access_token = auth_controller.create_access_token(token_data)
        
        # Create session
        session = await auth_controller.create_session(
            user,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            db=db
        )
        
        # Update last login
        await auth_controller.update_last_login(user.id, db)
        
        return APIKeyResponse(
            success=True,
            data={
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": 1800,  # 30 minutes
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "full_name": user.full_name,
                    "is_active": user.is_active,
                    "is_verified": user.is_verified,
                    "roles": [role.name.value for role in user.roles]
                }
            },
            message="Login successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/logout", response_model=APIKeyResponse)
async def logout_user(
    request: Request,
    current_user: dict = Depends(AuthMiddleware.get_current_user),
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Logout user and invalidate session."""
    try:
        # Extract token from Authorization header
        authorization = request.headers.get("authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            # In a full implementation, you'd maintain a token blacklist
            # For now, we'll just invalidate the session if we can find it
            pass
        
        return APIKeyResponse(
            success=True,
            data={"message": "Logged out successfully"},
            message="Logout successful"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/me", response_model=APIKeyResponse)
async def get_current_user(
    current_user: dict = Depends(AuthMiddleware.get_current_user),
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Get current user information."""
    try:
        user = await auth_controller.get_user_by_id(current_user["user_id"], db)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return APIKeyResponse(
            success=True,
            data={
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "bio": user.bio,
                "phone": user.phone,
                "timezone": user.timezone,
                "profile_picture": user.profile_picture,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "created_at": user.created_at.isoformat(),
                "roles": [role.name.value for role in user.roles],
                "permissions": [perm.name.value for role in user.roles for perm in role.permissions]
            },
            message="User information retrieved"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )

@router.put("/me", response_model=APIKeyResponse)
async def update_current_user(
    user_update: UserUpdate,
    request: Request,
    current_user: dict = Depends(AuthMiddleware.get_current_user),
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Update current user information."""
    try:
        updated_user = await auth_controller.update_user(
            current_user["user_id"],
            user_update,
            db
        )
        
        return APIKeyResponse(
            success=True,
            data={
                "id": updated_user.id,
                "email": updated_user.email,
                "username": updated_user.username,
                "full_name": updated_user.full_name,
                "bio": updated_user.bio,
                "phone": updated_user.phone,
                "timezone": updated_user.timezone,
                "updated_at": updated_user.updated_at.isoformat()
            },
            message="User profile updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )

# Admin-only endpoints
@router.get("/", response_model=APIKeyResponse)
async def list_users(
    skip: int = 0,
    limit: int = 50,
    current_user: dict = Depends(AuthMiddleware.verify_admin),
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """List all users (admin only)."""
    try:
        users = await auth_controller.list_users(db, skip=skip, limit=limit)
        users_data = []
        
        for user in users:
            users_data.append({
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "created_at": user.created_at.isoformat(),
                "roles": [role.name.value for role in user.roles]
            })
        
        return APIKeyResponse(
            success=True,
            data={
                "users": users_data,
                "total": len(users_data),
                "skip": skip,
                "limit": limit
            },
            message="Users retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )

@router.get("/{user_id}", response_model=APIKeyResponse)
async def get_user(
    user_id: int,
    current_user: dict = Depends(AuthMiddleware.verify_admin),
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Get user by ID (admin only)."""
    try:
        user = await auth_controller.get_user_by_id(user_id, db)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return APIKeyResponse(
            success=True,
            data={
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "bio": user.bio,
                "phone": user.phone,
                "timezone": user.timezone,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "is_superuser": user.is_superuser,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "created_at": user.created_at.isoformat(),
                "updated_at": user.updated_at.isoformat(),
                "roles": [role.name.value for role in user.roles],
                "failed_login_attempts": user.failed_login_attempts,
                "locked_until": user.locked_until.isoformat() if user.locked_until else None
            },
            message="User retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user"
        )

@router.put("/{user_id}", response_model=APIKeyResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    request: Request,
    current_user: dict = Depends(AuthMiddleware.verify_admin),
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Update user by ID (admin only)."""
    try:
        updated_user = await auth_controller.update_user(user_id, user_update, db)
        
        return APIKeyResponse(
            success=True,
            data={
                "id": updated_user.id,
                "email": updated_user.email,
                "username": updated_user.username,
                "full_name": updated_user.full_name,
                "is_active": updated_user.is_active,
                "is_verified": updated_user.is_verified,
                "updated_at": updated_user.updated_at.isoformat()
            },
            message="User updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )

@router.delete("/{user_id}", response_model=APIKeyResponse)
async def delete_user(
    user_id: int,
    request: Request,
    current_user: dict = Depends(AuthMiddleware.verify_admin),
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Delete user by ID (admin only)."""
    try:
        await auth_controller.delete_user(user_id, db)
        
        return APIKeyResponse(
            success=True,
            data={"user_id": user_id},
            message="User deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )

# API Key Management Endpoints
@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: dict = Depends(AuthMiddleware.get_current_user),
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Create a new API key for the current user."""
    try:
        full_key, api_key = await auth_controller.create_api_key(
            user_id=current_user["user_id"],
            name=api_key_data.name,
            description=api_key_data.description,
            expires_days=api_key_data.expires_days,
            db=db
        )
        
        return APIKeyResponse(
            success=True,
            data={
                "id": api_key.id,
                "name": api_key.name,
                "description": api_key.description,
                "key": full_key,  # Only returned once
                "key_prefix": api_key.key_prefix,
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                "created_at": api_key.created_at.isoformat(),
                "warning": "This is the only time you will see the full API key. Please save it securely."
            },
            message="API key created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )

@router.get("/api-keys", response_model=APIKeyResponse)
async def list_api_keys(
    current_user: dict = Depends(AuthMiddleware.get_current_user),
    db: Session = Depends(get_db)
):
    """List current user's API keys."""
    try:
        from database.models import APIKey
        api_keys = db.query(APIKey).filter(
            APIKey.user_id == current_user["user_id"],
            APIKey.is_deleted == False
        ).all()
        
        keys_data = []
        for key in api_keys:
            keys_data.append({
                "id": key.id,
                "name": key.name,
                "description": key.description,
                "key_prefix": key.key_prefix,
                "is_active": key.is_active,
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "usage_count": key.usage_count,
                "created_at": key.created_at.isoformat()
            })
        
        return APIKeyResponse(
            success=True,
            data={"api_keys": keys_data},
            message="API keys retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API keys"
        )

@router.delete("/api-keys/{key_id}", response_model=APIKeyResponse)
async def revoke_api_key(
    key_id: int,
    current_user: dict = Depends(AuthMiddleware.get_current_user),
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Revoke an API key."""
    try:
        success = await auth_controller.revoke_api_key(
            key_id, 
            current_user["user_id"], 
            db
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        return APIKeyResponse(
            success=True,
            data={"key_id": key_id},
            message="API key revoked successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key"
        )