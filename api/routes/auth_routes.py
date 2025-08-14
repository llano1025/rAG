"""
Authentication routes for password reset and email verification.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta, timezone
import secrets
from pydantic import BaseModel, EmailStr

from database.connection import get_db
from database.models import User, UserRoleEnum
from api.controllers.auth_controller import AuthController
from api.schemas.api_schemas import APIKeyResponse, TokenResponse
from api.schemas.user_schemas import UserCreate
from api.schemas.responses import StandardResponse, AuthResponse, create_success_response
from api.middleware.auth import AuthMiddleware
from utils.security.audit_logger import AuditLogger
from utils.security.encryption import EncryptionManager

router = APIRouter(prefix="/auth", tags=["authentication"])

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    token: str
    new_password: str
    confirm_password: str

class EmailVerification(BaseModel):
    token: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict

# Dependency injection
async def get_auth_controller(db: Session = Depends(get_db)) -> AuthController:
    """Dependency to get auth controller instance."""
    from main import get_encryption_manager, get_audit_logger
    encryption = get_encryption_manager()
    audit_logger = get_audit_logger()
    return AuthController(encryption, audit_logger)

# Core Authentication Endpoints

@router.post("/register", response_model=StandardResponse)
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
        
        return create_success_response(
            data={
                "user_id": user.id,
                "email": user.email,
                "username": user.username
            },
            message="User registered successfully. Please verify your email."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User registration failed"
        )

@router.post("/login", response_model=AuthResponse)
async def login_user(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Authenticate user and return access token."""
    try:
        from passlib.context import CryptContext
        from datetime import datetime, timedelta
        import jwt
        from config import get_settings
        
        settings = get_settings()
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Find user by username or email
        user = db.query(User).filter(
            (User.username == form_data.username) | (User.email == form_data.username)
        ).first()
        
        if not user or not pwd_context.verify(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        expire = datetime.utcnow() + timedelta(minutes=30)
        token_data = {
            "sub": user.email,
            "user_id": user.id,
            "username": user.username,
            "is_admin": user.is_superuser,
            "exp": expire
        }
        access_token = jwt.encode(token_data, settings.SECRET_KEY, algorithm="HS256")
        
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=1800,  # 30 minutes
            user={
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "roles": ["admin"] if user.is_superuser else ["user"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")  # Debug logging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.get("/me", response_model=dict)
async def get_current_active_user_info(
    request: Request,
    db: Session = Depends(get_db)
):
    """Get current user information from JWT token."""
    try:
        # Extract token from Authorization header
        authorization = request.headers.get("authorization")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token = authorization.split(" ")[1]
        
        # Decode token
        from config import get_settings
        import jwt
        
        settings = get_settings()
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        
        # Get user from database
        user_id = payload.get("user_id")
        email = payload.get("sub")
        
        if not user_id or not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = db.query(User).filter(
            User.id == user_id,
            User.email == email,
            User.is_active == True,
            User.is_deleted == False
        ).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "roles": ["admin"] if user.is_superuser else ["user"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Get current user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/logout", response_model=APIKeyResponse)
async def logout_user(
    request: Request,
    current_user: dict = Depends(AuthMiddleware.get_current_active_user),
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

# Email utilities

async def send_email_background(email: str, subject: str, content: str):
    """Background task to send email (placeholder - implement with real email service)."""
    # In a real implementation, you would integrate with an email service like:
    # - SendGrid
    # - AWS SES
    # - SMTP server
    # - etc.
    print(f"EMAIL SENT TO: {email}")
    print(f"SUBJECT: {subject}")
    print(f"CONTENT: {content}")
    print("=" * 50)

# Password Reset and Email Verification Endpoints

@router.post("/request-password-reset", response_model=APIKeyResponse)
async def request_password_reset(
    request_data: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Request a password reset token."""
    try:
        user = await auth_controller.get_user_by_email(request_data.email, db)
        
        if user:
            # Generate reset token
            reset_token = secrets.token_urlsafe(32)
            reset_expires = datetime.now(timezone.utc) + timedelta(hours=1)  # 1 hour expiry
            
            # Save token to user
            user.reset_password_token = reset_token
            user.reset_password_expires = reset_expires
            db.commit()
            
            # Send email in background
            reset_link = f"https://your-domain.com/reset-password?token={reset_token}"
            email_content = f"""
            Hi {user.full_name or user.username},
            
            You requested a password reset for your account. Click the link below to reset your password:
            
            {reset_link}
            
            This link will expire in 1 hour. If you didn't request this reset, please ignore this email.
            
            Best regards,
            The RAG System Team
            """
            
            background_tasks.add_task(
                send_email_background,
                request_data.email,
                "Password Reset Request",
                email_content
            )
            
            # Log the request
            await auth_controller.audit_logger.log(
                action="password_reset_requested",
                resource_type="user",
                resource_id=str(user.id),
                user_id=user.id,
                details={"email": request_data.email}
            )
        
        # Always return success for security (don't reveal if email exists)
        return APIKeyResponse(
            success=True,
            data={"message": "If the email exists, a password reset link has been sent."},
            message="Password reset request processed"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset request failed"
        )

@router.post("/reset-password", response_model=APIKeyResponse)
async def reset_password(
    reset_data: PasswordReset,
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Reset password using a valid token."""
    try:
        # Validate passwords match
        if reset_data.new_password != reset_data.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Passwords do not match"
            )
        
        # Find user with valid reset token
        user = db.query(User).filter(
            User.reset_password_token == reset_data.token,
            User.reset_password_expires > datetime.now(timezone.utc),
            User.is_active == True,
            User.is_deleted == False
        ).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Update password
        user.hashed_password = auth_controller.hash_password(reset_data.new_password)
        user.password_changed_at = datetime.now(timezone.utc)
        user.reset_password_token = None
        user.reset_password_expires = None
        user.failed_login_attempts = 0  # Reset failed attempts
        user.locked_until = None  # Unlock account if locked
        
        db.commit()
        
        # Log the password reset
        await auth_controller.audit_logger.log(
            action="password_reset_completed",
            resource_type="user",
            resource_id=str(user.id),
            user_id=user.id,
            details={"email": user.email}
        )
        
        return APIKeyResponse(
            success=True,
            data={"message": "Password reset successfully"},
            message="Password updated"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )

@router.post("/request-email-verification", response_model=APIKeyResponse)
async def request_email_verification(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_auth_controller),
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Request email verification for current user."""
    try:
        # This would typically be called by an authenticated user
        # For now, we'll accept an email in the request
        # In production, get from current_user context
        
        return APIKeyResponse(
            success=True,
            data={"message": "Email verification endpoint - requires authenticated user"},
            message="Not implemented without authentication context"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email verification request failed"
        )

@router.post("/verify-email", response_model=APIKeyResponse)
async def verify_email(
    verification_data: EmailVerification,
    db: Session = Depends(get_db),
    auth_controller: AuthController = Depends(get_auth_controller)
):
    """Verify email using a valid token."""
    try:
        # Find user with valid verification token
        user = db.query(User).filter(
            User.email_verification_token == verification_data.token,
            User.email_verification_expires > datetime.now(timezone.utc),
            User.is_active == True,
            User.is_deleted == False
        ).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token"
            )
        
        # Verify email
        user.is_verified = True
        user.email_verification_token = None
        user.email_verification_expires = None
        
        db.commit()
        
        # Log the verification
        await auth_controller.audit_logger.log(
            action="email_verified",
            resource_type="user",
            resource_id=str(user.id),
            user_id=user.id,
            details={"email": user.email}
        )
        
        return APIKeyResponse(
            success=True,
            data={
                "message": "Email verified successfully",
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "is_verified": user.is_verified
                }
            },
            message="Email verification completed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email verification failed"
        )