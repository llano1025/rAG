"""
Authentication routes for password reset and email verification.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta, timezone
import secrets
from pydantic import BaseModel, EmailStr

from database.connection import get_db
from database.models import User
from api.controllers.auth_controller import AuthController
from api.schemas.api_schemas import APIResponse
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

# Dependency injection
async def get_auth_controller(db: Session = Depends(get_db)) -> AuthController:
    """Dependency to get auth controller instance."""
    from main import get_encryption_manager, get_audit_logger
    encryption = get_encryption_manager()
    audit_logger = get_audit_logger()
    return AuthController(encryption, audit_logger)

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

@router.post("/request-password-reset", response_model=APIResponse)
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
        return APIResponse(
            success=True,
            data={"message": "If the email exists, a password reset link has been sent."},
            message="Password reset request processed"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset request failed"
        )

@router.post("/reset-password", response_model=APIResponse)
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
        
        return APIResponse(
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

@router.post("/request-email-verification", response_model=APIResponse)
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
        
        return APIResponse(
            success=True,
            data={"message": "Email verification endpoint - requires authenticated user"},
            message="Not implemented without authentication context"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email verification request failed"
        )

@router.post("/verify-email", response_model=APIResponse)
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
        
        return APIResponse(
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