"""
Authentication middleware for JWT token verification and user authentication.
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging

from config import get_settings
from database.connection import get_db
from database.models import User

logger = logging.getLogger(__name__)
security = HTTPBearer()
settings = get_settings()

class AuthMiddleware:
    """Authentication middleware for handling JWT tokens and user verification."""
    
    @staticmethod
    async def verify_token(token: str) -> Dict:
        """Verify and decode a JWT token."""
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
        except JWTError as e:
            logger.error(f"JWT verification failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Security(security),
        db: Session = Depends(get_db)
    ) -> Dict:
        """Dependency to get the current authenticated user from a request."""
        try:
            token = credentials.credentials
            payload = await AuthMiddleware.verify_token(token)
            
            # Check token expiration
            if payload.get("exp") < datetime.utcnow().timestamp():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Get user from database to ensure they still exist and are active
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
            
            # Return user info for use in endpoints
            return {
                "user_id": user.id,
                "email": user.email,
                "username": user.username,
                "is_admin": user.has_role("admin"),
                "is_superuser": user.is_superuser,
                "permissions": [perm.name.value for role in user.roles for perm in role.permissions]
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    async def verify_admin(
        current_user: Dict = Depends(get_current_user)
    ) -> Dict:
        """Verify that the current user has admin privileges."""
        if not current_user.get("is_admin") and not current_user.get("is_superuser"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        return current_user

    @staticmethod
    async def verify_permission(
        required_permission: str,
        current_user: Dict = Depends(get_current_user)
    ) -> Dict:
        """Verify that the current user has a specific permission."""
        if current_user.get("is_superuser"):
            return current_user
            
        user_permissions = current_user.get("permissions", [])
        if required_permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{required_permission}' required"
            )
        return current_user

    @staticmethod
    async def get_current_active_user(
        current_user: Dict = Depends(get_current_user),
        db: Session = Depends(get_db)
    ) -> User:
        """Get the current user as a database model instance."""
        user = db.query(User).filter(
            User.id == current_user["user_id"],
            User.is_active == True,
            User.is_deleted == False
        ).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        return user

    @staticmethod
    async def validate_api_key(
        api_key: str,
        db: Session
    ) -> Optional[User]:
        """Validate an API key and return the associated user."""
        from database.models import APIKey
        
        key_hash = str(hash(api_key))
        
        api_key_obj = db.query(APIKey).filter(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True,
            APIKey.is_deleted == False
        ).first()
        
        if api_key_obj and api_key_obj.is_valid():
            # Update usage tracking
            api_key_obj.last_used = datetime.utcnow()
            api_key_obj.usage_count += 1
            db.commit()
            
            return api_key_obj.user
        
        return None

    @staticmethod
    async def api_key_auth(
        credentials: HTTPAuthorizationCredentials = Security(security),
        db: Session = Depends(get_db)
    ) -> Dict:
        """Alternative authentication using API key."""
        try:
            api_key = credentials.credentials
            
            # Check if it's an API key (starts with 'rag_')
            if api_key.startswith('rag_'):
                user = await AuthMiddleware.validate_api_key(api_key, db)
                if user:
                    return {
                        "user_id": user.id,
                        "email": user.email,
                        "username": user.username,
                        "is_admin": user.has_role("admin"),
                        "is_superuser": user.is_superuser,
                        "auth_method": "api_key",
                        "permissions": [perm.name.value for role in user.roles for perm in role.permissions]
                    }
            
            # Fall back to JWT authentication
            return await AuthMiddleware.get_current_user(credentials, db)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API key authentication error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key or token",
                headers={"WWW-Authenticate": "Bearer"},
            )

# Convenience functions for dependency injection
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db)
) -> Dict:
    """Convenience function for getting current user."""
    return await AuthMiddleware.get_current_user(credentials, db)

async def verify_admin(current_user: Dict = Depends(get_current_user)) -> Dict:
    """Convenience function for admin verification."""
    return await AuthMiddleware.verify_admin(current_user)

async def get_current_active_user(
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> User:
    """Convenience function for getting current user as model."""
    return await AuthMiddleware.get_current_active_user(current_user, db)

async def api_key_auth(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db)
) -> Dict:
    """Convenience function for API key authentication."""
    return await AuthMiddleware.api_key_auth(credentials, db)