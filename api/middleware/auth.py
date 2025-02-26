from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)
security = HTTPBearer()

class AuthMiddleware:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = 30

    async def create_access_token(self, data: dict) -> str:
        """Create a new JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating access token: {str(e)}")
            raise HTTPException(status_code=500, detail="Could not create access token")

    async def verify_token(self, token: str) -> Dict:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.error(f"JWT verification failed: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Security(security)
    ) -> Dict:
        """Dependency to get the current authenticated user from a request."""
        try:
            token = credentials.credentials
            payload = await self.verify_token(token)
            
            if payload.get("exp") < datetime.utcnow().timestamp():
                raise HTTPException(
                    status_code=401,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            return payload
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def verify_admin(self, current_user: Dict = Depends(get_current_user)) -> Dict:
        """Verify that the current user has admin privileges."""
        if not current_user.get("is_admin"):
            raise HTTPException(
                status_code=403,
                detail="Admin privileges required"
            )
        return current_user

# Usage example:
"""
from fastapi import FastAPI
from config import Settings

app = FastAPI()
settings = Settings()
auth_middleware = AuthMiddleware(secret_key=settings.SECRET_KEY)

@app.get("/protected")
async def protected_route(current_user: Dict = Depends(auth_middleware.get_current_user)):
    return {"message": "Access granted", "user": current_user}

@app.get("/admin")
async def admin_route(current_user: Dict = Depends(auth_middleware.verify_admin)):
    return {"message": "Admin access granted", "user": current_user}
"""