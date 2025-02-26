from fastapi import HTTPException, Request
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Optional
import logging
from redis import Redis
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    requests_per_minute: int
    burst_size: Optional[int] = None
    cooldown_minutes: int = 15

class RateLimiter:
    def __init__(
        self,
        redis_client: Redis,
        config: RateLimitConfig,
        prefix: str = "rate_limit"
    ):
        self.redis = redis_client
        self.config = config
        self.prefix = prefix
        self.burst_size = config.burst_size or config.requests_per_minute * 2

    def _get_key(self, identifier: str, endpoint: str) -> str:
        """Generate Redis key for rate limiting."""
        return f"{self.prefix}:{endpoint}:{identifier}"

    async def is_rate_limited(self, identifier: str, endpoint: str) -> bool:
        """Check if the request should be rate limited."""
        key = self._get_key(identifier, endpoint)
        pipe = self.redis.pipeline()

        try:
            # Get current count and last reset time
            current_count = int(self.redis.get(key) or 0)
            last_reset = float(self.redis.get(f"{key}:last_reset") or 0)
            current_time = datetime.utcnow().timestamp()

            # Check if we need to reset the counter
            if current_time - last_reset > 60:  # More than a minute has passed
                pipe.set(key, 1)
                pipe.set(f"{key}:last_reset", current_time)
                pipe.expire(key, 60)
                pipe.expire(f"{key}:last_reset", 60)
                pipe.execute()
                return False

            # Check if request exceeds rate limit
            if current_count >= self.config.requests_per_minute:
                # Check for burst allowance
                if current_count >= self.burst_size:
                    # Apply cooldown period
                    pipe.setex(
                        f"{key}:cooldown",
                        self.config.cooldown_minutes * 60,
                        "1"
                    )
                    pipe.execute()
                    return True
                
            # Increment counter
            pipe.incr(key)
            pipe.execute()
            return False

        except Exception as e:
            logger.error(f"Rate limiting error: {str(e)}")
            return False  # Fail open in case of Redis errors

    async def get_remaining_requests(self, identifier: str, endpoint: str) -> Dict:
        """Get remaining requests information."""
        key = self._get_key(identifier, endpoint)
        try:
            current_count = int(self.redis.get(key) or 0)
            cooldown = self.redis.get(f"{key}:cooldown")
            
            return {
                "remaining": max(0, self.config.requests_per_minute - current_count),
                "total": self.config.requests_per_minute,
                "in_cooldown": bool(cooldown)
            }
        except Exception as e:
            logger.error(f"Error getting remaining requests: {str(e)}")
            return {
                "remaining": 0,
                "total": self.config.requests_per_minute,
                "in_cooldown": False
            }

async def rate_limit_middleware(
    request: Request,
    rate_limiter: RateLimiter,
    get_identifier=lambda r: r.client.host
):
    """FastAPI middleware for rate limiting."""
    identifier = get_identifier(request)
    endpoint = request.url.path

    if await rate_limiter.is_rate_limited(identifier, endpoint):
        remaining = await rate_limiter.get_remaining_requests(identifier, endpoint)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Too many requests",
                "remaining": remaining
            }
        )

    # Add rate limit headers
    remaining = await rate_limiter.get_remaining_requests(identifier, endpoint)
    request.state.rate_limit_remaining = remaining

# Usage example:
"""
from fastapi import FastAPI
from redis import Redis
from config import Settings

app = FastAPI()
settings = Settings()
redis_client = Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD
)

rate_limit_config = RateLimitConfig(
    requests_per_minute=60,
    burst_size=120,
    cooldown_minutes=15
)

rate_limiter = RateLimiter(redis_client, rate_limit_config)

@app.middleware("http")
async def rate_limiting(request: Request, call_next):
    await rate_limit_middleware(request, rate_limiter)
    response = await call_next(request)
    
    # Add rate limit headers
    remaining = request.state.rate_limit_remaining
    response.headers["X-RateLimit-Remaining"] = str(remaining["remaining"])
    response.headers["X-RateLimit-Limit"] = str(remaining["total"])
    
    return response
"""