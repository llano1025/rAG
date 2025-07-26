from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import os
from typing import Dict, Any
from pathlib import Path

from api.routes import document_routes, search_routes, vector_routes, library_routes, user_routes, auth_routes, health_routes, admin_routes, advanced_routes, analytics_routes, chat_routes
from api.middleware.auth import AuthMiddleware
from api.middleware.error_handler import ErrorHandler
from api.middleware.rate_limiter import RateLimiter
from utils.monitoring.health_check import HealthChecker
from utils.caching.redis_manager import RedisManager
from utils.security.encryption import EncryptionManager, EncryptionConfig
from utils.security.audit_logger import AuditLogger, AuditLoggerConfig
from database.connection import create_tables
from config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
health_check = None
redis_manager = None
encryption_manager = None
audit_logger = None
settings = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global health_check, redis_manager, encryption_manager, audit_logger, settings
    
    try:
        # Load settings
        settings = get_settings()
        logger.info("Settings loaded successfully")
        
        # Initialize database
        logger.info("Initializing database...")
        create_tables()
        logger.info("Database tables created successfully")
        
        # Initialize Redis connection
        redis_manager = RedisManager(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB
        )
        await redis_manager.connect()
        logger.info("Redis connection established")
        
        # Initialize encryption manager
        from pathlib import Path
        encryption_config = EncryptionConfig(
            key_path=Path("data/encryption.key"),
            salt_path=Path("data/encryption.salt")
        )
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        encryption_manager = EncryptionManager(encryption_config)
        logger.info("Encryption manager initialized")
        
        # Initialize audit logger
        audit_config = AuditLoggerConfig(
            log_path=Path("data/audit_logs"),
            rotation_size_mb=10,
            retention_days=90
        )
        # Ensure audit log directory exists
        Path("data/audit_logs").mkdir(parents=True, exist_ok=True)
        audit_logger = AuditLogger(audit_config)
        logger.info("Audit logger initialized")
        
        # Initialize health check
        health_check = HealthChecker()
        logger.info("Health check system initialized")
        
        # Store instances in app state
        app.state.redis = redis_manager
        app.state.encryption = encryption_manager
        app.state.audit_logger = audit_logger
        app.state.health = health_check
        app.state.settings = settings
        
        logger.info("RAG Application startup completed successfully")
        
        # Print startup success message to console
        host_display = "127.0.0.1" if settings.HOST == "0.0.0.0" else settings.HOST
        print(f"\n🚀 RAG is running at http://{host_display}:{settings.PORT}")
        print(f"📚 API docs available at http://{host_display}:{settings.PORT}/docs")
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise
    finally:
        # Cleanup on shutdown
        try:
            if redis_manager:
                await redis_manager.disconnect()
                logger.info("Redis connection closed")
            
            if health_check:
                logger.info("Health check cleanup completed")
                
            logger.info("RAG Application shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="RAG Document Processing API",
        description="A Retrieval-Augmented Generation system for document processing, vector search, and AI-powered question answering",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Configure CORS
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Add trusted host middleware
    if os.getenv("TRUSTED_HOSTS"):
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=os.getenv("TRUSTED_HOSTS").split(",")
        )
    
    # Add custom middleware
    app.add_middleware(ErrorHandler)
    # app.add_middleware(RateLimiter)  # Disabled for testing - requires proper configuration
    
    # Include routers with /api prefix
    app.include_router(auth_routes.router, prefix="/api")
    app.include_router(user_routes.router, prefix="/api")
    app.include_router(document_routes.router, prefix="/api")
    app.include_router(search_routes.router, prefix="/api")
    app.include_router(vector_routes.router, prefix="/api")
    app.include_router(library_routes.router, prefix="/api")
    app.include_router(chat_routes.router, prefix="/api")
    app.include_router(health_routes.router, prefix="/api")
    app.include_router(admin_routes.router, prefix="/api")
    app.include_router(advanced_routes.router, prefix="/api")
    app.include_router(analytics_routes.router, prefix="/api")
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "RAG Document Processing API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health")
    async def health_endpoint():
        """Health check endpoint."""
        try:
            if health_check:
                health_status = await health_check.run_all_checks()
                return health_status
            else:
                return {"status": "starting", "message": "Health check not yet initialized"}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise HTTPException(status_code=503, detail="Service unavailable")
    
    @app.get("/metrics")
    async def metrics_endpoint():
        """Metrics endpoint for monitoring."""
        try:
            if health_check:
                # Return basic system metrics from health check
                health_status = await health_check.run_all_checks()
                return {
                    "system_health": health_status,
                    "timestamp": health_status.get("timestamp"),
                    "overall_status": health_status.get("status").value if health_status.get("status") else "unknown"
                }
            else:
                return {"message": "Metrics not available"}
        except Exception as e:
            logger.error(f"Metrics collection failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Metrics unavailable")
    
    return app

# Dependency injection functions
def get_encryption_manager() -> EncryptionManager:
    """Get the global encryption manager instance."""
    global encryption_manager
    if encryption_manager is None:
        # Create encryption config similar to startup process
        from pathlib import Path
        encryption_config = EncryptionConfig(
            key_path=Path("data/encryption.key"),
            salt_path=Path("data/encryption.salt")
        )
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        encryption_manager = EncryptionManager(encryption_config)
    return encryption_manager

def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global audit_logger
    if audit_logger is None:
        # Create audit logger with default configuration
        audit_config = AuditLoggerConfig(
            log_path=Path("data/audit_logs"),
            rotation_size_mb=10,
            retention_days=90
        )
        # Ensure audit log directory exists
        Path("data/audit_logs").mkdir(parents=True, exist_ok=True)
        audit_logger = AuditLogger(audit_config)
    return audit_logger

def get_redis_manager() -> RedisManager:
    """Get the global Redis manager instance."""
    global redis_manager
    return redis_manager

def get_health_check() -> HealthChecker:
    """Get the global health check instance."""
    global health_check
    return health_check

# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    # Load settings for development server
    try:
        dev_settings = get_settings()
        uvicorn.run(
            "main:app",
            host=dev_settings.HOST,
            port=dev_settings.PORT,
            reload=dev_settings.DEBUG,
            log_level="info" if dev_settings.DEBUG else "warning"
        )
    except Exception as e:
        logger.error(f"Failed to start development server: {str(e)}")
        raise