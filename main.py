from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
import os
from pathlib import Path

from api.routes import document_routes, search_routes, vector_routes, library_routes, user_routes, auth_routes, health_routes, admin_routes, advanced_routes, analytics_routes, chat_routes, ocr_routes, model_routes
from api.middleware.error_handler import ErrorHandler
from utils.monitoring.health_check import HealthChecker
from utils.caching.redis_manager import RedisManager
from utils.security.encryption import EncryptionManager, EncryptionConfig
from utils.security.audit_logger import AuditLogger, AuditLoggerConfig
from utils.websocket import init_websocket_manager, get_websocket_manager
from database.connection import create_tables
from config import get_settings

# Configure logging with file handler
def configure_logging():
    """Configure logging to output to both console and file."""
    log_dir = Path("runtime/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for server.log
    file_handler = logging.FileHandler(log_dir / "server.log", mode='a')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

# Initialize logging
configure_logging()
logger = logging.getLogger(__name__)

# Global instances
health_check = None
redis_manager = None
encryption_manager = None
audit_logger = None
settings = None

async def _initialize_model_systems(settings):
    """Initialize model storage and preloading systems."""
    try:
        logger.info("Initializing model storage and preloader systems...")

        # Import here to avoid circular imports
        from vector_db.embedding_storage_manager import ModelStorageManager
        from vector_db.embedding_model_preloader import ModelPreloader, PreloadPolicy

        # Initialize model storage manager with settings
        storage_manager = ModelStorageManager(
            storage_base_path=settings.MODEL_STORAGE_BASE_PATH,
            enable_auto_download=settings.MODEL_STORAGE_ENABLE_AUTO_DOWNLOAD,
            max_concurrent_downloads=settings.MODEL_STORAGE_MAX_CONCURRENT_DOWNLOADS,
            storage_limit_gb=settings.MODEL_STORAGE_LIMIT_GB
        )

        # Parse preload policy
        policy_mapping = {
            "none": PreloadPolicy.NONE,
            "essential": PreloadPolicy.ESSENTIAL,
            "all": PreloadPolicy.ALL,
            "selective": PreloadPolicy.SELECTIVE
        }
        preload_policy = policy_mapping.get(
            settings.MODEL_PRELOAD_POLICY.lower(),
            PreloadPolicy.ALL  # Default to ALL
        )

        # Initialize model preloader
        preloader = ModelPreloader(
            storage_manager=storage_manager,
            preload_policy=preload_policy,
            max_concurrent_downloads=settings.MODEL_STORAGE_MAX_CONCURRENT_DOWNLOADS,
            download_timeout_minutes=settings.MODEL_PRELOAD_TIMEOUT_MINUTES
        )

        # Get startup summary
        startup_summary = await preloader.get_startup_summary()

        # Perform model preloading if enabled
        if settings.MODEL_PRELOAD_ON_STARTUP:
            if settings.MODEL_PRELOAD_BACKGROUND:
                # Start preloading in background
                logger.info("Starting background model preloading...")
                asyncio.create_task(_background_model_preload(preloader))
            else:
                # Preload synchronously (blocks startup)
                logger.info("Starting synchronous model preloading...")
                result = await preloader.preload_models()
                logger.info(f"Model preloading completed: {result.successfully_downloaded} downloaded, "
                           f"{result.already_available} already available, {result.failed_downloads} failed")

        logger.info("Model systems initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize model systems: {e}")
        # Don't fail startup if model systems fail
        logger.warning("Continuing startup without model preloading")

async def _background_model_preload(preloader):
    """Background task for model preloading."""
    try:
        logger.info("Background model preloading started")
        result = await preloader.preload_models()
        logger.info(f"Background model preloading completed: {result.successfully_downloaded} downloaded, "
                   f"{result.already_available} already available, {result.failed_downloads} failed")

        if result.errors:
            logger.warning(f"Model preloading errors: {result.errors}")

    except Exception as e:
        logger.error(f"Background model preloading failed: {e}")

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
            key_path=Path("runtime/data/encryption.key"),
            salt_path=Path("runtime/data/encryption.salt")
        )
        # Ensure data directory exists
        Path("runtime/data").mkdir(exist_ok=True)
        encryption_manager = EncryptionManager(encryption_config)
        logger.info("Encryption manager initialized")
        
        # Initialize audit logger
        audit_config = AuditLoggerConfig(
            log_path=Path("runtime/audit_logs"),
            rotation_size_mb=10,
            retention_days=90
        )
        # Ensure audit log directory exists
        Path("runtime/audit_logs").mkdir(parents=True, exist_ok=True)
        audit_logger = AuditLogger(audit_config)
        logger.info("Audit logger initialized")
        
        # Initialize health check
        health_check = HealthChecker()
        logger.info("Health check system initialized")

        # Initialize model storage and preloader
        await _initialize_model_systems(settings)

        # Initialize WebSocket manager
        websocket_manager = init_websocket_manager(app)
        logger.info("WebSocket manager initialized")

        # Store instances in app state
        app.state.redis = redis_manager
        app.state.encryption = encryption_manager
        app.state.audit_logger = audit_logger
        app.state.health = health_check
        app.state.settings = settings
        app.state.websocket = websocket_manager

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
    
    # Add request logging middleware for debugging
    @app.middleware("http")
    async def log_requests(request, call_next):
        import time
        import logging
        logger = logging.getLogger("middleware")
        
        start_time = time.time()
        logger.info(f"🔍 REQUEST: {request.method} {request.url.path}")
        logger.info(f"🔍 REQUEST HEADERS: {dict(request.headers)}")
        logger.info(f"🔍 REQUEST QUERY PARAMS: {dict(request.query_params)}")
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(f"🔍 RESPONSE: {response.status_code} in {process_time:.3f}s")
        
        return response
    
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
    app.include_router(ocr_routes.router, prefix="/api")
    app.include_router(health_routes.router, prefix="/api")
    app.include_router(admin_routes.router, prefix="/api")
    app.include_router(advanced_routes.router, prefix="/api")
    app.include_router(analytics_routes.router, prefix="/api")
    app.include_router(model_routes.router, prefix="/api")
    
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
            key_path=Path("runtime/data/encryption.key"),
            salt_path=Path("runtime/data/encryption.salt")
        )
        # Ensure data directory exists
        Path("runtime/data").mkdir(exist_ok=True)
        encryption_manager = EncryptionManager(encryption_config)
    return encryption_manager

def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global audit_logger
    if audit_logger is None:
        # Create audit logger with default configuration
        audit_config = AuditLoggerConfig(
            log_path=Path("runtime/audit_logs"),
            rotation_size_mb=10,
            retention_days=90
        )
        # Ensure audit log directory exists
        Path("runtime/audit_logs").mkdir(parents=True, exist_ok=True)
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

def get_websocket_manager_dep():
    """Get the global WebSocket manager instance for dependency injection."""
    return get_websocket_manager()

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