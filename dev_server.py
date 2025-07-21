#!/usr/bin/env python3
"""
Development server launcher for RAG system.
Bypasses complex imports for quick development and testing.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from pathlib import Path

# Basic imports that work
from config import get_settings
from database.connection import create_tables, SessionLocal, get_db
from database.models import User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dev_app() -> FastAPI:
    """Create a minimal FastAPI app for development."""
    
    settings = get_settings()
    
    app = FastAPI(
        title="RAG System - Development",
        version="1.0.0-dev",
        description="Development server for RAG Document Processing API",
        debug=settings.DEBUG
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In dev, allow all origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        try:
            # Test database connection
            db = SessionLocal()
            user_count = db.query(User).count()
            db.close()
            
            return {
                "status": "healthy",
                "message": "RAG system is running",
                "database": "connected",
                "users": user_count,
                "environment": settings.ENVIRONMENT,
                "debug": settings.DEBUG
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy", 
                    "error": str(e)
                }
            )
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with system info."""
        return {
            "message": "RAG Document Processing API - Development Mode",
            "version": "1.0.0-dev",
            "environment": settings.ENVIRONMENT,
            "docs": "/docs",
            "health": "/health"
        }
    
    # Development info endpoint
    @app.get("/dev/info")
    async def dev_info():
        """Development environment information."""
        return {
            "settings": {
                "app_name": settings.APP_NAME,
                "debug": settings.DEBUG,
                "environment": settings.ENVIRONMENT,
                "database_url": settings.DATABASE_URL[:30] + "..." if len(settings.DATABASE_URL) > 30 else settings.DATABASE_URL,
                "host": settings.HOST,
                "port": settings.PORT
            },
            "directories": {
                "upload": settings.UPLOAD_DIRECTORY,
                "temp": settings.TEMP_DIRECTORY,
                "exists_upload": Path(settings.UPLOAD_DIRECTORY).exists(),
                "exists_temp": Path(settings.TEMP_DIRECTORY).exists()
            }
        }
    
    # Create upload directories if they don't exist
    Path(settings.UPLOAD_DIRECTORY).mkdir(exist_ok=True)
    Path(settings.TEMP_DIRECTORY).mkdir(exist_ok=True)
    
    return app

def main():
    """Main development server launcher."""
    try:
        # Load settings
        settings = get_settings()
        logger.info(f"Starting RAG Development Server...")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Debug mode: {settings.DEBUG}")
        
        # Initialize database
        logger.info("Initializing database...")
        create_tables()
        logger.info("Database initialized successfully")
        
        # Create app
        app = create_dev_app()
        
        # Start server
        logger.info(f"Server starting on http://{settings.HOST}:{settings.PORT}")
        logger.info("API Documentation available at: http://localhost:8000/docs")
        logger.info("Health check available at: http://localhost:8000/health")
        logger.info("Development info at: http://localhost:8000/dev/info")
        
        uvicorn.run(
            app,
            host=settings.HOST,
            port=settings.PORT,
            log_level="info",
            reload=True,  # Auto-reload on file changes
            reload_dirs=["./api", "./database", "./utils"]  # Watch these directories
        )
        
    except Exception as e:
        logger.error(f"Failed to start development server: {e}")
        raise

if __name__ == "__main__":
    main()