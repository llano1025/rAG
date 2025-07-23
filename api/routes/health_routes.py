"""
Health check API routes for monitoring system and vector database health.
Provides endpoints for comprehensive system health monitoring.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import logging

from database.connection import get_db
from vector_db.health_checker import get_vector_health_checker, VectorHealthChecker
from utils.monitoring.health_check import HealthChecker, HealthStatus
from api.middleware.auth import get_current_active_user
from database.models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])

# Initialize health checkers
system_health_checker = HealthChecker()
vector_health_checker = get_vector_health_checker()

@router.get("/")
async def get_basic_health():
    """
    Basic health check endpoint for load balancers and monitoring.
    Returns simple OK status without detailed information.
    """
    return {"status": "ok", "service": "rAG API"}

@router.get("/system")
async def get_system_health():
    """
    Get comprehensive system health including resources and basic services.
    Public endpoint for monitoring system health.
    """
    try:
        health_results = await system_health_checker.run_all_checks()
        
        # Convert HealthStatus enum to string for JSON serialization
        if isinstance(health_results.get("status"), HealthStatus):
            health_results["status"] = health_results["status"].value
        
        for component, result in health_results.get("components", {}).items():
            if isinstance(result.get("status"), HealthStatus):
                result["status"] = result["status"].value
        
        # Set HTTP status code based on health
        if health_results["status"] == "unhealthy":
            return JSONResponse(
                content=health_results,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        elif health_results["status"] == "degraded":
            return JSONResponse(
                content=health_results,
                status_code=status.HTTP_200_OK
            )
        else:
            return health_results
            
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": "Health check failed",
                "message": str(e)
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@router.get("/vector")
async def get_vector_health(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get comprehensive vector database health status.
    Requires authentication as this provides detailed system information.
    """
    try:
        # Check if user has permission to view system health
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: admin privileges required"
            )
        
        health_results = await vector_health_checker.check_all_components(db)
        
        # Convert HealthStatus enum to string for JSON serialization
        if isinstance(health_results.get("status"), HealthStatus):
            health_results["status"] = health_results["status"].value
        
        for component, result in health_results.get("components", {}).items():
            if isinstance(result.get("status"), HealthStatus):
                result["status"] = result["status"].value
        
        # Set HTTP status code based on health
        if health_results["status"] == "unhealthy":
            return JSONResponse(
                content=health_results,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        elif health_results["status"] == "degraded":
            return JSONResponse(
                content=health_results,
                status_code=status.HTTP_200_OK
            )
        else:
            return health_results
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": "Vector health check failed",
                "message": str(e)
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@router.get("/vector/quick")
async def get_vector_quick_health(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get quick vector database health status without detailed checks.
    Faster endpoint for frequent monitoring.
    """
    try:
        # Check if user has permission to view system health
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: admin privileges required"
            )
        
        health_results = await vector_health_checker.get_quick_status()
        
        # Convert HealthStatus enum to string for JSON serialization
        if isinstance(health_results.get("status"), HealthStatus):
            health_results["status"] = health_results["status"].value
        
        # Set HTTP status code based on health
        if health_results["status"] == "unhealthy":
            return JSONResponse(
                content=health_results,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        else:
            return health_results
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick vector health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": "Quick vector health check failed",
                "message": str(e)
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@router.get("/complete")
async def get_complete_health(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get complete system health including both system and vector database checks.
    Most comprehensive health check endpoint.
    """
    try:
        # Check if user has permission to view system health
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: admin privileges required"
            )
        
        # Run both system and vector health checks
        system_results = await system_health_checker.run_all_checks()
        vector_results = await vector_health_checker.check_all_components(db)
        
        # Combine results
        combined_results = {
            "status": "healthy",  # Will be updated based on component status
            "timestamp": vector_results.get("timestamp"),
            "system_health": system_results,
            "vector_health": vector_results
        }
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        
        system_status = system_results.get("status")
        if isinstance(system_status, HealthStatus):
            if system_status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif system_status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED
        elif system_status == "unhealthy":
            overall_status = HealthStatus.UNHEALTHY
        elif system_status == "degraded" and overall_status != HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.DEGRADED
        
        vector_status = vector_results.get("status")
        if isinstance(vector_status, HealthStatus):
            if vector_status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif vector_status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.DEGRADED
        elif vector_status == "unhealthy":
            overall_status = HealthStatus.UNHEALTHY
        elif vector_status == "degraded" and overall_status != HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.DEGRADED
        
        combined_results["status"] = overall_status.value
        
        # Convert HealthStatus enums to strings for JSON serialization
        def convert_status_enums(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, HealthStatus):
                        obj[key] = value.value
                    elif isinstance(value, dict):
                        convert_status_enums(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                convert_status_enums(item)
        
        convert_status_enums(combined_results)
        
        # Set HTTP status code based on health
        if combined_results["status"] == "unhealthy":
            return JSONResponse(
                content=combined_results,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        elif combined_results["status"] == "degraded":
            return JSONResponse(
                content=combined_results,
                status_code=status.HTTP_200_OK
            )
        else:
            return combined_results
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Complete health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": "Complete health check failed",
                "message": str(e)
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@router.get("/metrics")
async def get_health_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get health metrics and statistics for monitoring dashboards.
    Returns key performance indicators and health trends.
    """
    try:
        # Check if user has permission to view system metrics
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: admin privileges required"
            )
        
        # Get performance stats from vector storage manager
        from vector_db.storage_manager import get_storage_manager
        storage_manager = get_storage_manager()
        
        performance_stats = await storage_manager.get_performance_stats()
        faiss_stats = await storage_manager.get_faiss_stats()
        
        # Get database statistics
        from database.models import Document, DocumentChunk, VectorIndex, User as UserModel
        
        metrics = {
            "timestamp": vector_health_checker.status.last_check.isoformat(),
            "database_metrics": {
                "total_users": db.query(UserModel).count(),
                "total_documents": db.query(Document).filter(Document.is_deleted == False).count(),
                "total_chunks": db.query(DocumentChunk).count(),
                "total_vector_indices": db.query(VectorIndex).count(),
                "documents_processing": db.query(Document).filter(
                    Document.status == "processing"
                ).count(),
                "documents_completed": db.query(Document).filter(
                    Document.status == "completed"
                ).count(),
                "documents_failed": db.query(Document).filter(
                    Document.status == "failed"
                ).count()
            },
            "vector_metrics": {
                "faiss_stats": faiss_stats,
                "storage_stats": performance_stats
            },
            "health_summary": {
                "database_status": "healthy",  # Simplified for metrics
                "vector_status": "healthy",
                "overall_status": "healthy"
            }
        }
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health metrics collection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to collect health metrics: {str(e)}"
        )