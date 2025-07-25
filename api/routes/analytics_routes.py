from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pydantic import BaseModel

from database.connection import get_db
from api.middleware.auth import get_current_active_user
from database.models import User
from utils.monitoring.health_check import HealthChecker

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Initialize health checker
health_checker = HealthChecker()

class UsageStats(BaseModel):
    total_documents: int = 0
    total_searches: int = 0
    active_users: int = 0
    storage_used: int = 0
    upload_trends: list = []
    search_trends: list = []

class SystemMetrics(BaseModel):
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    active_connections: int = 0

class ComponentHealth(BaseModel):
    name: str
    status: str
    type: str
    details: Dict[str, Any] = {}

class SystemHealth(BaseModel):
    status: str
    timestamp: str
    components: List[ComponentHealth] = []

@router.get("/usage-stats", response_model=UsageStats)
async def get_usage_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get system usage statistics for the specified time period.
    """
    try:
        # For now, return placeholder data
        # TODO: Implement actual analytics from database and monitoring systems
        stats = UsageStats(
            total_documents=0,
            total_searches=0,
            active_users=1,
            storage_used=0,
            upload_trends=[],
            search_trends=[]
        )
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve usage statistics: {str(e)}"
        )

@router.get("/system-metrics", response_model=SystemMetrics)
async def get_system_metrics(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current system performance metrics.
    """
    try:
        # Get actual system metrics from health checker
        health_results = await health_checker.run_all_checks()
        
        cpu_usage = 0.0
        memory_usage = 0.0
        disk_usage = 0.0
        
        # Extract metrics from health check results
        components = health_results.get('components', {})
        
        # Get system resources data
        if 'system_resources' in components:
            sys_details = components['system_resources'].get('details', {})
            cpu_usage = sys_details.get('cpu_usage_percent', 0.0)
            memory_usage = sys_details.get('memory_usage_percent', 0.0)
        
        # Get disk usage data
        if 'disk_usage' in components:
            disk_details = components['disk_usage'].get('details', {})
            disk_usage = disk_details.get('usage_percent', 0.0)
        
        metrics = SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            active_connections=1  # TODO: Implement actual connection counting
        )
        
        return metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system metrics: {str(e)}"
        )

@router.get("/system-health", response_model=SystemHealth)
async def get_system_health(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get system health status with detailed component information.
    """
    try:
        # Get health check results
        health_results = await health_checker.run_all_checks()
        
        # Build component health list
        components = []
        health_components = health_results.get('components', {})
        
        for component_name, component_data in health_components.items():
            # Determine component type based on name
            if component_name == 'system_resources':
                component_type = 'system_resources'
            elif component_name == 'disk_usage':
                component_type = 'disk_usage'
            else:
                component_type = 'other'
            
            components.append(ComponentHealth(
                name=component_name.replace('_', ' ').title(),
                status=component_data.get('status', 'unknown').value if hasattr(component_data.get('status', 'unknown'), 'value') else str(component_data.get('status', 'unknown')),
                type=component_type,
                details=component_data.get('details', {})
            ))
        
        # Determine overall status
        overall_status = health_results.get('status', 'unknown')
        if hasattr(overall_status, 'value'):
            overall_status = overall_status.value
        
        system_health = SystemHealth(
            status=overall_status,
            timestamp=health_results.get('timestamp', datetime.now().isoformat()),
            components=components
        )
        
        return system_health
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system health: {str(e)}"
        )

@router.get("/document-stats")
async def get_document_stats(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get document-related statistics.
    """
    try:
        return {
            "total_documents": 0,
            "documents_by_type": {},
            "upload_trends": [],
            "top_uploaders": []
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document statistics: {str(e)}"
        )

@router.get("/search-stats")
async def get_search_stats(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get search-related statistics.
    """
    try:
        return {
            "total_searches": 0,
            "search_trends": [],
            "popular_queries": [],
            "average_response_time": 0.0
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve search statistics: {str(e)}"
        )

@router.get("/user-activity")
async def get_user_activity(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get user activity statistics.
    """
    try:
        return {
            "active_users": 1,
            "user_activity_trends": [],
            "top_active_users": [],
            "new_registrations": []
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user activity: {str(e)}"
        )