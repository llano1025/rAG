from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct, desc, and_

from database.connection import get_db
from api.middleware.auth import get_current_active_user
from database.models import User, Document, SearchQuery, UserSession, UserActivityLog
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get system usage statistics for the specified time period.
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get total documents count
        total_documents = db.query(func.count(Document.id)).filter(
            Document.deleted_at.is_(None)
        ).scalar() or 0
        
        # Get total searches count in the time period
        total_searches = db.query(func.count(SearchQuery.id)).filter(
            SearchQuery.created_at >= cutoff_date
        ).scalar() or 0
        
        # Get active users (users who logged in or were active in the period)
        active_users = db.query(func.count(distinct(UserSession.user_id))).filter(
            and_(
                UserSession.created_at >= cutoff_date,
                UserSession.expires_at > datetime.now(timezone.utc)
            )
        ).scalar() or 0
        
        # Calculate total storage used (sum of document file sizes)
        storage_used = db.query(func.sum(Document.file_size)).filter(
            Document.deleted_at.is_(None)
        ).scalar() or 0
        
        # Get upload trends (documents created per day)
        upload_trends = []
        for i in range(min(days, 30)):  # Limit to 30 days for performance
            day_start = datetime.now(timezone.utc) - timedelta(days=i+1)
            day_end = datetime.now(timezone.utc) - timedelta(days=i)
            
            count = db.query(func.count(Document.id)).filter(
                and_(
                    Document.created_at >= day_start,
                    Document.created_at < day_end,
                    Document.deleted_at.is_(None)
                )
            ).scalar() or 0
            
            upload_trends.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "count": count
            })
        
        # Get search trends (searches per day)
        search_trends = []
        for i in range(min(days, 30)):  # Limit to 30 days for performance
            day_start = datetime.now(timezone.utc) - timedelta(days=i+1)
            day_end = datetime.now(timezone.utc) - timedelta(days=i)
            
            count = db.query(func.count(SearchQuery.id)).filter(
                and_(
                    SearchQuery.created_at >= day_start,
                    SearchQuery.created_at < day_end
                )
            ).scalar() or 0
            
            search_trends.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "count": count
            })
        
        stats = UsageStats(
            total_documents=total_documents,
            total_searches=total_searches,
            active_users=active_users,
            storage_used=storage_used,
            upload_trends=upload_trends,
            search_trends=search_trends
        )
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve usage statistics: {str(e)}"
        )

@router.get("/system-metrics", response_model=SystemMetrics)
async def get_system_metrics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
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
        
        # Get actual active connections count from user sessions
        active_connections = db.query(func.count(UserSession.id)).filter(
            UserSession.expires_at > datetime.now(timezone.utc)
        ).scalar() or 0
        
        metrics = SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            active_connections=active_connections
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get document-related statistics.
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get total documents count
        total_documents = db.query(func.count(Document.id)).filter(
            Document.deleted_at.is_(None)
        ).scalar() or 0
        
        # Get documents by content type
        documents_by_type = {}
        type_stats = db.query(
            Document.content_type, 
            func.count(Document.id)
        ).filter(
            Document.deleted_at.is_(None)
        ).group_by(Document.content_type).all()
        
        for content_type, count in type_stats:
            # Simplify MIME types for display
            if content_type.startswith('application/pdf'):
                key = 'PDF'
            elif content_type.startswith('text/'):
                key = 'Text'
            elif content_type.startswith('image/'):
                key = 'Image'
            elif content_type.startswith('application/vnd.openxmlformats-officedocument.wordprocessingml'):
                key = 'Word'
            elif content_type.startswith('application/vnd.openxmlformats-officedocument.spreadsheetml'):
                key = 'Excel'
            else:
                key = content_type.split('/')[-1].upper()[:10]
            
            documents_by_type[key] = documents_by_type.get(key, 0) + count
        
        # Get upload trends (documents created per day in the period)
        upload_trends = []
        for i in range(min(days, 30)):  # Limit to 30 days for performance
            day_start = datetime.now(timezone.utc) - timedelta(days=i+1)
            day_end = datetime.now(timezone.utc) - timedelta(days=i)
            
            count = db.query(func.count(Document.id)).filter(
                and_(
                    Document.created_at >= day_start,
                    Document.created_at < day_end,
                    Document.deleted_at.is_(None)
                )
            ).scalar() or 0
            
            upload_trends.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "count": count
            })
        
        # Get top uploaders (users with most documents in the period)
        top_uploaders = []
        uploader_stats = db.query(
            User.username,
            func.count(Document.id).label('doc_count')
        ).join(Document, Document.user_id == User.id).filter(
            and_(
                Document.created_at >= cutoff_date,
                Document.deleted_at.is_(None)
            )
        ).group_by(User.id, User.username).order_by(
            desc('doc_count')
        ).limit(5).all()
        
        for username, doc_count in uploader_stats:
            top_uploaders.append({
                "username": username,
                "document_count": doc_count
            })
        
        return {
            "total_documents": total_documents,
            "documents_by_type": documents_by_type,
            "upload_trends": upload_trends,
            "top_uploaders": top_uploaders
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document statistics: {str(e)}"
        )

@router.get("/search-stats")
async def get_search_stats(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get search-related statistics.
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get total searches count in the time period
        total_searches = db.query(func.count(SearchQuery.id)).filter(
            SearchQuery.created_at >= cutoff_date
        ).scalar() or 0
        
        # Get search trends (searches per day in the period)
        search_trends = []
        for i in range(min(days, 30)):  # Limit to 30 days for performance
            day_start = datetime.now(timezone.utc) - timedelta(days=i+1)
            day_end = datetime.now(timezone.utc) - timedelta(days=i)
            
            count = db.query(func.count(SearchQuery.id)).filter(
                and_(
                    SearchQuery.created_at >= day_start,
                    SearchQuery.created_at < day_end
                )
            ).scalar() or 0
            
            search_trends.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "count": count
            })
        
        # Get popular queries (most frequent query texts)
        popular_queries = []
        query_stats = db.query(
            SearchQuery.query_text,
            func.count(SearchQuery.id).label('query_count')
        ).filter(
            SearchQuery.created_at >= cutoff_date
        ).group_by(SearchQuery.query_text).order_by(
            desc('query_count')
        ).limit(10).all()
        
        for query_text, query_count in query_stats:
            popular_queries.append({
                "query": query_text[:100],  # Limit length for display
                "count": query_count
            })
        
        # Calculate average response time
        avg_response_time = db.query(
            func.avg(SearchQuery.search_time_ms)
        ).filter(
            and_(
                SearchQuery.created_at >= cutoff_date,
                SearchQuery.search_time_ms.is_not(None)
            )
        ).scalar()
        
        # Convert to float and handle None case
        average_response_time = float(avg_response_time) if avg_response_time else 0.0
        
        return {
            "total_searches": total_searches,
            "search_trends": search_trends,
            "popular_queries": popular_queries,
            "average_response_time": average_response_time
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve search statistics: {str(e)}"
        )

@router.get("/user-activity")
async def get_user_activity(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get user activity statistics.
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get active users count (users with sessions in the period)
        active_users = db.query(func.count(distinct(UserSession.user_id))).filter(
            UserSession.created_at >= cutoff_date
        ).scalar() or 0
        
        # Get user activity trends (active users per day)
        user_activity_trends = []
        for i in range(min(days, 30)):  # Limit to 30 days for performance
            day_start = datetime.now(timezone.utc) - timedelta(days=i+1)
            day_end = datetime.now(timezone.utc) - timedelta(days=i)
            
            count = db.query(func.count(distinct(UserSession.user_id))).filter(
                and_(
                    UserSession.created_at >= day_start,
                    UserSession.created_at < day_end
                )
            ).scalar() or 0
            
            user_activity_trends.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "active_users": count
            })
        
        # Get top active users (users with most activity logs)
        top_active_users = []
        activity_stats = db.query(
            User.username,
            func.count(UserActivityLog.id).label('activity_count')
        ).join(UserActivityLog, UserActivityLog.user_id == User.id).filter(
            UserActivityLog.created_at >= cutoff_date
        ).group_by(User.id, User.username).order_by(
            desc('activity_count')
        ).limit(5).all()
        
        for username, activity_count in activity_stats:
            top_active_users.append({
                "username": username,
                "activity_count": activity_count
            })
        
        # Get new registrations (users created in the period)
        new_registrations = []
        for i in range(min(days, 30)):  # Limit to 30 days for performance
            day_start = datetime.now(timezone.utc) - timedelta(days=i+1)
            day_end = datetime.now(timezone.utc) - timedelta(days=i)
            
            count = db.query(func.count(User.id)).filter(
                and_(
                    User.created_at >= day_start,
                    User.created_at < day_end,
                    User.deleted_at.is_(None)
                )
            ).scalar() or 0
            
            new_registrations.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "count": count
            })
        
        return {
            "active_users": active_users,
            "user_activity_trends": user_activity_trends,
            "top_active_users": top_active_users,
            "new_registrations": new_registrations
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user activity: {str(e)}"
        )