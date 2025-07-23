"""
Administrative routes for system management and monitoring.
Provides endpoints for system administrators to manage the RAG system.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel

from database.connection import get_db
from api.middleware.auth import get_current_active_user
from database.models import User, Document, DocumentChunk, VectorIndex, UserRole, Permission, SearchQuery
from api.controllers.auth_controller import AuthController
from vector_db.health_checker import get_vector_health_checker
from utils.monitoring.health_check import HealthChecker
from config import get_settings

router = APIRouter(prefix="/admin", tags=["administration"])
settings = get_settings()

class SystemStatsResponse(BaseModel):
    total_users: int
    active_users: int
    total_documents: int
    processing_documents: int
    total_chunks: int
    total_vector_indices: int
    total_searches: int
    storage_used_mb: float
    system_uptime_hours: float

class UserStatsResponse(BaseModel):
    user_id: int
    username: str
    email: str
    document_count: int
    chunk_count: int
    search_count: int
    storage_used_mb: float
    last_activity: Optional[str]

class RolePermissionRequest(BaseModel):
    user_id: int
    role_name: str

class PermissionRequest(BaseModel):
    user_id: int
    permission_name: str

async def verify_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Verify that the current user is an admin."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required"
        )
    return current_user

@router.get("/stats/system", response_model=SystemStatsResponse)
async def get_system_statistics(
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive system statistics."""
    try:
        # User statistics
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        
        # Document statistics
        total_documents = db.query(Document).filter(Document.is_deleted == False).count()
        processing_documents = db.query(Document).filter(
            Document.status == "processing"
        ).count()
        
        # Vector statistics
        total_chunks = db.query(DocumentChunk).count()
        total_vector_indices = db.query(VectorIndex).count()
        
        # Search statistics
        total_searches = db.query(SearchQuery).count()
        
        # Storage statistics (estimate)
        from vector_db.storage_manager import get_storage_manager
        storage_manager = get_storage_manager()
        perf_stats = await storage_manager.get_performance_stats()
        storage_used_mb = perf_stats.get('storage_size_mb', 0)
        
        # System uptime (simplified calculation)
        system_uptime_hours = 24.0  # Placeholder - would need actual startup time
        
        return SystemStatsResponse(
            total_users=total_users,
            active_users=active_users,
            total_documents=total_documents,
            processing_documents=processing_documents,
            total_chunks=total_chunks,
            total_vector_indices=total_vector_indices,
            total_searches=total_searches,
            storage_used_mb=storage_used_mb,
            system_uptime_hours=system_uptime_hours
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system statistics: {str(e)}"
        )

@router.get("/stats/users", response_model=List[UserStatsResponse])
async def get_user_statistics(
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """Get detailed statistics for all users."""
    try:
        users = db.query(User).offset(skip).limit(limit).all()
        user_stats = []
        
        for user in users:
            # Get user's document count
            doc_count = db.query(Document).filter(
                Document.user_id == user.id,
                Document.is_deleted == False
            ).count()
            
            # Get user's chunk count
            chunk_count = db.query(DocumentChunk).join(Document).filter(
                Document.user_id == user.id,
                Document.is_deleted == False
            ).count()
            
            # Get user's search count
            search_count = db.query(SearchQuery).filter(
                SearchQuery.user_id == user.id
            ).count()
            
            # Estimate storage used
            user_docs = db.query(Document).filter(
                Document.user_id == user.id,
                Document.is_deleted == False
            ).all()
            storage_used_mb = sum(doc.file_size for doc in user_docs) / (1024 * 1024)
            
            user_stats.append(UserStatsResponse(
                user_id=user.id,
                username=user.username,
                email=user.email,
                document_count=doc_count,
                chunk_count=chunk_count,
                search_count=search_count,
                storage_used_mb=storage_used_mb,
                last_activity=user.last_login.isoformat() if user.last_login else None
            ))
        
        return user_stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user statistics: {str(e)}"
        )

@router.get("/health/comprehensive")
async def get_comprehensive_health(
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive system health status."""
    try:
        # Get vector database health
        vector_health_checker = get_vector_health_checker()
        vector_health = await vector_health_checker.check_all_components(db)
        
        # Get system health
        system_health_checker = HealthChecker()
        system_health = await system_health_checker.run_all_checks()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": system_health,
            "vector_health": vector_health,
            "overall_status": "healthy" if all([
                system_health.get("status") == "healthy",
                vector_health.get("status") == "healthy"
            ]) else "degraded"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve health status: {str(e)}"
        )

@router.post("/users/{user_id}/roles")
async def assign_role_to_user(
    user_id: int,
    role_request: RolePermissionRequest,
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """Assign a role to a user."""
    try:
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get role
        role = db.query(UserRole).filter(UserRole.name == role_request.role_name).first()
        if not role:
            raise HTTPException(status_code=404, detail="Role not found")
        
        # Check if user already has the role
        if role in user.roles:
            raise HTTPException(status_code=400, detail="User already has this role")
        
        # Assign role
        user.roles.append(role)
        db.commit()
        
        return {
            "message": f"Role '{role_request.role_name}' assigned to user '{user.username}'",
            "user_id": user_id,
            "role_name": role_request.role_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assign role: {str(e)}"
        )

@router.delete("/users/{user_id}/roles/{role_name}")
async def remove_role_from_user(
    user_id: int,
    role_name: str,
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """Remove a role from a user."""
    try:
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get role
        role = db.query(UserRole).filter(UserRole.name == role_name).first()
        if not role:
            raise HTTPException(status_code=404, detail="Role not found")
        
        # Check if user has the role
        if role not in user.roles:
            raise HTTPException(status_code=400, detail="User does not have this role")
        
        # Remove role
        user.roles.remove(role)
        db.commit()
        
        return {
            "message": f"Role '{role_name}' removed from user '{user.username}'",
            "user_id": user_id,
            "role_name": role_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove role: {str(e)}"
        )

@router.post("/users/{user_id}/activate")
async def activate_user(
    user_id: int,
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """Activate a user account."""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user.is_active = True
        user.is_verified = True
        db.commit()
        
        return {
            "message": f"User '{user.username}' activated successfully",
            "user_id": user_id,
            "is_active": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate user: {str(e)}"
        )

@router.post("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: int,
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """Deactivate a user account."""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Prevent deactivating superusers
        if user.is_superuser:
            raise HTTPException(
                status_code=400, 
                detail="Cannot deactivate superuser accounts"
            )
        
        user.is_active = False
        db.commit()
        
        return {
            "message": f"User '{user.username}' deactivated successfully",
            "user_id": user_id,
            "is_active": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deactivate user: {str(e)}"
        )

@router.get("/documents/orphaned")
async def get_orphaned_documents(
    limit: int = Query(50, ge=1, le=100),
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """Get documents without associated vector indices (orphaned documents)."""
    try:
        # Find documents that don't have corresponding vector indices
        orphaned_docs = db.query(Document).outerjoin(VectorIndex).filter(
            Document.is_deleted == False,
            VectorIndex.document_id.is_(None)
        ).limit(limit).all()
        
        orphaned_data = []
        for doc in orphaned_docs:
            orphaned_data.append({
                "id": doc.id,
                "filename": doc.filename,
                "user_id": doc.user_id,
                "status": doc.status.value,
                "created_at": doc.created_at.isoformat(),
                "file_size": doc.file_size,
                "chunks_count": len(doc.chunks)
            })
        
        return {
            "orphaned_documents": orphaned_data,
            "count": len(orphaned_data),
            "message": f"Found {len(orphaned_data)} orphaned documents"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve orphaned documents: {str(e)}"
        )

@router.post("/cleanup/orphaned-documents")
async def cleanup_orphaned_documents(
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """Clean up orphaned documents by creating missing vector indices or removing invalid documents."""
    try:
        # Find orphaned documents
        orphaned_docs = db.query(Document).outerjoin(VectorIndex).filter(
            Document.is_deleted == False,
            VectorIndex.document_id.is_(None)
        ).all()
        
        cleaned_count = 0
        errors = []
        
        for doc in orphaned_docs:
            try:
                # Try to create missing vector index
                from vector_db.document_version_manager import get_version_manager
                version_manager = get_version_manager()
                
                # This would re-process the document to create vector indices
                # For now, we'll just mark them for manual review
                doc.status = "needs_reprocessing"
                cleaned_count += 1
                
            except Exception as e:
                errors.append({
                    "document_id": doc.id,
                    "error": str(e)
                })
        
        db.commit()
        
        return {
            "message": f"Marked {cleaned_count} orphaned documents for reprocessing",
            "cleaned_count": cleaned_count,
            "errors": errors
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup orphaned documents: {str(e)}"
        )

@router.get("/search/analytics")
async def get_search_analytics(
    days: int = Query(7, ge=1, le=365),
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """Get search analytics for the specified time period."""
    try:
        from sqlalchemy import func
        from datetime import datetime, timedelta
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Total searches
        total_searches = db.query(SearchQuery).filter(
            SearchQuery.created_at >= since_date
        ).count()
        
        # Searches by type
        search_types = db.query(
            SearchQuery.query_type,
            func.count(SearchQuery.id).label('count')
        ).filter(
            SearchQuery.created_at >= since_date
        ).group_by(SearchQuery.query_type).all()
        
        # Average search time
        avg_search_time = db.query(
            func.avg(SearchQuery.search_time_ms)
        ).filter(
            SearchQuery.created_at >= since_date,
            SearchQuery.search_time_ms.isnot(None)
        ).scalar()
        
        # Top users by search count
        top_users = db.query(
            User.username,
            func.count(SearchQuery.id).label('search_count')
        ).join(SearchQuery).filter(
            SearchQuery.created_at >= since_date
        ).group_by(User.id, User.username).order_by(
            func.count(SearchQuery.id).desc()
        ).limit(10).all()
        
        return {
            "period_days": days,
            "total_searches": total_searches,
            "search_types": [{"type": st.query_type, "count": st.count} for st in search_types],
            "average_search_time_ms": float(avg_search_time) if avg_search_time else 0,
            "top_users": [{"username": user.username, "search_count": user.search_count} for user in top_users]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve search analytics: {str(e)}"
        )

@router.get("/roles")
async def list_roles(
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """List all available roles and their permissions."""
    try:
        roles = db.query(UserRole).all()
        roles_data = []
        
        for role in roles:
            roles_data.append({
                "id": role.id,
                "name": role.name.value,
                "description": role.description,
                "permissions": [perm.name.value for perm in role.permissions],
                "user_count": len(role.users),
                "created_at": role.created_at.isoformat()
            })
        
        return {
            "roles": roles_data,
            "total_roles": len(roles_data)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve roles: {str(e)}"
        )

@router.get("/permissions")
async def list_permissions(
    admin_user: User = Depends(verify_admin_user),
    db: Session = Depends(get_db)
):
    """List all available permissions."""
    try:
        permissions = db.query(Permission).all()
        permissions_data = []
        
        for perm in permissions:
            permissions_data.append({
                "id": perm.id,
                "name": perm.name.value,
                "description": perm.description,
                "created_at": perm.created_at.isoformat()
            })
        
        return {
            "permissions": permissions_data,
            "total_permissions": len(permissions_data)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve permissions: {str(e)}"
        )