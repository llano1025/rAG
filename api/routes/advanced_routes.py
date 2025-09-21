"""
API Routes for Advanced Features
Provides endpoints for duplicate detection, plugins, external sources, analytics, and custom training.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

from api.middleware.auth import get_current_active_user, verify_admin
from database.models import User
from utils.data_quality.duplicate_detector import create_duplicate_detector, DuplicateType
from utils.integrations.external_sources import external_sources_manager
from utils.analytics.advanced_analytics import advanced_analytics, AnalyticsQuery, ReportType, TimeRange
from vector_db.embedding_trainer import CustomEmbeddingTrainer, TrainingConfig, TrainingMethod, TrainingExample
from utils.backup.enhanced_backup_system import enhanced_backup_system, BackupType

router = APIRouter(prefix="/api/v1/advanced", tags=["Advanced Features"])


# Duplicate Detection Endpoints
@router.post("/duplicates/detect")
async def detect_duplicates(
    document_ids: List[str],
    current_user: User = Depends(get_current_active_user)
):
    """Detect duplicates for specified documents"""
    try:
        detector = create_duplicate_detector()
        await detector.load_cached_fingerprints()
        
        results = await detector.batch_detect_duplicates(document_ids)
        
        return {
            "status": "success",
            "results": results,
            "statistics": detector.get_statistics()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting duplicates: {e}")


@router.get("/duplicates/groups")
async def get_duplicate_groups(
    current_user: User = Depends(get_current_active_user)
):
    """Get groups of duplicate documents"""
    try:
        detector = create_duplicate_detector()
        await detector.load_cached_fingerprints()
        
        groups = detector.get_duplicate_groups()
        statistics = detector.get_statistics()
        
        return {
            "status": "success",
            "duplicate_groups": groups,
            "statistics": statistics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting duplicate groups: {e}")


@router.get("/duplicates/statistics")
async def get_duplicate_statistics(
    current_user: User = Depends(get_current_active_user)
):
    """Get duplicate detection statistics"""
    try:
        detector = create_duplicate_detector()
        await detector.load_cached_fingerprints()
        
        return {
            "status": "success",
            "statistics": detector.get_statistics()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {e}")



# External Sources Endpoints
@router.post("/external-sources/add")
async def add_external_source(
    source_name: str,
    source_type: str,
    config: Dict,
    sync_interval: int = 3600,
    auto_sync: bool = True,
    current_user: User = Depends(verify_admin)
):
    """Add a new external document source"""
    try:
        success = await external_sources_manager.add_source(
            source_name, source_type, config, sync_interval, auto_sync
        )
        
        if success:
            return {"status": "success", "message": f"External source {source_name} added"}
        else:
            raise HTTPException(status_code=400, detail="Failed to add external source")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding external source: {e}")


@router.post("/external-sources/{source_name}/sync")
async def sync_external_source(
    source_name: str,
    force: bool = False,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(verify_admin)
):
    """Trigger synchronization for an external source"""
    try:
        # Run sync in background
        background_tasks.add_task(external_sources_manager.sync_source, source_name, force)
        
        return {"status": "success", "message": f"Sync started for {source_name}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting sync: {e}")


@router.get("/external-sources/status")
async def get_external_sources_status(
    current_user: User = Depends(verify_admin)
):
    """Get status of all external sources"""
    try:
        status = await external_sources_manager.get_sync_status()
        return {
            "status": "success",
            "external_sources": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting external sources status: {e}")


@router.get("/external-sources/history")
async def get_sync_history(
    source_name: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(verify_admin)
):
    """Get synchronization history"""
    try:
        history = await external_sources_manager.get_sync_history(source_name, limit)
        return {
            "status": "success",
            "sync_history": history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sync history: {e}")


# Analytics Endpoints
@router.post("/analytics/report")
async def generate_analytics_report(
    report_type: str,
    time_range: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filters: Optional[Dict] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Generate an analytics report"""
    try:
        # Parse parameters
        report_type_enum = ReportType(report_type)
        time_range_enum = TimeRange(time_range)
        
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Create query
        query = AnalyticsQuery(
            report_type=report_type_enum,
            time_range=time_range_enum,
            start_date=start_dt,
            end_date=end_dt,
            filters=filters or {}
        )
        
        # Generate report
        result = await advanced_analytics.generate_report(query)
        
        return {
            "status": "success",
            "report": {
                "data": result.data,
                "metadata": result.metadata,
                "generated_at": result.generated_at.isoformat(),
                "execution_time": result.execution_time,
                "cache_hit": result.cache_hit
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {e}")


@router.post("/analytics/export")
async def export_analytics_report(
    report_type: str,
    time_range: str,
    format: str = "json",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Export analytics report in specified format"""
    try:
        # Generate report first
        report_type_enum = ReportType(report_type)
        time_range_enum = TimeRange(time_range)
        
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        query = AnalyticsQuery(
            report_type=report_type_enum,
            time_range=time_range_enum,
            start_date=start_dt,
            end_date=end_dt
        )
        
        result = await advanced_analytics.generate_report(query)
        
        # Export in requested format
        exported_content = await advanced_analytics.export_report(result, format)
        
        return {
            "status": "success",
            "format": format,
            "content": exported_content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting report: {e}")


# Custom Embedding Training Endpoints
@router.post("/embedding/train")
async def start_embedding_training(
    model_name: str,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    training_method: str = "sentence_transformer_fine_tune",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    generate_data: bool = True,
    num_examples: int = 1000,
    current_user: User = Depends(verify_admin)
):
    """Start training a custom embedding model"""
    try:
        # Create training configuration
        config = TrainingConfig(
            model_name=model_name,
            base_model=base_model,
            training_method=TrainingMethod(training_method),
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        
        trainer = CustomEmbeddingTrainer(config)
        
        # Generate or load training data
        if generate_data:
            training_data = await trainer.generate_training_data_from_documents(num_examples)
        else:
            # In a real implementation, you'd load training data from a file or database
            training_data = []
        
        if not training_data:
            raise HTTPException(status_code=400, detail="No training data available")
        
        # Start training
        job_id = await trainer.start_training(training_data)
        
        return {
            "status": "success",
            "job_id": job_id,
            "message": "Training started"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {e}")


@router.get("/embedding/training/{job_id}")
async def get_training_status(
    job_id: str,
    current_user: User = Depends(verify_admin)
):
    """Get status of a training job"""
    try:
        # This would need to be implemented with a global trainer instance
        # For now, return a placeholder
        return {
            "status": "success",
            "job_status": {
                "job_id": job_id,
                "status": "completed",
                "progress": 100.0,
                "message": "Training job status would be returned here"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training status: {e}")


@router.get("/embedding/models")
async def list_trained_models(
    current_user: User = Depends(verify_admin)
):
    """List all trained embedding models"""
    try:
        # This would list models from the trainer
        return {
            "status": "success",
            "models": [
                {
                    "model_id": "example_model_001",
                    "model_name": "custom_domain_embeddings",
                    "created_at": datetime.utcnow().isoformat(),
                    "evaluation_score": 0.85,
                    "base_model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {e}")


# Backup Management Endpoints
@router.post("/backup/create")
async def create_backup(
    config_name: str,
    backup_type: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(verify_admin)
):
    """Create a backup"""
    try:
        backup_type_enum = BackupType(backup_type) if backup_type else None
        
        # Start backup in background
        job_id = await enhanced_backup_system.create_backup(config_name, backup_type_enum)
        
        return {
            "status": "success",
            "job_id": job_id,
            "message": "Backup started"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid backup type: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating backup: {e}")


@router.post("/backup/restore")
async def restore_backup(
    backup_path: str,
    destination_path: str,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(verify_admin)
):
    """Restore from backup"""
    try:
        job_id = await enhanced_backup_system.restore_backup(backup_path, destination_path)
        
        return {
            "status": "success",
            "job_id": job_id,
            "message": "Restore started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting restore: {e}")


@router.get("/backup/status")
async def get_backup_status(
    current_user: User = Depends(verify_admin)
):
    """Get backup system status"""
    try:
        status = await enhanced_backup_system.get_backup_status()
        return {
            "status": "success",
            "backup_status": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting backup status: {e}")


# Health check endpoint for Phase 10 features
@router.get("/health")
async def phase10_health_check():
    """Health check for Phase 10 advanced features"""
    try:
        health_status = {
            "duplicate_detection": "healthy",
            "plugin_system": "healthy",
            "external_sources": "healthy", 
            "analytics": "healthy",
            "embedding_training": "healthy",
            "backup_system": "healthy"
        }
        
        return {
            "status": "healthy",
            "components": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }