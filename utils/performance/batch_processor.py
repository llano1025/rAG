"""
Enhanced batch processing system for large-scale operations.
Provides queue-based processing, progress tracking, and optimized performance.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Callable, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import json
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class BatchStatus(Enum):
    """Batch processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Priority(Enum):
    """Processing priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class BatchItem:
    """Individual item in a batch."""
    id: str
    data: Any
    priority: Priority = Priority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Any = None

@dataclass
class BatchJob:
    """Batch processing job."""
    id: str
    name: str
    items: List[BatchItem]
    status: BatchStatus = BatchStatus.PENDING
    progress: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    worker_count: int = 4
    batch_size: int = 10
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)

class BatchProcessor:
    """
    Enhanced batch processor for large-scale operations.
    
    Features:
    - Queue-based processing with priorities
    - Configurable worker pools
    - Progress tracking and monitoring
    - Retry mechanisms with backoff
    - Memory management for large batches
    - Async/sync operation support
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        default_batch_size: int = 50,
        max_memory_mb: int = 1024,
        enable_monitoring: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of concurrent workers
            default_batch_size: Default batch processing size
            max_memory_mb: Maximum memory usage in MB
            enable_monitoring: Enable performance monitoring
        """
        self.max_workers = max_workers
        self.default_batch_size = default_batch_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_monitoring = enable_monitoring
        
        # Job management
        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue = asyncio.Queue()
        self.workers_running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # Performance monitoring
        self.stats = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'total_items_processed': 0,
            'average_processing_time': 0.0,
            'memory_usage_mb': 0.0
        }
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    
    async def start_workers(self):
        """Start worker tasks for processing batches."""
        if self.workers_running:
            return
        
        self.workers_running = True
        
        # Create worker tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        logger.info(f"Started {self.max_workers} batch processing workers")
    
    async def stop_workers(self):
        """Stop all worker tasks."""
        if not self.workers_running:
            return
        
        self.workers_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Clear worker tasks
        self.worker_tasks.clear()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Stopped all batch processing workers")
    
    async def submit_batch(
        self,
        job_id: str,
        job_name: str,
        items: List[Any],
        processor: Callable,
        priority: Priority = Priority.NORMAL,
        batch_size: Optional[int] = None,
        worker_count: Optional[int] = None,
        timeout_seconds: int = 300,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a batch job for processing.
        
        Args:
            job_id: Unique job identifier
            job_name: Human-readable job name
            items: List of items to process
            processor: Processing function (async or sync)
            priority: Job priority
            batch_size: Items per batch (overrides default)
            worker_count: Number of workers for this job
            timeout_seconds: Job timeout in seconds
            metadata: Additional job metadata
            
        Returns:
            Job ID
        """
        if job_id in self.jobs:
            raise ValueError(f"Job {job_id} already exists")
        
        # Create batch items
        batch_items = []
        for i, item_data in enumerate(items):
            batch_item = BatchItem(
                id=f"{job_id}-item-{i}",
                data=item_data,
                priority=priority
            )
            batch_items.append(batch_item)
        
        # Create batch job
        job = BatchJob(
            id=job_id,
            name=job_name,
            items=batch_items,
            total_items=len(batch_items),
            worker_count=worker_count or min(self.max_workers, len(batch_items)),
            batch_size=batch_size or self.default_batch_size,
            timeout_seconds=timeout_seconds,
            metadata=metadata or {}
        )
        
        # Store job
        self.jobs[job_id] = job
        
        # Add to processing queue
        await self.job_queue.put((priority.value, job_id, processor))
        
        # Update stats
        self.stats['total_jobs'] += 1
        
        logger.info(f"Submitted batch job {job_id} with {len(items)} items")
        
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a batch job."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        return {
            'id': job.id,
            'name': job.name,
            'status': job.status.value,
            'progress': job.progress,
            'total_items': job.total_items,
            'completed_items': job.completed_items,
            'failed_items': job.failed_items,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'elapsed_time': self._calculate_elapsed_time(job),
            'estimated_remaining': self._estimate_remaining_time(job),
            'metadata': job.metadata,
            'worker_count': job.worker_count,
            'batch_size': job.batch_size
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job."""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if job.status not in [BatchStatus.PENDING, BatchStatus.PROCESSING]:
            return False
        
        job.status = BatchStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        
        logger.info(f"Cancelled batch job {job_id}")
        
        return True
    
    async def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get results of a completed batch job."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        if job.status != BatchStatus.COMPLETED:
            return None
        
        # Collect results
        successful_results = []
        failed_results = []
        
        for item in job.items:
            if item.result is not None:
                successful_results.append({
                    'id': item.id,
                    'result': item.result,
                    'processing_time': (
                        item.completed_at - item.started_at
                    ).total_seconds() if item.completed_at and item.started_at else None
                })
            elif item.error:
                failed_results.append({
                    'id': item.id,
                    'error': item.error,
                    'retry_count': item.retry_count
                })
        
        return {
            'job_id': job_id,
            'successful_results': successful_results,
            'failed_results': failed_results,
            'summary': {
                'total_items': job.total_items,
                'successful': len(successful_results),
                'failed': len(failed_results),
                'processing_time': (
                    job.completed_at - job.started_at
                ).total_seconds() if job.completed_at and job.started_at else None
            }
        }
    
    async def list_jobs(
        self,
        status_filter: Optional[BatchStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List batch jobs with optional filtering."""
        jobs = []
        
        for job in self.jobs.values():
            if status_filter and job.status != status_filter:
                continue
            
            job_info = await self.get_job_status(job.id)
            jobs.append(job_info)
            
            if len(jobs) >= limit:
                break
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jobs
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        import psutil
        import os
        
        # Memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Active jobs
        active_jobs = sum(1 for job in self.jobs.values() 
                         if job.status in [BatchStatus.PENDING, BatchStatus.PROCESSING])
        
        return {
            **self.stats,
            'active_jobs': active_jobs,
            'total_jobs_in_memory': len(self.jobs),
            'current_memory_mb': memory_mb,
            'worker_count': len(self.worker_tasks),
            'workers_running': self.workers_running,
            'queue_size': self.job_queue.qsize()
        }
    
    async def cleanup_completed_jobs(self, older_than_hours: int = 24):
        """Remove completed jobs older than specified hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        
        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if (job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED] and
                job.completed_at and job.completed_at < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        logger.info(f"Cleaned up {len(jobs_to_remove)} completed jobs")
        
        return len(jobs_to_remove)
    
    async def _worker(self, worker_name: str):
        """Worker task for processing batch jobs."""
        logger.info(f"Started batch worker: {worker_name}")
        
        while self.workers_running:
            try:
                # Get next job from queue (with timeout)
                priority, job_id, processor = await asyncio.wait_for(
                    self.job_queue.get(), timeout=1.0
                )
                
                if job_id not in self.jobs:
                    continue
                
                job = self.jobs[job_id]
                
                # Skip if job was cancelled
                if job.status == BatchStatus.CANCELLED:
                    continue
                
                # Process the job
                await self._process_job(job, processor, worker_name)
                
            except asyncio.TimeoutError:
                # No jobs in queue, continue waiting
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                continue
        
        logger.info(f"Stopped batch worker: {worker_name}")
    
    async def _process_job(self, job: BatchJob, processor: Callable, worker_name: str):
        """Process a batch job."""
        logger.info(f"Worker {worker_name} processing job {job.id}")
        
        # Update job status
        job.status = BatchStatus.PROCESSING
        job.started_at = datetime.now(timezone.utc)
        
        try:
            # Process items in batches
            for i in range(0, len(job.items), job.batch_size):
                batch_items = job.items[i:i + job.batch_size]
                
                # Process batch items concurrently
                tasks = []
                for item in batch_items:
                    if job.status == BatchStatus.CANCELLED:
                        break
                    
                    task = asyncio.create_task(
                        self._process_item(item, processor, worker_name)
                    )
                    tasks.append(task)
                
                # Wait for batch to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update progress
                completed = sum(1 for item in job.items if item.result is not None or item.error)
                job.completed_items = completed
                job.failed_items = sum(1 for item in job.items if item.error)
                job.progress = completed / job.total_items * 100
                
                # Check memory usage
                await self._check_memory_usage()
            
            # Mark job as completed
            if job.status != BatchStatus.CANCELLED:
                job.status = BatchStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                self.stats['completed_jobs'] += 1
            
        except Exception as e:
            logger.error(f"Job {job.id} processing failed: {e}")
            job.status = BatchStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            self.stats['failed_jobs'] += 1
        
        # Update processing time stats
        if job.started_at and job.completed_at:
            processing_time = (job.completed_at - job.started_at).total_seconds()
            self._update_average_processing_time(processing_time)
        
        logger.info(f"Worker {worker_name} completed job {job.id} with status {job.status.value}")
    
    async def _process_item(self, item: BatchItem, processor: Callable, worker_name: str):
        """Process a single batch item."""
        item.started_at = datetime.now(timezone.utc)
        
        for attempt in range(item.max_retries + 1):
            try:
                # Check if processor is async or sync
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(item.data)
                else:
                    # Run sync processor in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.thread_pool, processor, item.data
                    )
                
                item.result = result
                item.completed_at = datetime.now(timezone.utc)
                self.stats['total_items_processed'] += 1
                break
                
            except Exception as e:
                item.retry_count = attempt
                error_msg = f"Attempt {attempt + 1}/{item.max_retries + 1}: {str(e)}"
                
                if attempt < item.max_retries:
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(2 ** attempt)
                    logger.warning(f"Item {item.id} retry {attempt + 1}: {e}")
                else:
                    # Max retries reached
                    item.error = error_msg
                    item.completed_at = datetime.now(timezone.utc)
                    logger.error(f"Item {item.id} failed after {item.max_retries + 1} attempts: {e}")
                    break
    
    async def _check_memory_usage(self):
        """Check current memory usage and log warnings."""
        if not self.enable_monitoring:
            return
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_bytes = process.memory_info().rss
            memory_mb = memory_bytes / (1024 * 1024)
            
            self.stats['memory_usage_mb'] = memory_mb
            
            if memory_bytes > self.max_memory_bytes:
                logger.warning(
                    f"Memory usage ({memory_mb:.1f}MB) exceeds limit "
                    f"({self.max_memory_bytes / (1024 * 1024):.1f}MB)"
                )
                
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
    
    def _calculate_elapsed_time(self, job: BatchJob) -> Optional[float]:
        """Calculate elapsed time for a job."""
        if not job.started_at:
            return None
        
        end_time = job.completed_at or datetime.now(timezone.utc)
        return (end_time - job.started_at).total_seconds()
    
    def _estimate_remaining_time(self, job: BatchJob) -> Optional[float]:
        """Estimate remaining time for a job."""
        if job.status != BatchStatus.PROCESSING or not job.started_at:
            return None
        
        if job.completed_items == 0:
            return None
        
        elapsed = (datetime.now(timezone.utc) - job.started_at).total_seconds()
        rate = job.completed_items / elapsed  # items per second
        
        remaining_items = job.total_items - job.completed_items
        
        if rate > 0:
            return remaining_items / rate
        
        return None
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time statistic."""
        current_avg = self.stats['average_processing_time']
        completed_jobs = self.stats['completed_jobs']
        
        if completed_jobs == 1:
            self.stats['average_processing_time'] = processing_time
        else:
            # Running average
            new_avg = ((current_avg * (completed_jobs - 1)) + processing_time) / completed_jobs
            self.stats['average_processing_time'] = new_avg


# Global batch processor instance
_batch_processor: Optional[BatchProcessor] = None

def get_batch_processor() -> BatchProcessor:
    """Get global batch processor instance."""
    global _batch_processor
    
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    
    return _batch_processor

@asynccontextmanager
async def batch_processor_context():
    """Context manager for batch processor lifecycle."""
    processor = get_batch_processor()
    
    try:
        await processor.start_workers()
        yield processor
    finally:
        await processor.stop_workers()


# Convenience functions for common batch operations
async def process_document_batch(
    documents: List[Dict[str, Any]],
    processor_func: Callable,
    job_name: str = "Document Batch Processing",
    batch_size: int = 10,
    priority: Priority = Priority.NORMAL
) -> str:
    """
    Convenience function for processing document batches.
    
    Args:
        documents: List of document data
        processor_func: Function to process each document
        job_name: Name for the batch job
        batch_size: Number of documents per batch
        priority: Processing priority
        
    Returns:
        Job ID
    """
    processor = get_batch_processor()
    
    job_id = f"doc_batch_{int(time.time())}"
    
    return await processor.submit_batch(
        job_id=job_id,
        job_name=job_name,
        items=documents,
        processor=processor_func,
        priority=priority,
        batch_size=batch_size
    )

async def process_search_batch(
    queries: List[str],
    search_func: Callable,
    job_name: str = "Search Batch Processing",
    batch_size: int = 20,
    priority: Priority = Priority.HIGH
) -> str:
    """
    Convenience function for processing search batches.
    
    Args:
        queries: List of search queries
        search_func: Function to process each query
        job_name: Name for the batch job
        batch_size: Number of queries per batch
        priority: Processing priority
        
    Returns:
        Job ID
    """
    processor = get_batch_processor()
    
    job_id = f"search_batch_{int(time.time())}"
    
    return await processor.submit_batch(
        job_id=job_id,
        job_name=job_name,
        items=queries,
        processor=search_func,
        priority=priority,
        batch_size=batch_size
    )