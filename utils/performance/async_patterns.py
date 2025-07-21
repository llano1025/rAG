"""
Advanced async processing patterns for high-performance RAG operations.
Provides worker pools, task queues, streaming patterns, and async context management.
"""

import asyncio
import logging
import time
import weakref
from typing import (
    List, Dict, Any, Optional, Callable, AsyncGenerator, AsyncIterator,
    Union, Tuple, Coroutine, TypeVar, Generic, AsyncContextManager
)
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import deque
from contextlib import asynccontextmanager
import json
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class WorkerStatus(Enum):
    """Worker status."""
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    STOPPED = "stopped"

@dataclass
class AsyncTask:
    """Async task representation."""
    id: str
    priority: TaskPriority
    coroutine: Coroutine
    callback: Optional[Callable] = None
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None

@dataclass
class WorkerMetrics:
    """Worker performance metrics."""
    worker_id: str
    status: WorkerStatus = WorkerStatus.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    current_task_id: Optional[str] = None
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time."""
        if self.tasks_completed == 0:
            return 0.0
        return self.total_processing_time / self.tasks_completed

class AsyncWorkerPool:
    """
    High-performance async worker pool with advanced patterns.
    
    Features:
    - Priority-based task scheduling
    - Dynamic worker scaling
    - Task streaming and batching
    - Graceful shutdown
    - Performance monitoring
    - Error handling and retries
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        min_workers: int = 2,
        queue_size: int = 1000,
        enable_auto_scaling: bool = True,
        worker_timeout: float = 300.0
    ):
        """
        Initialize async worker pool.
        
        Args:
            max_workers: Maximum number of workers
            min_workers: Minimum number of workers
            queue_size: Maximum queue size
            enable_auto_scaling: Enable automatic worker scaling
            worker_timeout: Worker timeout in seconds
        """
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.queue_size = queue_size
        self.enable_auto_scaling = enable_auto_scaling
        self.worker_timeout = worker_timeout
        
        # Task management
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=queue_size)
        self.pending_tasks: Dict[str, AsyncTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        
        # Worker management
        self.workers: Dict[str, asyncio.Task] = {}
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        self.worker_counter = 0
        
        # Pool state
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance metrics
        self.metrics = {
            'total_tasks_submitted': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'queue_size': 0,
            'active_workers': 0,
            'average_queue_time': 0.0,
            'average_processing_time': 0.0
        }
        
        # Auto-scaling
        self.scaling_task: Optional[asyncio.Task] = None
        self.last_scale_time = time.time()
        self.scale_cooldown = 30.0  # seconds
    
    async def start(self):
        """Start the worker pool."""
        if self.running:
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start minimum workers
        for _ in range(self.min_workers):
            await self._start_worker()
        
        # Start auto-scaling if enabled
        if self.enable_auto_scaling:
            self.scaling_task = asyncio.create_task(self._auto_scaling_loop())
        
        logger.info(f"Started async worker pool with {len(self.workers)} workers")
    
    async def stop(self, timeout: float = 30.0):
        """Stop the worker pool gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping async worker pool...")
        
        # Signal shutdown
        self.running = False
        self.shutdown_event.set()
        
        # Stop auto-scaling
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        
        # Wait for workers to finish current tasks
        if self.workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.workers.values(), return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Worker pool shutdown timeout, cancelling remaining workers")
                for worker in self.workers.values():
                    worker.cancel()
        
        # Clear state
        self.workers.clear()
        self.worker_metrics.clear()
        
        logger.info("Async worker pool stopped")
    
    async def submit_task(
        self,
        task_id: str,
        coroutine: Coroutine,
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Optional[Callable] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3
    ) -> str:
        """
        Submit a task to the worker pool.
        
        Args:
            task_id: Unique task identifier
            coroutine: Coroutine to execute
            priority: Task priority
            callback: Optional completion callback
            timeout: Task timeout
            max_retries: Maximum retry attempts
            
        Returns:
            Task ID
        """
        if not self.running:
            raise RuntimeError("Worker pool is not running")
        
        if task_id in self.pending_tasks:
            raise ValueError(f"Task {task_id} already exists")
        
        # Create task
        task = AsyncTask(
            id=task_id,
            priority=priority,
            coroutine=coroutine,
            callback=callback,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Add to pending tasks
        self.pending_tasks[task_id] = task
        
        # Add to queue (priority queue uses negative values for higher priority)
        priority_value = -priority.value
        await self.task_queue.put((priority_value, time.time(), task))
        
        # Update metrics
        self.metrics['total_tasks_submitted'] += 1
        self.metrics['queue_size'] = self.task_queue.qsize()
        
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a completed task.
        
        Args:
            task_id: Task identifier
            timeout: Wait timeout
            
        Returns:
            Task result
        """
        start_time = time.time()
        
        while True:
            # Check if task is completed
            if task_id not in self.pending_tasks:
                # Look in completed tasks
                for completed_task in self.completed_tasks:
                    if completed_task.id == task_id:
                        if completed_task.error:
                            raise completed_task.error
                        return completed_task.result
                
                raise ValueError(f"Task {task_id} not found")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timeout")
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if cancelled, False if not found or already completed
        """
        if task_id not in self.pending_tasks:
            return False
        
        task = self.pending_tasks.pop(task_id)
        
        # Cancel the coroutine if it's running
        if hasattr(task.coroutine, 'cancel'):
            task.coroutine.cancel()
        
        return True
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status."""
        return {
            'running': self.running,
            'workers': len(self.workers),
            'max_workers': self.max_workers,
            'min_workers': self.min_workers,
            'queue_size': self.task_queue.qsize(),
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'metrics': self.metrics.copy(),
            'worker_status': {
                worker_id: {
                    'status': metrics.status.value,
                    'tasks_completed': metrics.tasks_completed,
                    'tasks_failed': metrics.tasks_failed,
                    'average_processing_time': metrics.average_processing_time,
                    'current_task': metrics.current_task_id
                }
                for worker_id, metrics in self.worker_metrics.items()
            }
        }
    
    async def _start_worker(self) -> str:
        """Start a new worker."""
        worker_id = f"worker-{self.worker_counter}"
        self.worker_counter += 1
        
        # Create worker task
        worker_task = asyncio.create_task(self._worker_loop(worker_id))
        
        # Store worker
        self.workers[worker_id] = worker_task
        self.worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
        
        # Update metrics
        self.metrics['active_workers'] = len(self.workers)
        
        return worker_id
    
    async def _stop_worker(self, worker_id: str):
        """Stop a specific worker."""
        if worker_id not in self.workers:
            return
        
        worker_task = self.workers.pop(worker_id)
        worker_task.cancel()
        
        # Clean up metrics
        if worker_id in self.worker_metrics:
            del self.worker_metrics[worker_id]
        
        # Update metrics
        self.metrics['active_workers'] = len(self.workers)
    
    async def _worker_loop(self, worker_id: str):
        """Main worker loop."""
        metrics = self.worker_metrics[worker_id]
        
        logger.debug(f"Started worker: {worker_id}")
        
        try:
            while self.running:
                try:
                    # Get task from queue
                    try:
                        priority, submit_time, task = await asyncio.wait_for(
                            self.task_queue.get(),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                    
                    # Update worker status
                    metrics.status = WorkerStatus.BUSY
                    metrics.current_task_id = task.id
                    metrics.last_activity = datetime.now(timezone.utc)
                    
                    # Process task
                    await self._process_task(task, metrics)
                    
                    # Update worker status
                    metrics.status = WorkerStatus.IDLE
                    metrics.current_task_id = None
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    metrics.status = WorkerStatus.IDLE
                    metrics.current_task_id = None
        
        finally:
            metrics.status = WorkerStatus.STOPPED
            logger.debug(f"Stopped worker: {worker_id}")
    
    async def _process_task(self, task: AsyncTask, worker_metrics: WorkerMetrics):
        """Process a single task."""
        task.started_at = datetime.now(timezone.utc)
        start_time = time.time()
        
        try:
            # Execute task with timeout
            if task.timeout:
                result = await asyncio.wait_for(task.coroutine, timeout=task.timeout)
            else:
                result = await task.coroutine
            
            # Task completed successfully
            task.result = result
            task.completed_at = datetime.now(timezone.utc)
            
            # Update metrics
            processing_time = time.time() - start_time
            worker_metrics.tasks_completed += 1
            worker_metrics.total_processing_time += processing_time
            
            self.metrics['total_tasks_completed'] += 1
            
            # Call callback if provided
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(task.result)
                    else:
                        task.callback(task.result)
                except Exception as e:
                    logger.error(f"Task callback error: {e}")
        
        except Exception as e:
            # Task failed
            task.error = e
            task.completed_at = datetime.now(timezone.utc)
            
            # Check if we should retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
                
                # Re-queue task
                priority_value = -task.priority.value
                await self.task_queue.put((priority_value, time.time(), task))
                return
            
            # Max retries reached
            worker_metrics.tasks_failed += 1
            self.metrics['total_tasks_failed'] += 1
            
            logger.error(f"Task {task.id} failed after {task.max_retries} retries: {e}")
        
        finally:
            # Move task from pending to completed
            if task.id in self.pending_tasks:
                self.pending_tasks.pop(task.id)
            self.completed_tasks.append(task)
            
            # Update queue size metric
            self.metrics['queue_size'] = self.task_queue.qsize()
    
    async def _auto_scaling_loop(self):
        """Auto-scaling loop to adjust worker count based on load."""
        while self.running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                if time.time() - self.last_scale_time < self.scale_cooldown:
                    continue
                
                queue_size = self.task_queue.qsize()
                active_workers = len(self.workers)
                
                # Scale up if queue is getting full
                if (queue_size > active_workers * 2 and 
                    active_workers < self.max_workers):
                    
                    await self._start_worker()
                    self.last_scale_time = time.time()
                    logger.info(f"Scaled up to {len(self.workers)} workers")
                
                # Scale down if queue is mostly empty
                elif (queue_size < active_workers // 2 and 
                      active_workers > self.min_workers):
                    
                    # Find an idle worker to stop
                    idle_worker = None
                    for worker_id, metrics in self.worker_metrics.items():
                        if metrics.status == WorkerStatus.IDLE:
                            idle_worker = worker_id
                            break
                    
                    if idle_worker:
                        await self._stop_worker(idle_worker)
                        self.last_scale_time = time.time()
                        logger.info(f"Scaled down to {len(self.workers)} workers")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")


class AsyncStreamProcessor:
    """
    Stream processor for handling async data streams.
    
    Supports:
    - Backpressure handling
    - Stream transformation
    - Batch processing
    - Error recovery
    """
    
    def __init__(self, buffer_size: int = 1000):
        """Initialize stream processor."""
        self.buffer_size = buffer_size
        self.buffer: deque = deque(maxlen=buffer_size)
        self.processors: List[Callable] = []
        self.running = False
    
    async def process_stream(
        self,
        stream: AsyncIterator[T],
        batch_size: int = 10,
        max_concurrency: int = 5
    ) -> AsyncGenerator[List[R], None]:
        """
        Process async stream with batching and concurrency control.
        
        Args:
            stream: Input async stream
            batch_size: Number of items per batch
            max_concurrency: Maximum concurrent processors
            
        Yields:
            Processed batches
        """
        self.running = True
        semaphore = asyncio.Semaphore(max_concurrency)
        
        batch = []
        
        try:
            async for item in stream:
                if not self.running:
                    break
                
                batch.append(item)
                
                if len(batch) >= batch_size:
                    # Process batch
                    async with semaphore:
                        processed_batch = await self._process_batch(batch)
                        if processed_batch:
                            yield processed_batch
                    
                    batch = []
            
            # Process remaining items
            if batch:
                async with semaphore:
                    processed_batch = await self._process_batch(batch)
                    if processed_batch:
                        yield processed_batch
        
        finally:
            self.running = False
    
    async def _process_batch(self, batch: List[T]) -> List[R]:
        """Process a batch of items."""
        results = []
        
        for item in batch:
            try:
                # Apply all processors
                processed_item = item
                for processor in self.processors:
                    if asyncio.iscoroutinefunction(processor):
                        processed_item = await processor(processed_item)
                    else:
                        processed_item = processor(processed_item)
                
                results.append(processed_item)
                
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                # Skip failed items
                continue
        
        return results
    
    def add_processor(self, processor: Callable):
        """Add a processor to the pipeline."""
        self.processors.append(processor)
    
    def stop(self):
        """Stop stream processing."""
        self.running = False


@asynccontextmanager
async def async_resource_pool(
    resource_factory: Callable,
    max_resources: int = 10,
    min_resources: int = 2
):
    """
    Async context manager for resource pooling.
    
    Args:
        resource_factory: Factory function for creating resources
        max_resources: Maximum number of resources
        min_resources: Minimum number of resources
    """
    pool = asyncio.Queue(maxsize=max_resources)
    created_resources = []
    
    try:
        # Create initial resources
        for _ in range(min_resources):
            if asyncio.iscoroutinefunction(resource_factory):
                resource = await resource_factory()
            else:
                resource = resource_factory()
            
            created_resources.append(resource)
            await pool.put(resource)
        
        yield pool
        
    finally:
        # Cleanup resources
        while not pool.empty():
            try:
                resource = pool.get_nowait()
                if hasattr(resource, 'close'):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
            except asyncio.QueueEmpty:
                break


class AsyncPipeline:
    """
    Async processing pipeline with stages.
    
    Supports:
    - Multi-stage processing
    - Error handling
    - Performance monitoring
    - Backpressure control
    """
    
    def __init__(self, name: str = "pipeline"):
        """Initialize async pipeline."""
        self.name = name
        self.stages: List[Callable] = []
        self.metrics = {
            'items_processed': 0,
            'items_failed': 0,
            'total_processing_time': 0.0,
            'stage_metrics': {}
        }
    
    def add_stage(self, stage: Callable, name: Optional[str] = None):
        """Add a processing stage."""
        stage_name = name or f"stage_{len(self.stages)}"
        self.stages.append(stage)
        self.metrics['stage_metrics'][stage_name] = {
            'items_processed': 0,
            'items_failed': 0,
            'processing_time': 0.0
        }
    
    async def process(self, items: List[T]) -> List[R]:
        """Process items through the pipeline."""
        start_time = time.time()
        results = []
        
        for item in items:
            try:
                # Process through all stages
                processed_item = item
                
                for i, stage in enumerate(self.stages):
                    stage_start = time.time()
                    stage_name = f"stage_{i}"
                    
                    try:
                        if asyncio.iscoroutinefunction(stage):
                            processed_item = await stage(processed_item)
                        else:
                            processed_item = stage(processed_item)
                        
                        # Update stage metrics
                        stage_time = time.time() - stage_start
                        self.metrics['stage_metrics'][stage_name]['items_processed'] += 1
                        self.metrics['stage_metrics'][stage_name]['processing_time'] += stage_time
                        
                    except Exception as e:
                        self.metrics['stage_metrics'][stage_name]['items_failed'] += 1
                        raise e
                
                results.append(processed_item)
                self.metrics['items_processed'] += 1
                
            except Exception as e:
                logger.error(f"Pipeline {self.name} processing error: {e}")
                self.metrics['items_failed'] += 1
                continue
        
        # Update total metrics
        total_time = time.time() - start_time
        self.metrics['total_processing_time'] += total_time
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return self.metrics.copy()


# Global instances
_worker_pool: Optional[AsyncWorkerPool] = None
_stream_processor: Optional[AsyncStreamProcessor] = None

def get_worker_pool() -> AsyncWorkerPool:
    """Get global async worker pool."""
    global _worker_pool
    
    if _worker_pool is None:
        _worker_pool = AsyncWorkerPool()
    
    return _worker_pool

def get_stream_processor() -> AsyncStreamProcessor:
    """Get global stream processor."""
    global _stream_processor
    
    if _stream_processor is None:
        _stream_processor = AsyncStreamProcessor()
    
    return _stream_processor


# Convenience functions
async def run_concurrently(
    tasks: List[Coroutine],
    max_concurrency: int = 10,
    return_exceptions: bool = True
) -> List[Any]:
    """
    Run coroutines concurrently with concurrency limit.
    
    Args:
        tasks: List of coroutines
        max_concurrency: Maximum concurrent tasks
        return_exceptions: Whether to return exceptions or raise them
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def bounded_task(coro):
        async with semaphore:
            return await coro
    
    bounded_tasks = [bounded_task(task) for task in tasks]
    
    return await asyncio.gather(*bounded_tasks, return_exceptions=return_exceptions)

async def stream_process(
    items: List[T],
    processor: Callable[[T], Coroutine[Any, Any, R]],
    batch_size: int = 10,
    max_concurrency: int = 5
) -> AsyncGenerator[R, None]:
    """
    Stream process items with batching and concurrency control.
    
    Args:
        items: Items to process
        processor: Processing function
        batch_size: Items per batch
        max_concurrency: Maximum concurrent processors
        
    Yields:
        Processed items
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        async def process_item(item):
            async with semaphore:
                return await processor(item)
        
        tasks = [process_item(item) for item in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if not isinstance(result, Exception):
                yield result