import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
import json
from pathlib import Path
import uuid
from dataclasses import dataclass, asdict
import queue
from threading import Thread, Event
import time
from contextlib import contextmanager
import shutil
import os

@dataclass
class AuditEvent:
    timestamp: str
    event_type: str
    user_id: str
    action: str
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    status: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class AuditLoggerConfig:
    log_path: Path
    rotation_size_mb: int = 10
    retention_days: int = 90
    batch_size: int = 100
    flush_interval_seconds: int = 5
    enable_async: bool = True

class AuditLogger:
    """
    Handles audit logging with support for asynchronous writes, log rotation, and retention policies.
    Thread-safe implementation for concurrent logging in production environments.
    """
    
    def __init__(self, config: AuditLoggerConfig):
        self.config = config
        self.event_queue = queue.Queue()
        self._stop_event = Event()
        self._current_log_path = self._initialize_log_file()
        
        # Initialize logging thread if async is enabled
        self._worker_thread = None
        if self.config.enable_async:
            self._worker_thread = Thread(target=self._process_queue, daemon=True)
            self._worker_thread.start()

    def _initialize_log_file(self) -> Path:
        """Initialize the log file with proper directory structure."""
        self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.config.log_path.parent / f"audit_{timestamp}.log"
        return log_file

    def _rotate_log_if_needed(self) -> None:
        """Rotate log file if it exceeds the configured size."""
        if self._current_log_path.exists():
            size_mb = self._current_log_path.stat().st_size / (1024 * 1024)
            if size_mb >= self.config.rotation_size_mb:
                new_log_path = self._initialize_log_file()
                self._current_log_path = new_log_path

    def _clean_old_logs(self) -> None:
        """Remove log files older than the retention period."""
        retention_date = datetime.now() - timedelta(days=self.config.retention_days)
        log_dir = self.config.log_path.parent
        
        for log_file in log_dir.glob("audit_*.log"):
            try:
                # Extract timestamp from filename
                timestamp_str = log_file.stem.split('_')[1]
                file_date = datetime.strptime(timestamp_str, "%Y%m%d")
                
                if file_date < retention_date:
                    log_file.unlink()
                    logging.info(f"Removed old audit log: {log_file}")
            except (ValueError, IndexError):
                logging.warning(f"Could not parse date from log filename: {log_file}")

    def _process_queue(self) -> None:
        """Process events from the queue in batches."""
        batch = []
        last_flush_time = time.time()

        while not self._stop_event.is_set():
            try:
                # Try to get an event from the queue
                try:
                    event = self.event_queue.get(timeout=1.0)
                    batch.append(event)
                except queue.Empty:
                    event = None

                current_time = time.time()
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (batch and current_time - last_flush_time >= self.config.flush_interval_seconds)
                )

                if should_flush:
                    self._write_batch(batch)
                    batch = []
                    last_flush_time = current_time

            except Exception as e:
                logging.error(f"Error in audit log processing thread: {str(e)}")

    def _write_batch(self, batch: list) -> None:
        """Write a batch of events to the log file."""
        if not batch:
            return

        try:
            self._rotate_log_if_needed()
            
            with open(self._current_log_path, 'a') as f:
                for event in batch:
                    json_line = json.dumps(asdict(event))
                    f.write(json_line + '\n')
                    
        except Exception as e:
            logging.error(f"Failed to write audit log batch: {str(e)}")

    def log_event(self, 
                  event_type: str,
                  user_id: str,
                  action: str,
                  resource_id: Optional[str] = None,
                  resource_type: Optional[str] = None,
                  status: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  ip_address: Optional[str] = None,
                  session_id: Optional[str] = None) -> None:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event (e.g., 'access', 'modification', 'deletion')
            user_id: ID of the user performing the action
            action: Specific action taken
            resource_id: Optional ID of the resource being acted upon
            resource_type: Optional type of the resource
            status: Optional status of the action (e.g., 'success', 'failure')
            details: Optional additional details about the event
            ip_address: Optional IP address of the user
            session_id: Optional session identifier
        """
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            user_id=user_id,
            action=action,
            resource_id=resource_id,
            resource_type=resource_type,
            status=status,
            details=details,
            ip_address=ip_address,
            session_id=session_id
        )

        if self.config.enable_async:
            self.event_queue.put(event)
        else:
            self._write_batch([event])

    def query_logs(self, 
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  event_type: Optional[str] = None,
                  user_id: Optional[str] = None,
                  action: Optional[str] = None) -> list:
        """
        Query audit logs with optional filters.
        
        Args:
            start_time: Optional start time for the query
            end_time: Optional end time for the query
            event_type: Optional event type filter
            user_id: Optional user ID filter
            action: Optional action filter
            
        Returns:
            list: Matching audit events
        """
        results = []
        log_dir = self.config.log_path.parent

        for log_file in log_dir.glob("audit_*.log"):
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        event_dict = json.loads(line.strip())
                        event_time = datetime.fromisoformat(event_dict['timestamp'])
                        
                        # Apply filters
                        if start_time and event_time < start_time:
                            continue
                        if end_time and event_time > end_time:
                            continue
                        if event_type and event_dict['event_type'] != event_type:
                            continue
                        if user_id and event_dict['user_id'] != user_id:
                            continue
                        if action and event_dict['action'] != action:
                            continue
                            
                        results.append(event_dict)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.error(f"Error parsing audit log entry: {str(e)}")

        return results

    @contextmanager
    def audit_context(self, 
                     event_type: str,
                     user_id: str,
                     action: str,
                     **kwargs) -> None:
        """
        Context manager for automatically logging the start and end of an operation.
        
        Args:
            event_type: Type of event
            user_id: ID of the user
            action: Action being performed
            **kwargs: Additional arguments to pass to log_event
        """
        start_time = datetime.utcnow()
        
        try:
            self.log_event(
                event_type=event_type,
                user_id=user_id,
                action=f"{action}_started",
                details={"start_time": start_time.isoformat()},
                **kwargs
            )
            yield
            
        except Exception as e:
            self.log_event(
                event_type=event_type,
                user_id=user_id,
                action=f"{action}_failed",
                status="error",
                details={
                    "error": str(e),
                    "start_time": start_time.isoformat(),
                    "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
                },
                **kwargs
            )
            raise
            
        else:
            self.log_event(
                event_type=event_type,
                user_id=user_id,
                action=f"{action}_completed",
                status="success",
                details={
                    "start_time": start_time.isoformat(),
                    "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
                },
                **kwargs
            )

    def shutdown(self) -> None:
        """Gracefully shut down the audit logger."""
        if self.config.enable_async:
            self._stop_event.set()
            if self._worker_thread:
                self._worker_thread.join(timeout=10.0)
            
            # Process any remaining events in the queue
            remaining_events = []
            while not self.event_queue.empty():
                remaining_events.append(self.event_queue.get_nowait())
            
            if remaining_events:
                self._write_batch(remaining_events)

        self._clean_old_logs()