from typing import Dict, Any, Optional, List
import logging
import traceback
from datetime import datetime
import threading
from collections import deque
import json
import hashlib
from dataclasses import dataclass, field
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

@dataclass
class ErrorEvent:
    error_type: str
    message: str
    stack_trace: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    error_hash: str = field(init=False)

    def __post_init__(self):
        # Generate a unique hash for the error based on type and stack trace
        self.error_hash = hashlib.md5(
            f"{self.error_type}:{self.stack_trace}".encode()
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error event to dictionary format."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "error_hash": self.error_hash
        }

class ErrorAggregation:
    def __init__(self, error_hash: str, error_type: str):
        self.error_hash = error_hash
        self.error_type = error_type
        self.count = 0
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.example_events: deque = deque(maxlen=5)  # Keep last 5 examples

    def update(self, event: ErrorEvent):
        """Update aggregation with new error event."""
        self.count += 1
        self.last_seen = event.timestamp
        self.example_events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregation to dictionary format."""
        return {
            "error_hash": self.error_hash,
            "error_type": self.error_type,
            "count": self.count,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "example_events": [e.to_dict() for e in self.example_events]
        }

class ErrorTracker:
    def __init__(self, max_events: int = 1000):
        self._events: deque = deque(maxlen=max_events)
        self._aggregations: Dict[str, ErrorAggregation] = {}
        self._lock = threading.Lock()
        self._error_handlers: List[callable] = []

    def register_error_handler(self, handler: callable):
        """Register a callback function to handle errors."""
        self._error_handlers.append(handler)

    def track_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorEvent:
        """Track a new error with optional context."""
        event = ErrorEvent(
            error_type=type(error).__name__,
            message=str(error),
            stack_trace=''.join(traceback.format_tb(error.__traceback__)),
            context=context or {}
        )

        with self._lock:
            self._events.append(event)
            
            # Update aggregations
            if event.error_hash not in self._aggregations:
                self._aggregations[event.error_hash] = ErrorAggregation(
                    event.error_hash,
                    event.error_type
                )
            self._aggregations[event.error_hash].update(event)

        # Notify error handlers
        self._notify_handlers(event)
        
        return event

    def _notify_handlers(self, event: ErrorEvent):
        """Notify all registered error handlers."""
        for handler in self._error_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error handler failed: {str(e)}")

    def get_recent_errors(self, limit: int = 50) -> List[ErrorEvent]:
        """Get most recent error events."""
        with self._lock:
            return list(self._events)[-limit:]

    def get_error_aggregations(self) -> Dict[str, Dict[str, Any]]:
        """Get error aggregations."""
        with self._lock:
            return {
                error_hash: agg.to_dict()
                for error_hash, agg in self._aggregations.items()
            }

    def get_error_frequency(self, error_hash: str, timeframe_minutes: int = 60) -> int:
        """Get frequency of specific error in recent timeframe."""
        cutoff = datetime.now().timestamp() - (timeframe_minutes * 60)
        count = 0
        
        with self._lock:
            for event in self._events:
                if event.error_hash == error_hash and event.timestamp.timestamp() > cutoff:
                    count += 1
        
        return count

    @contextmanager
    def catch_and_track(self, context: Optional[Dict[str, Any]] = None):
        """Context manager to catch and track errors."""
        try:
            yield
        except Exception as e:
            self.track_error(e, context)
            raise

    def export_error_report(self, filepath: str):
        """Export error tracking data to JSON file."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "recent_errors": [e.to_dict() for e in self.get_recent_errors()],
            "error_aggregations": self.get_error_aggregations()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

# Example usage:
if __name__ == "__main__":
    # Initialize error tracker
    tracker = ErrorTracker()

    # Add example error handler
    def log_error_to_console(event: ErrorEvent):
        print(f"Error detected: {event.error_type}")
        print(f"Message: {event.message}")
        print(f"Stack trace:\n{event.stack_trace}")
        print(f"Context: {event.context}")
        print("-" * 80)

    tracker.register_error_handler(log_error_to_console)

    # Example usage with context manager
    try:
        with tracker.catch_and_track({"operation": "example_operation"}):
            # Simulate some error
            raise ValueError("This is an example error")
    except Exception:
        pass

    # Get error statistics
    print("\nError Aggregations:")
    aggregations = tracker.get_error_aggregations()
    for agg in aggregations.values():
        print(f"Error type: {agg['error_type']}")
        print(f"Count: {agg['count']}")
        print(f"First seen: {agg['first_seen']}")
        print(f"Last seen: {agg['last_seen']}")
        print("-" * 80)

    # Export error report
    tracker.export_error_report("error_report.json")