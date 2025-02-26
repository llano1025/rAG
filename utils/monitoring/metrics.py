from typing import Dict, Any, Optional
import time
from dataclasses import dataclass, field
from datetime import datetime
import threading
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    def __init__(self):
        self._metrics = defaultdict(list)
        self._lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value with optional labels."""
        with self._lock:
            self._metrics[name].append(MetricPoint(
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            ))
            
    def get_metric(self, name: str, lookback_seconds: Optional[float] = None) -> list[MetricPoint]:
        """Get metrics for a given name within the lookback window."""
        with self._lock:
            if name not in self._metrics:
                return []
                
            if lookback_seconds is None:
                return self._metrics[name]
                
            cutoff = time.time() - lookback_seconds
            return [m for m in self._metrics[name] if m.timestamp >= cutoff]
            
    def calculate_statistics(self, name: str, lookback_seconds: Optional[float] = None) -> Dict[str, float]:
        """Calculate statistics for a given metric."""
        metrics = self.get_metric(name, lookback_seconds)
        if not metrics:
            return {}
            
        values = [m.value for m in metrics]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values)
        }

class SystemMetrics:
    def __init__(self):
        self.collector = MetricsCollector()
        
    def record_request_duration(self, endpoint: str, duration: float):
        """Record API request duration."""
        self.collector.record_metric(
            'request_duration_seconds',
            duration,
            {'endpoint': endpoint}
        )
        
    def record_document_processing_time(self, doc_type: str, size_bytes: int, duration: float):
        """Record document processing duration."""
        self.collector.record_metric(
            'document_processing_seconds',
            duration,
            {
                'document_type': doc_type,
                'size_range': self._get_size_bucket(size_bytes)
            }
        )
        
    def record_vector_db_operation(self, operation: str, duration: float):
        """Record vector database operation duration."""
        self.collector.record_metric(
            'vector_db_operation_seconds',
            duration,
            {'operation': operation}
        )
        
    def record_memory_usage(self, component: str, bytes_used: int):
        """Record memory usage for a component."""
        self.collector.record_metric(
            'memory_usage_bytes',
            bytes_used,
            {'component': component}
        )
        
    def record_error_count(self, error_type: str):
        """Record error occurrence."""
        self.collector.record_metric(
            'error_count',
            1.0,
            {'error_type': error_type}
        )
        
    @staticmethod
    def _get_size_bucket(size_bytes: int) -> str:
        """Categorize file size into buckets."""
        if size_bytes < 1024:
            return '<1KB'
        elif size_bytes < 1024 * 1024:
            return '1KB-1MB'
        elif size_bytes < 1024 * 1024 * 10:
            return '1MB-10MB'
        else:
            return '>10MB'

# Global metrics instance
system_metrics = SystemMetrics()

# Example usage:
if __name__ == "__main__":
    # Record some example metrics
    metrics = SystemMetrics()
    
    # Record API request
    metrics.record_request_duration('/api/documents', 0.15)
    
    # Record document processing
    metrics.record_document_processing_time('pdf', 1024 * 1024, 2.5)
    
    # Get statistics
    stats = metrics.collector.calculate_statistics('request_duration_seconds', 3600)
    print(f"Request duration statistics: {stats}")