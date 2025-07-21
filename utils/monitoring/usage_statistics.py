"""
Usage statistics and analytics system for RAG application.
Tracks user behavior, API usage patterns, storage growth, and system utilization.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
from pathlib import Path

from utils.caching.redis_manager import get_redis_manager
from database.models import User, Document, SearchQuery

logger = logging.getLogger(__name__)

class UsageEventType(Enum):
    """Types of usage events to track."""
    API_REQUEST = "api_request"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_SEARCH = "document_search"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    STORAGE_CHANGE = "storage_change"
    VECTOR_OPERATION = "vector_operation"
    LLM_REQUEST = "llm_request"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    ERROR_OCCURRED = "error_occurred"

class TimePeriod(Enum):
    """Time periods for analytics."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"

@dataclass
class UsageEvent:
    """Individual usage event."""
    event_type: UsageEventType
    user_id: Optional[int] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    response_time_ms: Optional[float] = None
    status_code: Optional[int] = None
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None
    storage_delta_bytes: Optional[int] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class UsageStats:
    """Aggregated usage statistics."""
    total_requests: int = 0
    unique_users: int = 0
    total_storage_bytes: int = 0
    average_response_time_ms: float = 0.0
    error_rate: float = 0.0
    top_endpoints: List[Tuple[str, int]] = field(default_factory=list)
    user_activity: Dict[int, int] = field(default_factory=dict)
    hourly_distribution: Dict[int, int] = field(default_factory=dict)
    daily_trends: Dict[str, int] = field(default_factory=dict)

class UsageStatisticsCollector:
    """
    Comprehensive usage statistics and analytics collector.
    
    Features:
    - Real-time usage event tracking
    - API endpoint usage analytics
    - User behavior analysis
    - Storage growth monitoring
    - Performance trend analysis
    - Cache efficiency tracking
    """
    
    def __init__(
        self,
        max_events_in_memory: int = 10000,
        flush_interval_seconds: int = 300,
        enable_redis_persistence: bool = True
    ):
        """
        Initialize usage statistics collector.
        
        Args:
            max_events_in_memory: Maximum events to keep in memory
            flush_interval_seconds: How often to flush to persistent storage
            enable_redis_persistence: Whether to persist to Redis
        """
        self.max_events_in_memory = max_events_in_memory
        self.flush_interval_seconds = flush_interval_seconds
        self.enable_redis_persistence = enable_redis_persistence
        
        # In-memory event storage
        self.events: deque = deque(maxlen=max_events_in_memory)
        
        # Aggregated statistics
        self.stats_cache: Dict[str, UsageStats] = {}
        self.last_stats_update = datetime.now(timezone.utc)
        
        # Real-time counters
        self.realtime_counters = {
            'active_users': set(),
            'requests_per_minute': deque(maxlen=60),
            'errors_per_minute': deque(maxlen=60),
            'total_requests_today': 0,
            'total_errors_today': 0,
            'cache_hits_today': 0,
            'cache_misses_today': 0
        }
        
        # Redis manager for persistence
        self.redis_manager = get_redis_manager() if enable_redis_persistence else None
        
        # Background tasks
        self.flush_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self):
        """Start the usage statistics collector."""
        if self.running:
            return
        
        self.running = True
        
        # Start background flush task
        if self.enable_redis_persistence:
            self.flush_task = asyncio.create_task(self._flush_loop())
        
        logger.info("Usage statistics collector started")
    
    async def stop(self):
        """Stop the usage statistics collector."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop background tasks
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        if self.enable_redis_persistence:
            await self._flush_events()
        
        logger.info("Usage statistics collector stopped")
    
    async def track_api_request(
        self,
        endpoint: str,
        method: str,
        user_id: Optional[int] = None,
        response_time_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        request_size_bytes: Optional[int] = None,
        response_size_bytes: Optional[int] = None,
        error_type: Optional[str] = None
    ):
        """Track API request usage."""
        event = UsageEvent(
            event_type=UsageEventType.API_REQUEST,
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            response_time_ms=response_time_ms,
            status_code=status_code,
            request_size_bytes=request_size_bytes,
            response_size_bytes=response_size_bytes,
            error_type=error_type
        )
        
        await self._record_event(event)
        
        # Update real-time counters
        self.realtime_counters['total_requests_today'] += 1
        
        if user_id:
            self.realtime_counters['active_users'].add(user_id)
        
        if status_code and status_code >= 400:
            self.realtime_counters['total_errors_today'] += 1
    
    async def track_document_upload(
        self,
        user_id: int,
        filename: str,
        file_size_bytes: int,
        processing_time_ms: float,
        success: bool = True
    ):
        """Track document upload activity."""
        event = UsageEvent(
            event_type=UsageEventType.DOCUMENT_UPLOAD,
            user_id=user_id,
            response_time_ms=processing_time_ms,
            storage_delta_bytes=file_size_bytes if success else 0,
            metadata={
                'filename': filename,
                'file_size_bytes': file_size_bytes,
                'success': success
            }
        )
        
        await self._record_event(event)
    
    async def track_search_query(
        self,
        user_id: int,
        query: str,
        results_count: int,
        response_time_ms: float,
        query_type: str = "similarity"
    ):
        """Track search query activity."""
        event = UsageEvent(
            event_type=UsageEventType.DOCUMENT_SEARCH,
            user_id=user_id,
            response_time_ms=response_time_ms,
            metadata={
                'query': query[:100],  # Truncate for privacy
                'results_count': results_count,
                'query_type': query_type,
                'query_length': len(query)
            }
        )
        
        await self._record_event(event)
    
    async def track_user_session(
        self,
        user_id: int,
        action: str,  # "login" or "logout"
        session_duration_ms: Optional[float] = None
    ):
        """Track user session activity."""
        event_type = UsageEventType.USER_LOGIN if action == "login" else UsageEventType.USER_LOGOUT
        
        event = UsageEvent(
            event_type=event_type,
            user_id=user_id,
            metadata={
                'action': action,
                'session_duration_ms': session_duration_ms
            }
        )
        
        await self._record_event(event)
        
        # Update active users
        if action == "login":
            self.realtime_counters['active_users'].add(user_id)
        elif action == "logout":
            self.realtime_counters['active_users'].discard(user_id)
    
    async def track_storage_change(
        self,
        user_id: int,
        storage_delta_bytes: int,
        operation: str,  # "add", "delete", "update"
        content_type: Optional[str] = None
    ):
        """Track storage usage changes."""
        event = UsageEvent(
            event_type=UsageEventType.STORAGE_CHANGE,
            user_id=user_id,
            storage_delta_bytes=storage_delta_bytes,
            metadata={
                'operation': operation,
                'content_type': content_type
            }
        )
        
        await self._record_event(event)
    
    async def track_vector_operation(
        self,
        user_id: Optional[int],
        operation: str,  # "index", "search", "update", "delete"
        vector_count: int,
        processing_time_ms: float,
        index_name: Optional[str] = None
    ):
        """Track vector database operations."""
        event = UsageEvent(
            event_type=UsageEventType.VECTOR_OPERATION,
            user_id=user_id,
            response_time_ms=processing_time_ms,
            metadata={
                'operation': operation,
                'vector_count': vector_count,
                'index_name': index_name
            }
        )
        
        await self._record_event(event)
    
    async def track_llm_request(
        self,
        user_id: Optional[int],
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        response_time_ms: float,
        success: bool = True
    ):
        """Track LLM API requests."""
        event = UsageEvent(
            event_type=UsageEventType.LLM_REQUEST,
            user_id=user_id,
            response_time_ms=response_time_ms,
            metadata={
                'model_name': model_name,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'success': success
            }
        )
        
        await self._record_event(event)
    
    async def track_cache_operation(
        self,
        user_id: Optional[int],
        operation: str,  # "hit" or "miss"
        cache_type: str,  # "query", "result", "session"
        response_time_ms: Optional[float] = None
    ):
        """Track cache hit/miss statistics."""
        event_type = UsageEventType.CACHE_HIT if operation == "hit" else UsageEventType.CACHE_MISS
        
        event = UsageEvent(
            event_type=event_type,
            user_id=user_id,
            response_time_ms=response_time_ms,
            metadata={
                'cache_type': cache_type,
                'operation': operation
            }
        )
        
        await self._record_event(event)
        
        # Update daily counters
        if operation == "hit":
            self.realtime_counters['cache_hits_today'] += 1
        else:
            self.realtime_counters['cache_misses_today'] += 1
    
    async def get_usage_statistics(
        self,
        time_period: TimePeriod = TimePeriod.DAY,
        user_id: Optional[int] = None,
        refresh_cache: bool = False
    ) -> UsageStats:
        """
        Get aggregated usage statistics for a time period.
        
        Args:
            time_period: Time period for statistics
            user_id: Filter by specific user
            refresh_cache: Force refresh of cached statistics
            
        Returns:
            Aggregated usage statistics
        """
        cache_key = f"{time_period.value}_{user_id or 'all'}"
        
        # Check cache first
        if not refresh_cache and cache_key in self.stats_cache:
            cached_stats = self.stats_cache[cache_key]
            cache_age = datetime.now(timezone.utc) - self.last_stats_update
            
            # Use cached stats if less than 5 minutes old
            if cache_age.total_seconds() < 300:
                return cached_stats
        
        # Calculate statistics
        stats = await self._calculate_usage_statistics(time_period, user_id)
        
        # Cache the result
        self.stats_cache[cache_key] = stats
        self.last_stats_update = datetime.now(timezone.utc)
        
        return stats
    
    async def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time usage metrics."""
        now = datetime.now(timezone.utc)
        
        # Calculate cache hit rate
        total_cache_ops = (
            self.realtime_counters['cache_hits_today'] + 
            self.realtime_counters['cache_misses_today']
        )
        cache_hit_rate = (
            (self.realtime_counters['cache_hits_today'] / total_cache_ops * 100)
            if total_cache_ops > 0 else 0.0
        )
        
        # Calculate error rate
        total_requests = self.realtime_counters['total_requests_today']
        error_rate = (
            (self.realtime_counters['total_errors_today'] / total_requests * 100)
            if total_requests > 0 else 0.0
        )
        
        return {
            'timestamp': now.isoformat(),
            'active_users_count': len(self.realtime_counters['active_users']),
            'active_user_ids': list(self.realtime_counters['active_users']),
            'total_requests_today': total_requests,
            'total_errors_today': self.realtime_counters['total_errors_today'],
            'error_rate_percent': error_rate,
            'cache_hits_today': self.realtime_counters['cache_hits_today'],
            'cache_misses_today': self.realtime_counters['cache_misses_today'],
            'cache_hit_rate_percent': cache_hit_rate,
            'events_in_memory': len(self.events),
            'memory_usage_percent': (len(self.events) / self.max_events_in_memory) * 100
        }
    
    async def get_user_analytics(
        self,
        user_id: int,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get detailed analytics for a specific user."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Filter events for the user
        user_events = [
            event for event in self.events
            if event.user_id == user_id and event.timestamp >= cutoff_date
        ]
        
        if not user_events:
            return {
                'user_id': user_id,
                'total_events': 0,
                'analysis_period_days': days,
                'message': 'No activity found for this user'
            }
        
        # Analyze user activity
        activity_by_type = defaultdict(int)
        total_response_time = 0.0
        response_time_count = 0
        storage_used = 0
        document_uploads = 0
        search_queries = 0
        login_sessions = 0
        
        daily_activity = defaultdict(int)
        hourly_distribution = defaultdict(int)
        
        for event in user_events:
            activity_by_type[event.event_type.value] += 1
            
            # Response time analysis
            if event.response_time_ms:
                total_response_time += event.response_time_ms
                response_time_count += 1
            
            # Storage analysis
            if event.storage_delta_bytes:
                storage_used += event.storage_delta_bytes
            
            # Activity type counting
            if event.event_type == UsageEventType.DOCUMENT_UPLOAD:
                document_uploads += 1
            elif event.event_type == UsageEventType.DOCUMENT_SEARCH:
                search_queries += 1
            elif event.event_type == UsageEventType.USER_LOGIN:
                login_sessions += 1
            
            # Time distribution analysis
            day_key = event.timestamp.strftime('%Y-%m-%d')
            daily_activity[day_key] += 1
            
            hour = event.timestamp.hour
            hourly_distribution[hour] += 1
        
        # Calculate averages
        avg_response_time = (
            total_response_time / response_time_count
            if response_time_count > 0 else 0.0
        )
        
        return {
            'user_id': user_id,
            'analysis_period_days': days,
            'total_events': len(user_events),
            'activity_summary': {
                'document_uploads': document_uploads,
                'search_queries': search_queries,
                'login_sessions': login_sessions,
                'storage_used_bytes': storage_used,
                'average_response_time_ms': avg_response_time
            },
            'activity_by_type': dict(activity_by_type),
            'daily_activity': dict(daily_activity),
            'hourly_distribution': dict(hourly_distribution),
            'most_active_hour': max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None,
            'most_active_day': max(daily_activity.items(), key=lambda x: x[1])[0] if daily_activity else None
        }
    
    async def generate_usage_report(
        self,
        time_period: TimePeriod = TimePeriod.WEEK,
        include_trends: bool = True,
        include_user_breakdown: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive usage report."""
        stats = await self.get_usage_statistics(time_period)
        realtime_metrics = await self.get_realtime_metrics()
        
        report = {
            'report_generated_at': datetime.now(timezone.utc).isoformat(),
            'time_period': time_period.value,
            'summary': {
                'total_requests': stats.total_requests,
                'unique_users': stats.unique_users,
                'total_storage_bytes': stats.total_storage_bytes,
                'average_response_time_ms': stats.average_response_time_ms,
                'error_rate_percent': stats.error_rate
            },
            'top_endpoints': stats.top_endpoints[:10],
            'realtime_metrics': realtime_metrics
        }
        
        if include_trends:
            report['trends'] = {
                'daily_activity': stats.daily_trends,
                'hourly_distribution': stats.hourly_distribution
            }
        
        if include_user_breakdown:
            # Get top 10 most active users
            top_users = sorted(
                stats.user_activity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            report['top_users'] = [
                {'user_id': user_id, 'request_count': count}
                for user_id, count in top_users
            ]
        
        return report
    
    async def _record_event(self, event: UsageEvent):
        """Record a usage event."""
        self.events.append(event)
        
        # Immediate Redis persistence for high-priority events
        if (self.enable_redis_persistence and 
            event.event_type in [UsageEventType.USER_LOGIN, UsageEventType.ERROR_OCCURRED]):
            await self._persist_event(event)
    
    async def _calculate_usage_statistics(
        self,
        time_period: TimePeriod,
        user_id: Optional[int] = None
    ) -> UsageStats:
        """Calculate aggregated usage statistics."""
        # Determine time cutoff
        now = datetime.now(timezone.utc)
        if time_period == TimePeriod.HOUR:
            cutoff = now - timedelta(hours=1)
        elif time_period == TimePeriod.DAY:
            cutoff = now - timedelta(days=1)
        elif time_period == TimePeriod.WEEK:
            cutoff = now - timedelta(weeks=1)
        elif time_period == TimePeriod.MONTH:
            cutoff = now - timedelta(days=30)
        elif time_period == TimePeriod.YEAR:
            cutoff = now - timedelta(days=365)
        else:
            cutoff = now - timedelta(days=1)
        
        # Filter events
        filtered_events = [
            event for event in self.events
            if event.timestamp >= cutoff and (user_id is None or event.user_id == user_id)
        ]
        
        if not filtered_events:
            return UsageStats()
        
        # Calculate statistics
        stats = UsageStats()
        
        # Basic counts
        stats.total_requests = len([e for e in filtered_events if e.event_type == UsageEventType.API_REQUEST])
        stats.unique_users = len(set(e.user_id for e in filtered_events if e.user_id))
        
        # Storage calculation
        storage_events = [e for e in filtered_events if e.storage_delta_bytes]
        stats.total_storage_bytes = sum(e.storage_delta_bytes for e in storage_events)
        
        # Response time calculation
        response_time_events = [e for e in filtered_events if e.response_time_ms]
        if response_time_events:
            stats.average_response_time_ms = sum(e.response_time_ms for e in response_time_events) / len(response_time_events)
        
        # Error rate calculation
        api_events = [e for e in filtered_events if e.event_type == UsageEventType.API_REQUEST]
        error_events = [e for e in api_events if e.status_code and e.status_code >= 400]
        if api_events:
            stats.error_rate = (len(error_events) / len(api_events)) * 100
        
        # Top endpoints
        endpoint_counts = defaultdict(int)
        for event in api_events:
            if event.endpoint:
                endpoint_counts[event.endpoint] += 1
        
        stats.top_endpoints = sorted(
            endpoint_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # User activity
        user_counts = defaultdict(int)
        for event in filtered_events:
            if event.user_id:
                user_counts[event.user_id] += 1
        stats.user_activity = dict(user_counts)
        
        # Time distribution
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        
        for event in filtered_events:
            hour = event.timestamp.hour
            day = event.timestamp.strftime('%Y-%m-%d')
            
            hourly_counts[hour] += 1
            daily_counts[day] += 1
        
        stats.hourly_distribution = dict(hourly_counts)
        stats.daily_trends = dict(daily_counts)
        
        return stats
    
    async def _persist_event(self, event: UsageEvent):
        """Persist single event to Redis."""
        if not self.redis_manager:
            return
        
        try:
            key = f"usage_event:{event.timestamp.strftime('%Y%m%d%H')}:{event.event_type.value}"
            event_data = {
                'event_type': event.event_type.value,
                'user_id': event.user_id,
                'endpoint': event.endpoint,
                'method': event.method,
                'response_time_ms': event.response_time_ms,
                'status_code': event.status_code,
                'request_size_bytes': event.request_size_bytes,
                'response_size_bytes': event.response_size_bytes,
                'storage_delta_bytes': event.storage_delta_bytes,
                'error_type': event.error_type,
                'metadata': event.metadata,
                'timestamp': event.timestamp.isoformat()
            }
            
            await self.redis_manager.lpush(key, json.dumps(event_data))
            await self.redis_manager.expire(key, 7 * 24 * 3600)  # 7 days retention
            
        except Exception as e:
            logger.error(f"Failed to persist usage event: {e}")
    
    async def _flush_events(self):
        """Flush all in-memory events to Redis."""
        if not self.redis_manager or not self.events:
            return
        
        try:
            # Group events by hour for efficient storage
            events_by_hour = defaultdict(list)
            
            for event in list(self.events):
                hour_key = event.timestamp.strftime('%Y%m%d%H')
                events_by_hour[hour_key].append(event)
            
            # Persist each hour's events
            for hour_key, hour_events in events_by_hour.items():
                redis_key = f"usage_events_batch:{hour_key}"
                
                batch_data = []
                for event in hour_events:
                    event_data = {
                        'event_type': event.event_type.value,
                        'user_id': event.user_id,
                        'endpoint': event.endpoint,
                        'method': event.method,
                        'response_time_ms': event.response_time_ms,
                        'status_code': event.status_code,
                        'request_size_bytes': event.request_size_bytes,
                        'response_size_bytes': event.response_size_bytes,
                        'storage_delta_bytes': event.storage_delta_bytes,
                        'error_type': event.error_type,
                        'metadata': event.metadata,
                        'timestamp': event.timestamp.isoformat()
                    }
                    batch_data.append(event_data)
                
                await self.redis_manager.set_json(redis_key, batch_data, 7 * 24 * 3600)
            
            logger.info(f"Flushed {len(self.events)} usage events to Redis")
            
        except Exception as e:
            logger.error(f"Failed to flush usage events: {e}")
    
    async def _flush_loop(self):
        """Background task to periodically flush events."""
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval_seconds)
                await self._flush_events()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")


# Global usage statistics collector instance
_usage_stats_collector: Optional[UsageStatisticsCollector] = None

def get_usage_stats_collector() -> UsageStatisticsCollector:
    """Get global usage statistics collector instance."""
    global _usage_stats_collector
    
    if _usage_stats_collector is None:
        _usage_stats_collector = UsageStatisticsCollector()
    
    return _usage_stats_collector

# Convenience tracking functions
async def track_api_request(
    endpoint: str,
    method: str,
    user_id: Optional[int] = None,
    response_time_ms: Optional[float] = None,
    status_code: Optional[int] = None,
    request_size: Optional[int] = None,
    response_size: Optional[int] = None,
    error_type: Optional[str] = None
):
    """Convenience function to track API requests."""
    collector = get_usage_stats_collector()
    await collector.track_api_request(
        endpoint, method, user_id, response_time_ms,
        status_code, request_size, response_size, error_type
    )

async def track_document_upload(
    user_id: int,
    filename: str,
    file_size_bytes: int,
    processing_time_ms: float,
    success: bool = True
):
    """Convenience function to track document uploads."""
    collector = get_usage_stats_collector()
    await collector.track_document_upload(
        user_id, filename, file_size_bytes, processing_time_ms, success
    )

async def track_search_query(
    user_id: int,
    query: str,
    results_count: int,
    response_time_ms: float,
    query_type: str = "similarity"
):
    """Convenience function to track search queries."""
    collector = get_usage_stats_collector()
    await collector.track_search_query(
        user_id, query, results_count, response_time_ms, query_type
    )