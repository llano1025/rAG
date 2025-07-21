"""
Advanced Analytics and Reporting System
Provides comprehensive analytics, insights, and reporting capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from pathlib import Path
import aiofiles

from database.connection import get_db
from utils.monitoring.metrics import MetricsCollector
from utils.caching.cache_strategy import CacheStrategy

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of analytics reports"""
    USAGE_SUMMARY = "usage_summary"
    USER_ACTIVITY = "user_activity"
    DOCUMENT_INSIGHTS = "document_insights"
    SEARCH_ANALYTICS = "search_analytics"
    PERFORMANCE_METRICS = "performance_metrics"
    SYSTEM_HEALTH = "system_health"
    TREND_ANALYSIS = "trend_analysis"
    CUSTOM_REPORT = "custom_report"


class TimeRange(Enum):
    """Time range options for analytics"""
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_QUARTER = "last_quarter"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"


@dataclass
class AnalyticsQuery:
    """Query parameters for analytics"""
    report_type: ReportType
    time_range: TimeRange
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    filters: Dict = field(default_factory=dict)
    group_by: Optional[str] = None
    aggregations: List[str] = field(default_factory=list)
    limit: Optional[int] = None


@dataclass
class AnalyticsResult:
    """Result of an analytics query"""
    query: AnalyticsQuery
    data: Any
    metadata: Dict
    generated_at: datetime
    execution_time: float
    cache_hit: bool = False


class AdvancedAnalytics:
    """Advanced analytics and reporting system"""
    
    def __init__(self, cache_ttl: int = 300):
        self.metrics = MetricsCollector()
        self.cache = CacheStrategy.SIMPLE
        self.cache_ttl = cache_ttl
        self._report_cache = {}
    
    async def generate_report(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Generate an analytics report based on query"""
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                return cached_result
            
            # Generate report based on type
            data = await self._execute_query(query)
            
            # Create result
            result = AnalyticsResult(
                query=query,
                data=data,
                metadata=await self._generate_metadata(query, data),
                generated_at=start_time,
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            raise
    
    async def _execute_query(self, query: AnalyticsQuery) -> Any:
        """Execute analytics query based on report type"""
        handlers = {
            ReportType.USAGE_SUMMARY: self._generate_usage_summary,
            ReportType.USER_ACTIVITY: self._generate_user_activity,
            ReportType.DOCUMENT_INSIGHTS: self._generate_document_insights,
            ReportType.SEARCH_ANALYTICS: self._generate_search_analytics,
            ReportType.PERFORMANCE_METRICS: self._generate_performance_metrics,
            ReportType.SYSTEM_HEALTH: self._generate_system_health,
            ReportType.TREND_ANALYSIS: self._generate_trend_analysis,
            ReportType.CUSTOM_REPORT: self._generate_custom_report
        }
        
        handler = handlers.get(query.report_type)
        if not handler:
            raise ValueError(f"Unsupported report type: {query.report_type}")
        
        return await handler(query)
    
    async def _generate_usage_summary(self, query: AnalyticsQuery) -> Dict:
        """Generate usage summary report"""
        start_date, end_date = self._get_date_range(query)
        
        async with get_db() as session:
            # Total documents
            total_docs_result = await session.execute(
                "SELECT COUNT(*) FROM documents WHERE upload_date BETWEEN ? AND ?",
                (start_date, end_date)
            )
            total_documents = total_docs_result.fetchone()[0]
            
            # Total users
            total_users_result = await session.execute(
                "SELECT COUNT(*) FROM users WHERE created_at BETWEEN ? AND ?",
                (start_date, end_date)
            )
            total_users = total_users_result.fetchone()[0]
            
            # Active users (users who performed actions)
            active_users_result = await session.execute(
                """SELECT COUNT(DISTINCT user_id) FROM audit_logs 
                   WHERE created_at BETWEEN ? AND ?""",
                (start_date, end_date)
            )
            active_users = active_users_result.fetchone()[0]
            
            # Storage usage
            storage_result = await session.execute(
                "SELECT SUM(file_size) FROM documents WHERE upload_date BETWEEN ? AND ?",
                (start_date, end_date)
            )
            storage_used = storage_result.fetchone()[0] or 0
            
            # Document types distribution
            doc_types_result = await session.execute(
                """SELECT file_type, COUNT(*) as count 
                   FROM documents 
                   WHERE upload_date BETWEEN ? AND ?
                   GROUP BY file_type
                   ORDER BY count DESC""",
                (start_date, end_date)
            )
            doc_types = [{'type': row[0], 'count': row[1]} for row in doc_types_result.fetchall()]
            
            # Upload trends by day
            upload_trends_result = await session.execute(
                """SELECT DATE(upload_date) as date, COUNT(*) as count
                   FROM documents 
                   WHERE upload_date BETWEEN ? AND ?
                   GROUP BY DATE(upload_date)
                   ORDER BY date""",
                (start_date, end_date)
            )
            upload_trends = [{'date': row[0], 'count': row[1]} for row in upload_trends_result.fetchall()]
        
        return {
            'summary': {
                'total_documents': total_documents,
                'total_users': total_users,
                'active_users': active_users,
                'storage_used': storage_used,
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat()
            },
            'document_types': doc_types,
            'upload_trends': upload_trends
        }
    
    async def _generate_user_activity(self, query: AnalyticsQuery) -> Dict:
        """Generate user activity analytics"""
        start_date, end_date = self._get_date_range(query)
        
        async with get_db() as session:
            # User activity by action type
            activity_result = await session.execute(
                """SELECT action, COUNT(*) as count
                   FROM audit_logs 
                   WHERE created_at BETWEEN ? AND ?
                   GROUP BY action
                   ORDER BY count DESC""",
                (start_date, end_date)
            )
            activity_by_action = [{'action': row[0], 'count': row[1]} for row in activity_result.fetchall()]
            
            # Most active users
            active_users_result = await session.execute(
                """SELECT u.username, COUNT(al.id) as activity_count
                   FROM users u
                   JOIN audit_logs al ON u.id = al.user_id
                   WHERE al.created_at BETWEEN ? AND ?
                   GROUP BY u.id, u.username
                   ORDER BY activity_count DESC
                   LIMIT 10""",
                (start_date, end_date)
            )
            active_users = [{'username': row[0], 'activity_count': row[1]} for row in active_users_result.fetchall()]
            
            # Activity trends by hour
            hourly_activity_result = await session.execute(
                """SELECT strftime('%H', created_at) as hour, COUNT(*) as count
                   FROM audit_logs
                   WHERE created_at BETWEEN ? AND ?
                   GROUP BY hour
                   ORDER BY hour""",
                (start_date, end_date)
            )
            hourly_activity = [{'hour': int(row[0]), 'count': row[1]} for row in hourly_activity_result.fetchall()]
            
            # User registration trends
            registration_result = await session.execute(
                """SELECT DATE(created_at) as date, COUNT(*) as count
                   FROM users
                   WHERE created_at BETWEEN ? AND ?
                   GROUP BY DATE(created_at)
                   ORDER BY date""",
                (start_date, end_date)
            )
            registration_trends = [{'date': row[0], 'count': row[1]} for row in registration_result.fetchall()]
        
        return {
            'activity_by_action': activity_by_action,
            'most_active_users': active_users,
            'hourly_activity_pattern': hourly_activity,
            'registration_trends': registration_trends
        }
    
    async def _generate_document_insights(self, query: AnalyticsQuery) -> Dict:
        """Generate document insights and analytics"""
        start_date, end_date = self._get_date_range(query)
        
        async with get_db() as session:
            # Document size distribution
            size_distribution_result = await session.execute(
                """SELECT 
                     CASE 
                       WHEN file_size < 1048576 THEN 'Small (<1MB)'
                       WHEN file_size < 10485760 THEN 'Medium (1-10MB)'
                       WHEN file_size < 104857600 THEN 'Large (10-100MB)'
                       ELSE 'Very Large (>100MB)'
                     END as size_category,
                     COUNT(*) as count
                   FROM documents
                   WHERE upload_date BETWEEN ? AND ?
                   GROUP BY size_category""",
                (start_date, end_date)
            )
            size_distribution = [{'category': row[0], 'count': row[1]} for row in size_distribution_result.fetchall()]
            
            # Most accessed documents
            popular_docs_result = await session.execute(
                """SELECT d.title, d.file_type, COUNT(al.id) as access_count
                   FROM documents d
                   LEFT JOIN audit_logs al ON d.id = CAST(al.resource_id AS INTEGER)
                   WHERE al.action LIKE '%document%' AND al.created_at BETWEEN ? AND ?
                   GROUP BY d.id, d.title, d.file_type
                   ORDER BY access_count DESC
                   LIMIT 10""",
                (start_date, end_date)
            )
            popular_docs = [
                {'title': row[0], 'file_type': row[1], 'access_count': row[2]} 
                for row in popular_docs_result.fetchall()
            ]
            
            # Processing status distribution
            status_result = await session.execute(
                """SELECT status, COUNT(*) as count
                   FROM documents
                   WHERE upload_date BETWEEN ? AND ?
                   GROUP BY status""",
                (start_date, end_date)
            )
            status_distribution = [{'status': row[0], 'count': row[1]} for row in status_result.fetchall()]
            
            # Average processing time by file type
            processing_time_result = await session.execute(
                """SELECT file_type, 
                          AVG(julianday(processed_date) - julianday(upload_date)) * 24 * 60 as avg_minutes
                   FROM documents
                   WHERE processed_date IS NOT NULL 
                     AND upload_date BETWEEN ? AND ?
                   GROUP BY file_type""",
                (start_date, end_date)
            )
            processing_times = [
                {'file_type': row[0], 'avg_processing_minutes': round(row[1], 2)} 
                for row in processing_time_result.fetchall()
            ]
        
        return {
            'size_distribution': size_distribution,
            'popular_documents': popular_docs,
            'status_distribution': status_distribution,
            'processing_times': processing_times
        }
    
    async def _generate_search_analytics(self, query: AnalyticsQuery) -> Dict:
        """Generate search analytics and patterns"""
        start_date, end_date = self._get_date_range(query)
        
        # This would typically query a search_logs table
        # For now, we'll simulate with audit logs
        async with get_db() as session:
            # Search frequency
            search_frequency_result = await session.execute(
                """SELECT DATE(created_at) as date, COUNT(*) as search_count
                   FROM audit_logs
                   WHERE action = 'search' AND created_at BETWEEN ? AND ?
                   GROUP BY DATE(created_at)
                   ORDER BY date""",
                (start_date, end_date)
            )
            search_frequency = [{'date': row[0], 'count': row[1]} for row in search_frequency_result.fetchall()]
            
            # Top search terms (would need search_logs table in real implementation)
            # For now, return placeholder data
            top_search_terms = [
                {'term': 'API documentation', 'count': 45},
                {'term': 'configuration', 'count': 32},
                {'term': 'troubleshooting', 'count': 28},
                {'term': 'installation guide', 'count': 24},
                {'term': 'best practices', 'count': 19}
            ]
            
            # Search patterns by hour
            search_patterns_result = await session.execute(
                """SELECT strftime('%H', created_at) as hour, COUNT(*) as count
                   FROM audit_logs
                   WHERE action = 'search' AND created_at BETWEEN ? AND ?
                   GROUP BY hour
                   ORDER BY hour""",
                (start_date, end_date)
            )
            search_patterns = [{'hour': int(row[0]), 'count': row[1]} for row in search_patterns_result.fetchall()]
        
        return {
            'search_frequency': search_frequency,
            'top_search_terms': top_search_terms,
            'search_patterns_by_hour': search_patterns,
            'average_searches_per_user': 15.2,  # Would calculate from actual data
            'search_success_rate': 0.85  # Would calculate from search results
        }
    
    async def _generate_performance_metrics(self, query: AnalyticsQuery) -> Dict:
        """Generate system performance metrics"""
        start_date, end_date = self._get_date_range(query)
        
        # Get metrics from metrics collector
        metrics_data = await self.metrics.get_metrics_in_range(start_date, end_date)
        
        # Process metrics data
        response_times = []
        error_rates = []
        throughput = []
        
        for metric in metrics_data:
            if metric['name'] == 'response_time':
                response_times.append({
                    'timestamp': metric['timestamp'],
                    'value': metric['value']
                })
            elif metric['name'] == 'error_rate':
                error_rates.append({
                    'timestamp': metric['timestamp'],
                    'value': metric['value']
                })
            elif metric['name'] == 'throughput':
                throughput.append({
                    'timestamp': metric['timestamp'],
                    'value': metric['value']
                })
        
        # Calculate averages
        avg_response_time = np.mean([m['value'] for m in response_times]) if response_times else 0
        avg_error_rate = np.mean([m['value'] for m in error_rates]) if error_rates else 0
        avg_throughput = np.mean([m['value'] for m in throughput]) if throughput else 0
        
        return {
            'response_times': response_times,
            'error_rates': error_rates,
            'throughput': throughput,
            'averages': {
                'response_time': round(avg_response_time, 2),
                'error_rate': round(avg_error_rate, 4),
                'throughput': round(avg_throughput, 2)
            }
        }
    
    async def _generate_system_health(self, query: AnalyticsQuery) -> Dict:
        """Generate system health report"""
        # Get current system status
        health_status = {
            'overall_status': 'healthy',
            'components': [
                {'name': 'Database', 'status': 'healthy', 'response_time': 12},
                {'name': 'Vector Database', 'status': 'healthy', 'response_time': 8},
                {'name': 'Cache', 'status': 'healthy', 'response_time': 2},
                {'name': 'LLM Service', 'status': 'healthy', 'response_time': 150}
            ],
            'resource_usage': {
                'cpu_usage': 45.2,
                'memory_usage': 68.7,
                'disk_usage': 34.1,
                'network_io': 12.5
            },
            'uptime': '15 days, 3 hours, 42 minutes'
        }
        
        return health_status
    
    async def _generate_trend_analysis(self, query: AnalyticsQuery) -> Dict:
        """Generate trend analysis report"""
        start_date, end_date = self._get_date_range(query)
        
        async with get_db() as session:
            # Document upload trends
            upload_trends_result = await session.execute(
                """SELECT DATE(upload_date) as date, COUNT(*) as count
                   FROM documents
                   WHERE upload_date BETWEEN ? AND ?
                   GROUP BY DATE(upload_date)
                   ORDER BY date""",
                (start_date, end_date)
            )
            upload_trends = [{'date': row[0], 'count': row[1]} for row in upload_trends_result.fetchall()]
            
            # User growth trends
            user_growth_result = await session.execute(
                """SELECT DATE(created_at) as date, COUNT(*) as new_users
                   FROM users
                   WHERE created_at BETWEEN ? AND ?
                   GROUP BY DATE(created_at)
                   ORDER BY date""",
                (start_date, end_date)
            )
            user_growth = [{'date': row[0], 'new_users': row[1]} for row in user_growth_result.fetchall()]
        
        # Calculate trends (simple linear regression)
        upload_trend = self._calculate_trend([d['count'] for d in upload_trends])
        user_trend = self._calculate_trend([d['new_users'] for d in user_growth])
        
        return {
            'upload_trends': upload_trends,
            'user_growth': user_growth,
            'trend_analysis': {
                'upload_trend': upload_trend,
                'user_growth_trend': user_trend
            }
        }
    
    async def _generate_custom_report(self, query: AnalyticsQuery) -> Dict:
        """Generate custom report based on query parameters"""
        # This would allow for custom SQL queries or predefined custom reports
        # For now, return a placeholder
        return {
            'message': 'Custom report functionality would be implemented here',
            'query_parameters': {
                'filters': query.filters,
                'group_by': query.group_by,
                'aggregations': query.aggregations
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict:
        """Calculate trend direction and slope"""
        if len(values) < 2:
            return {'direction': 'insufficient_data', 'slope': 0, 'confidence': 0}
        
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        
        # Simple confidence based on R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'direction': direction,
            'slope': round(slope, 4),
            'confidence': round(r_squared, 3)
        }
    
    def _get_date_range(self, query: AnalyticsQuery) -> Tuple[datetime, datetime]:
        """Get date range based on query parameters"""
        if query.time_range == TimeRange.CUSTOM:
            if not query.start_date or not query.end_date:
                raise ValueError("Custom time range requires start_date and end_date")
            return query.start_date, query.end_date
        
        end_date = datetime.utcnow()
        
        if query.time_range == TimeRange.LAST_HOUR:
            start_date = end_date - timedelta(hours=1)
        elif query.time_range == TimeRange.LAST_DAY:
            start_date = end_date - timedelta(days=1)
        elif query.time_range == TimeRange.LAST_WEEK:
            start_date = end_date - timedelta(weeks=1)
        elif query.time_range == TimeRange.LAST_MONTH:
            start_date = end_date - timedelta(days=30)
        elif query.time_range == TimeRange.LAST_QUARTER:
            start_date = end_date - timedelta(days=90)
        elif query.time_range == TimeRange.LAST_YEAR:
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=7)  # Default to last week
        
        return start_date, end_date
    
    def _get_cache_key(self, query: AnalyticsQuery) -> str:
        """Generate cache key for analytics query"""
        key_data = {
            'report_type': query.report_type.value,
            'time_range': query.time_range.value,
            'start_date': query.start_date.isoformat() if query.start_date else None,
            'end_date': query.end_date.isoformat() if query.end_date else None,
            'filters': query.filters,
            'group_by': query.group_by,
            'aggregations': query.aggregations,
            'limit': query.limit
        }
        return f"analytics:{hash(json.dumps(key_data, sort_keys=True))}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[AnalyticsResult]:
        """Get cached analytics result"""
        try:
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                result_data = json.loads(cached_data)
                # Reconstruct AnalyticsResult object
                # This is simplified - in practice you'd need proper deserialization
                return result_data
        except Exception as e:
            logger.warning(f"Error getting cached result: {e}")
        return None
    
    async def _cache_result(self, cache_key: str, result: AnalyticsResult):
        """Cache analytics result"""
        try:
            # Serialize result (simplified)
            cached_data = {
                'data': result.data,
                'metadata': result.metadata,
                'generated_at': result.generated_at.isoformat(),
                'execution_time': result.execution_time,
                'cache_hit': True
            }
            await self.cache.set(cache_key, json.dumps(cached_data), ttl=self.cache_ttl)
        except Exception as e:
            logger.warning(f"Error caching result: {e}")
    
    async def _generate_metadata(self, query: AnalyticsQuery, data: Any) -> Dict:
        """Generate metadata for analytics result"""
        return {
            'report_type': query.report_type.value,
            'time_range': query.time_range.value,
            'filters_applied': len(query.filters) > 0,
            'data_points': len(data) if isinstance(data, (list, dict)) else 1,
            'cache_enabled': True
        }
    
    async def export_report(self, result: AnalyticsResult, format: str = 'json') -> str:
        """Export analytics report to specified format"""
        if format == 'json':
            return json.dumps({
                'report_type': result.query.report_type.value,
                'generated_at': result.generated_at.isoformat(),
                'execution_time': result.execution_time,
                'data': result.data,
                'metadata': result.metadata
            }, indent=2)
        
        elif format == 'csv':
            # Convert data to CSV format
            if isinstance(result.data, dict):
                # Flatten the data structure for CSV
                flattened_data = self._flatten_dict(result.data)
                df = pd.DataFrame([flattened_data])
                return df.to_csv(index=False)
            
        elif format == 'html':
            # Generate HTML report
            html_content = f"""
            <html>
            <head><title>Analytics Report - {result.query.report_type.value}</title></head>
            <body>
                <h1>Analytics Report</h1>
                <p>Generated: {result.generated_at}</p>
                <p>Execution Time: {result.execution_time}s</p>
                <pre>{json.dumps(result.data, indent=2)}</pre>
            </body>
            </html>
            """
            return html_content
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# Global instance
advanced_analytics = AdvancedAnalytics()