"""
System health dashboard and reporting system for RAG application.
Provides real-time health monitoring, reporting, and visualization capabilities.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
from pathlib import Path

from utils.monitoring.health_check import get_health_checker, HealthStatus
from utils.monitoring.metrics import get_metrics_collector
from utils.monitoring.error_tracker import get_error_tracker
from utils.monitoring.usage_statistics import get_usage_stats_collector
from utils.performance.load_balancer import get_load_balancer
from utils.performance.batch_processor import get_batch_processor
from utils.performance.query_optimizer import get_query_optimizer

logger = logging.getLogger(__name__)

class DashboardTheme(Enum):
    """Dashboard themes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"

class ReportFormat(Enum):
    """Report formats."""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"

@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Any
    unit: Optional[str] = None
    status: HealthStatus = HealthStatus.HEALTHY
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trend: Optional[str] = None  # "up", "down", "stable"

@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    refresh_interval: int = 30  # seconds
    theme: DashboardTheme = DashboardTheme.AUTO
    enable_real_time: bool = True
    max_history_points: int = 100
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    visible_sections: List[str] = field(default_factory=lambda: [
        "system_overview", "performance", "errors", "usage", "services"
    ])

class HealthDashboard:
    """
    Comprehensive system health dashboard and reporting system.
    
    Features:
    - Real-time health monitoring
    - Performance metrics visualization
    - Error tracking and alerts
    - Usage analytics display
    - Service health overview
    - Historical trend analysis
    - Automated reporting
    - Custom alerting rules
    """
    
    def __init__(
        self,
        config: DashboardConfig = None,
        reports_dir: str = "health_reports"
    ):
        """
        Initialize health dashboard.
        
        Args:
            config: Dashboard configuration
            reports_dir: Directory for generated reports
        """
        self.config = config or DashboardConfig()
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Dashboard state
        self.running = False
        self.last_update = datetime.now(timezone.utc)
        self.health_history: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        # Component references
        self.health_checker = get_health_checker()
        self.metrics_collector = get_metrics_collector()
        self.error_tracker = get_error_tracker()
        self.usage_stats = get_usage_stats_collector()
        self.load_balancer = get_load_balancer()
        self.batch_processor = get_batch_processor()
        self.query_optimizer = get_query_optimizer()
        
        # Background tasks
        self.update_task: Optional[asyncio.Task] = None
        self.report_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the health dashboard."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.update_task = asyncio.create_task(self._update_loop())
        self.report_task = asyncio.create_task(self._report_loop())
        
        logger.info("Health dashboard started")
    
    async def stop(self):
        """Stop the health dashboard."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop background tasks
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        if self.report_task:
            self.report_task.cancel()
            try:
                await self.report_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health dashboard stopped")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data."""
        try:
            current_time = datetime.now(timezone.utc)
            
            dashboard_data = {
                'timestamp': current_time.isoformat(),
                'last_update': self.last_update.isoformat(),
                'config': {
                    'refresh_interval': self.config.refresh_interval,
                    'theme': self.config.theme.value,
                    'enable_real_time': self.config.enable_real_time
                }
            }
            
            # System overview
            if "system_overview" in self.config.visible_sections:
                dashboard_data['system_overview'] = await self._get_system_overview()
            
            # Performance metrics
            if "performance" in self.config.visible_sections:
                dashboard_data['performance'] = await self._get_performance_metrics()
            
            # Error tracking
            if "errors" in self.config.visible_sections:
                dashboard_data['errors'] = await self._get_error_summary()
            
            # Usage statistics
            if "usage" in self.config.visible_sections:
                dashboard_data['usage'] = await self._get_usage_summary()
            
            # Service health
            if "services" in self.config.visible_sections:
                dashboard_data['services'] = await self._get_service_health()
            
            # Alerts and notifications
            dashboard_data['alerts'] = await self._get_active_alerts()
            
            # Historical trends
            dashboard_data['trends'] = await self._get_trend_data()
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def get_system_status_summary(self) -> Dict[str, Any]:
        """Get high-level system status summary."""
        try:
            # Get basic health check
            overall_health = await self.health_checker.check_overall_health()
            
            # Get critical metrics
            if self.metrics_collector:
                metrics = self.metrics_collector.get_current_metrics()
                error_rate = metrics.get('error_rate', 0.0)
                avg_response_time = metrics.get('average_response_time', 0.0)
            else:
                error_rate = 0.0
                avg_response_time = 0.0
            
            # Get usage stats
            if self.usage_stats:
                realtime_metrics = await self.usage_stats.get_realtime_metrics()
                active_users = realtime_metrics.get('active_users_count', 0)
                requests_today = realtime_metrics.get('total_requests_today', 0)
            else:
                active_users = 0
                requests_today = 0
            
            # Determine overall status
            if overall_health['status'] == 'healthy' and error_rate < 5.0:
                overall_status = 'healthy'
                status_color = 'green'
            elif overall_health['status'] == 'degraded' or error_rate < 10.0:
                overall_status = 'warning'
                status_color = 'yellow'
            else:
                overall_status = 'critical'
                status_color = 'red'
            
            return {
                'overall_status': overall_status,
                'status_color': status_color,
                'health_score': overall_health.get('health_score', 0),
                'key_metrics': {
                    'error_rate': error_rate,
                    'avg_response_time': avg_response_time,
                    'active_users': active_users,
                    'requests_today': requests_today
                },
                'components': {
                    'database': overall_health.get('components', {}).get('database', {}).get('status', 'unknown'),
                    'vector_db': overall_health.get('components', {}).get('vector_database', {}).get('status', 'unknown'),
                    'cache': overall_health.get('components', {}).get('cache', {}).get('status', 'unknown'),
                    'api': overall_health.get('components', {}).get('api', {}).get('status', 'unknown')
                },
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status summary: {e}")
            return {
                'overall_status': 'unknown',
                'status_color': 'gray',
                'error': str(e),
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
    
    async def generate_health_report(
        self,
        report_format: ReportFormat = ReportFormat.HTML,
        time_period_hours: int = 24,
        include_charts: bool = True
    ) -> str:
        """
        Generate comprehensive health report.
        
        Args:
            report_format: Output format for the report
            time_period_hours: Time period to cover in hours
            include_charts: Whether to include charts and graphs
            
        Returns:
            Path to generated report file
        """
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_filename = f"health_report_{timestamp}.{report_format.value}"
            report_path = self.reports_dir / report_filename
            
            # Gather report data
            report_data = await self._gather_report_data(time_period_hours)
            
            # Generate report based on format
            if report_format == ReportFormat.HTML:
                await self._generate_html_report(report_data, report_path, include_charts)
            elif report_format == ReportFormat.JSON:
                await self._generate_json_report(report_data, report_path)
            elif report_format == ReportFormat.CSV:
                await self._generate_csv_report(report_data, report_path)
            elif report_format == ReportFormat.PDF:
                await self._generate_pdf_report(report_data, report_path, include_charts)
            else:
                raise ValueError(f"Unsupported report format: {report_format}")
            
            logger.info(f"Generated health report: {report_path}")
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            raise
    
    async def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends for the specified time period."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Get historical data points
            historical_data = [
                point for point in self.health_history
                if datetime.fromisoformat(point['timestamp']) >= cutoff_time
            ]
            
            if not historical_data:
                return {'message': 'No historical data available'}
            
            # Calculate trends
            trends = {
                'response_time': self._calculate_trend([p.get('avg_response_time', 0) for p in historical_data]),
                'error_rate': self._calculate_trend([p.get('error_rate', 0) for p in historical_data]),
                'memory_usage': self._calculate_trend([p.get('memory_usage_percent', 0) for p in historical_data]),
                'cpu_usage': self._calculate_trend([p.get('cpu_usage_percent', 0) for p in historical_data]),
                'active_users': self._calculate_trend([p.get('active_users', 0) for p in historical_data])
            }
            
            # Get current vs previous period comparison
            mid_point = len(historical_data) // 2
            if mid_point > 0:
                recent_data = historical_data[mid_point:]
                older_data = historical_data[:mid_point]
                
                comparisons = {}
                for metric in ['avg_response_time', 'error_rate', 'memory_usage_percent']:
                    recent_avg = sum(p.get(metric, 0) for p in recent_data) / len(recent_data)
                    older_avg = sum(p.get(metric, 0) for p in older_data) / len(older_data)
                    
                    if older_avg > 0:
                        change_percent = ((recent_avg - older_avg) / older_avg) * 100
                        comparisons[metric] = {
                            'recent_avg': recent_avg,
                            'older_avg': older_avg,
                            'change_percent': change_percent,
                            'improving': change_percent < 0 if metric != 'active_users' else change_percent > 0
                        }
            else:
                comparisons = {}
            
            return {
                'period_hours': hours,
                'data_points': len(historical_data),
                'trends': trends,
                'comparisons': comparisons,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return {'error': str(e)}
    
    async def _get_system_overview(self) -> Dict[str, Any]:
        """Get system overview data."""
        try:
            overview = await self.health_checker.check_system_health()
            
            # Add uptime calculation
            if hasattr(self, 'start_time'):
                uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                overview['uptime'] = {
                    'seconds': uptime_seconds,
                    'human_readable': self._format_uptime(uptime_seconds)
                }
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting system overview: {e}")
            return {'error': str(e)}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics data."""
        try:
            performance_data = {}
            
            # Get metrics from collector
            if self.metrics_collector:
                current_metrics = self.metrics_collector.get_current_metrics()
                performance_data['current_metrics'] = current_metrics
            
            # Get load balancer metrics
            if self.load_balancer:
                lb_metrics = await self.load_balancer.get_performance_metrics()
                performance_data['load_balancer'] = lb_metrics
            
            # Get batch processor metrics
            if self.batch_processor:
                batch_metrics = await self.batch_processor.get_performance_stats()
                performance_data['batch_processor'] = batch_metrics
            
            # Get query optimizer metrics
            if self.query_optimizer:
                query_metrics = await self.query_optimizer.get_query_statistics(hours=1)
                performance_data['query_optimizer'] = query_metrics
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {'error': str(e)}
    
    async def _get_error_summary(self) -> Dict[str, Any]:
        """Get error tracking summary."""
        try:
            if not self.error_tracker:
                return {'message': 'Error tracker not available'}
            
            # Get recent errors
            recent_errors = self.error_tracker.get_recent_errors(hours=24)
            
            # Categorize errors
            error_categories = {}
            for error in recent_errors:
                category = error.get('category', 'unknown')
                if category not in error_categories:
                    error_categories[category] = []
                error_categories[category].append(error)
            
            # Get error trends
            error_trends = self.error_tracker.get_error_trends(hours=24)
            
            return {
                'recent_errors_count': len(recent_errors),
                'error_categories': {k: len(v) for k, v in error_categories.items()},
                'error_trends': error_trends,
                'top_errors': recent_errors[:10]  # Top 10 recent errors
            }
            
        except Exception as e:
            logger.error(f"Error getting error summary: {e}")
            return {'error': str(e)}
    
    async def _get_usage_summary(self) -> Dict[str, Any]:
        """Get usage statistics summary."""
        try:
            if not self.usage_stats:
                return {'message': 'Usage statistics not available'}
            
            # Get real-time metrics
            realtime_metrics = await self.usage_stats.get_realtime_metrics()
            
            # Get usage statistics for different periods
            daily_stats = await self.usage_stats.get_usage_statistics(time_period='day')
            weekly_stats = await self.usage_stats.get_usage_statistics(time_period='week')
            
            return {
                'realtime': realtime_metrics,
                'daily': daily_stats,
                'weekly': weekly_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting usage summary: {e}")
            return {'error': str(e)}
    
    async def _get_service_health(self) -> Dict[str, Any]:
        """Get service health data."""
        try:
            # Get overall service health
            service_health = await self.health_checker.check_services_health()
            
            # Add load balancer service status if available
            if self.load_balancer:
                lb_status = await self.load_balancer.get_service_health()
                service_health['load_balancer_services'] = lb_status
            
            return service_health
            
        except Exception as e:
            logger.error(f"Error getting service health: {e}")
            return {'error': str(e)}
    
    async def _get_active_alerts(self) -> Dict[str, Any]:
        """Get active alerts and notifications."""
        try:
            # This would integrate with the alert manager
            # For now, return placeholder data
            return {
                'active_alerts_count': 0,
                'alerts_by_severity': {
                    'critical': 0,
                    'warning': 0,
                    'info': 0
                },
                'recent_alerts': []
            }
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return {'error': str(e)}
    
    async def _get_trend_data(self) -> Dict[str, Any]:
        """Get historical trend data."""
        try:
            # Return recent history points for trending
            recent_history = self.health_history[-self.config.max_history_points:]
            
            return {
                'history_points': len(recent_history),
                'data': recent_history[-20:]  # Last 20 points for charts
            }
            
        except Exception as e:
            logger.error(f"Error getting trend data: {e}")
            return {'error': str(e)}
    
    async def _update_loop(self):
        """Background task to update dashboard data."""
        while self.running:
            try:
                # Collect current health data
                health_data = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'system_health': await self._get_system_overview(),
                    'performance': await self._get_performance_metrics(),
                    'errors': await self._get_error_summary(),
                    'usage': await self._get_usage_summary()
                }
                
                # Add to history
                self.health_history.append(health_data)
                
                # Limit history size
                if len(self.health_history) > self.config.max_history_points:
                    self.health_history = self.health_history[-self.config.max_history_points:]
                
                self.last_update = datetime.now(timezone.utc)
                
                await asyncio.sleep(self.config.refresh_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(self.config.refresh_interval)
    
    async def _report_loop(self):
        """Background task for automated reporting."""
        while self.running:
            try:
                # Generate daily reports at midnight
                now = datetime.now(timezone.utc)
                if now.hour == 0 and now.minute < 5:  # Run in first 5 minutes of day
                    try:
                        report_path = await self.generate_health_report(
                            report_format=ReportFormat.HTML,
                            time_period_hours=24
                        )
                        logger.info(f"Generated automated daily report: {report_path}")
                    except Exception as e:
                        logger.error(f"Error generating automated report: {e}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in report loop: {e}")
                await asyncio.sleep(300)
    
    async def _gather_report_data(self, hours: int) -> Dict[str, Any]:
        """Gather data for report generation."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Get current dashboard data
        current_data = await self.get_dashboard_data()
        
        # Get historical data for the period
        historical_data = [
            point for point in self.health_history
            if datetime.fromisoformat(point['timestamp']) >= cutoff_time
        ]
        
        # Get performance trends
        trends = await self.get_performance_trends(hours)
        
        return {
            'report_period_hours': hours,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'current_data': current_data,
            'historical_data': historical_data,
            'trends': trends,
            'summary': await self.get_system_status_summary()
        }
    
    async def _generate_html_report(self, data: Dict[str, Any], report_path: Path, include_charts: bool):
        """Generate HTML health report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>System Health Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 3px; }
                .status-healthy { color: green; }
                .status-warning { color: orange; }
                .status-critical { color: red; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>System Health Report</h1>
                <p>Generated: {generated_at}</p>
                <p>Report Period: {period} hours</p>
            </div>
            
            <div class="section">
                <h2>System Overview</h2>
                <div class="metric">
                    <strong>Overall Status:</strong> 
                    <span class="status-{status_class}">{overall_status}</span>
                </div>
                <div class="metric">
                    <strong>Health Score:</strong> {health_score}%
                </div>
            </div>
            
            <div class="section">
                <h2>Key Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Current Value</th><th>Trend</th></tr>
                    <tr><td>Error Rate</td><td>{error_rate}%</td><td>{error_rate_trend}</td></tr>
                    <tr><td>Average Response Time</td><td>{avg_response_time}ms</td><td>{response_time_trend}</td></tr>
                    <tr><td>Active Users</td><td>{active_users}</td><td>{users_trend}</td></tr>
                    <tr><td>Requests Today</td><td>{requests_today}</td><td>-</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Component Health</h2>
                <table>
                    <tr><th>Component</th><th>Status</th></tr>
                    <tr><td>Database</td><td class="status-{db_status}">{database_status}</td></tr>
                    <tr><td>Vector Database</td><td class="status-{vector_status}">{vector_db_status}</td></tr>
                    <tr><td>Cache</td><td class="status-{cache_status}">{cache_status}</td></tr>
                    <tr><td>API</td><td class="status-{api_status}">{api_status}</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        # Extract data for template
        summary = data.get('summary', {})
        key_metrics = summary.get('key_metrics', {})
        components = summary.get('components', {})
        
        # Format the HTML
        html_content = html_template.format(
            generated_at=data['generated_at'],
            period=data['report_period_hours'],
            overall_status=summary.get('overall_status', 'unknown'),
            status_class=summary.get('overall_status', 'unknown'),
            health_score=summary.get('health_score', 0),
            error_rate=key_metrics.get('error_rate', 0),
            error_rate_trend='↓' if key_metrics.get('error_rate', 0) < 5 else '↑',
            avg_response_time=key_metrics.get('avg_response_time', 0),
            response_time_trend='↓' if key_metrics.get('avg_response_time', 0) < 1000 else '↑',
            active_users=key_metrics.get('active_users', 0),
            users_trend='↑',
            requests_today=key_metrics.get('requests_today', 0),
            database_status=components.get('database', 'unknown'),
            db_status=components.get('database', 'unknown'),
            vector_db_status=components.get('vector_db', 'unknown'),
            vector_status=components.get('vector_db', 'unknown'),
            cache_status=components.get('cache', 'unknown'),
            api_status=components.get('api', 'unknown')
        )
        
        # Write to file
        with open(report_path, 'w') as f:
            f.write(html_content)
    
    async def _generate_json_report(self, data: Dict[str, Any], report_path: Path):
        """Generate JSON health report."""
        with open(report_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def _generate_csv_report(self, data: Dict[str, Any], report_path: Path):
        """Generate CSV health report."""
        # Extract key metrics for CSV format
        summary = data.get('summary', {})
        key_metrics = summary.get('key_metrics', {})
        
        csv_content = [
            "Metric,Value,Unit",
            f"Overall Status,{summary.get('overall_status', 'unknown')},",
            f"Health Score,{summary.get('health_score', 0)},%",
            f"Error Rate,{key_metrics.get('error_rate', 0)},%",
            f"Average Response Time,{key_metrics.get('avg_response_time', 0)},ms",
            f"Active Users,{key_metrics.get('active_users', 0)},count",
            f"Requests Today,{key_metrics.get('requests_today', 0)},count"
        ]
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(csv_content))
    
    async def _generate_pdf_report(self, data: Dict[str, Any], report_path: Path, include_charts: bool):
        """Generate PDF health report."""
        # For PDF generation, you would typically use a library like reportlab
        # For now, generate HTML and note that PDF conversion would be added
        html_path = report_path.with_suffix('.html')
        await self._generate_html_report(data, html_path, include_charts)
        
        # Note: In production, convert HTML to PDF using tools like wkhtmltopdf or similar
        logger.info(f"HTML report generated at {html_path} (PDF conversion not implemented)")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        if not first_half or not second_half:
            return "stable"
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"


# Global health dashboard instance
_health_dashboard: Optional[HealthDashboard] = None

def get_health_dashboard() -> Optional[HealthDashboard]:
    """Get global health dashboard instance."""
    global _health_dashboard
    return _health_dashboard

def initialize_health_dashboard(
    config: DashboardConfig = None,
    reports_dir: str = "health_reports"
) -> HealthDashboard:
    """Initialize global health dashboard."""
    global _health_dashboard
    
    _health_dashboard = HealthDashboard(config, reports_dir)
    return _health_dashboard