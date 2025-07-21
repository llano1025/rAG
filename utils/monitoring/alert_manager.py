"""
Advanced alert management system for RAG application.
Provides real-time alerting, notification channels, and alert escalation.
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    SMS = "sms"
    CONSOLE = "console"

@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    condition: str  # Condition expression
    severity: AlertSeverity
    threshold: float
    evaluation_window: int  # seconds
    cooldown_period: int = 300  # seconds between duplicate alerts
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    message: str
    status: AlertStatus = AlertStatus.OPEN
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    escalation_level: int = 0
    notification_attempts: int = 0
    last_notification_at: Optional[datetime] = None

@dataclass
class NotificationConfig:
    """Notification configuration."""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    severity_filter: List[AlertSeverity] = field(default_factory=list)
    escalation_delay: int = 300  # seconds before escalation

class AlertManager:
    """
    Advanced alert management system.
    
    Features:
    - Real-time alert generation and tracking
    - Multiple notification channels
    - Alert escalation and de-duplication
    - Configurable alert rules
    - Alert history and analytics
    - Automatic alert resolution
    """
    
    def __init__(
        self,
        notification_configs: List[NotificationConfig] = None,
        max_alerts_in_memory: int = 1000,
        enable_auto_resolution: bool = True
    ):
        """
        Initialize alert manager.
        
        Args:
            notification_configs: List of notification configurations
            max_alerts_in_memory: Maximum alerts to keep in memory
            enable_auto_resolution: Enable automatic alert resolution
        """
        self.notification_configs = notification_configs or []
        self.max_alerts_in_memory = max_alerts_in_memory
        self.enable_auto_resolution = enable_auto_resolution
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_counters: Dict[str, Dict[str, Any]] = {}
        
        # Cooldown tracking
        self.rule_cooldowns: Dict[str, datetime] = {}
        
        # Background tasks
        self.running = False
        self.evaluation_task: Optional[asyncio.Task] = None
        self.escalation_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Notification handlers
        self.notification_handlers = {
            NotificationChannel.EMAIL: self._send_email_notification,
            NotificationChannel.WEBHOOK: self._send_webhook_notification,
            NotificationChannel.SLACK: self._send_slack_notification,
            NotificationChannel.SMS: self._send_sms_notification,
            NotificationChannel.CONSOLE: self._send_console_notification
        }
        
        # Default alert rules
        self._setup_default_rules()
    
    async def start(self):
        """Start the alert manager."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        self.escalation_task = asyncio.create_task(self._escalation_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Alert manager started")
    
    async def stop(self):
        """Stop the alert manager."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop background tasks
        for task in [self.evaluation_task, self.escalation_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Alert manager stopped")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.rule_id}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def enable_alert_rule(self, rule_id: str, enabled: bool = True):
        """Enable or disable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = enabled
            logger.info(f"Alert rule {rule_id} {'enabled' if enabled else 'disabled'}")
    
    async def trigger_alert(
        self,
        rule_id: str,
        title: str,
        message: str,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Trigger an alert.
        
        Args:
            rule_id: Alert rule ID
            title: Alert title
            message: Alert message
            severity: Override severity from rule
            source: Alert source
            tags: Alert tags
            metadata: Additional metadata
            
        Returns:
            Alert ID
        """
        # Check if rule exists and is enabled
        rule = self.alert_rules.get(rule_id)
        if not rule or not rule.enabled:
            return ""
        
        # Check cooldown period
        now = datetime.now(timezone.utc)
        if rule_id in self.rule_cooldowns:
            time_since_last = (now - self.rule_cooldowns[rule_id]).total_seconds()
            if time_since_last < rule.cooldown_period:
                logger.debug(f"Alert rule {rule_id} in cooldown period")
                return ""
        
        # Create alert
        alert_id = f"{rule_id}_{int(now.timestamp())}"
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule_id,
            severity=severity or rule.severity,
            title=title,
            message=message,
            source=source,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Add to active alerts
        self.active_alerts[alert_id] = alert
        
        # Update cooldown
        self.rule_cooldowns[rule_id] = now
        
        # Send notifications
        await self._send_notifications(alert)
        
        # Update counters
        self._update_alert_counters(alert)
        
        logger.info(f"Triggered alert: {alert_id} (rule: {rule_id}, severity: {alert.severity.value})")
        
        return alert_id
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        comment: Optional[str] = None
    ) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            acknowledged_by: User who acknowledged the alert
            comment: Optional comment
            
        Returns:
            True if successful
        """
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(timezone.utc)
        alert.acknowledged_by = acknowledged_by
        alert.updated_at = datetime.now(timezone.utc)
        
        if comment:
            alert.metadata['acknowledgment_comment'] = comment
        
        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        
        return True
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: Optional[str] = None,
        comment: Optional[str] = None
    ) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID
            resolved_by: User who resolved the alert
            comment: Optional resolution comment
            
        Returns:
            True if successful
        """
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc)
        alert.updated_at = datetime.now(timezone.utc)
        
        if resolved_by:
            alert.metadata['resolved_by'] = resolved_by
        
        if comment:
            alert.metadata['resolution_comment'] = comment
        
        # Move to history
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]
        
        # Limit history size
        if len(self.alert_history) > self.max_alerts_in_memory:
            self.alert_history = self.alert_history[-self.max_alerts_in_memory:]
        
        logger.info(f"Alert resolved: {alert_id}")
        
        return True
    
    async def suppress_alert(
        self,
        alert_id: str,
        suppressed_by: str,
        duration_minutes: int = 60,
        comment: Optional[str] = None
    ) -> bool:
        """
        Suppress an alert for a specified duration.
        
        Args:
            alert_id: Alert ID
            suppressed_by: User who suppressed the alert
            duration_minutes: Suppression duration in minutes
            comment: Optional comment
            
        Returns:
            True if successful
        """
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED
        alert.updated_at = datetime.now(timezone.utc)
        
        alert.metadata.update({
            'suppressed_by': suppressed_by,
            'suppressed_until': (datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)).isoformat(),
            'suppression_comment': comment
        })
        
        logger.info(f"Alert suppressed: {alert_id} for {duration_minutes} minutes by {suppressed_by}")
        
        return True
    
    def get_active_alerts(
        self,
        severity_filter: Optional[List[AlertSeverity]] = None,
        status_filter: Optional[List[AlertStatus]] = None,
        tag_filter: Optional[List[str]] = None
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.
        
        Args:
            severity_filter: Filter by severity levels
            status_filter: Filter by status
            tag_filter: Filter by tags
            
        Returns:
            List of filtered alerts
        """
        alerts = list(self.active_alerts.values())
        
        # Apply filters
        if severity_filter:
            alerts = [a for a in alerts if a.severity in severity_filter]
        
        if status_filter:
            alerts = [a for a in alerts if a.status in status_filter]
        
        if tag_filter:
            alerts = [a for a in alerts if any(tag in a.tags for tag in tag_filter)]
        
        # Sort by severity and creation time
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3
        }
        
        alerts.sort(key=lambda a: (severity_order[a.severity], a.created_at), reverse=True)
        
        return alerts
    
    def get_alert_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get alert statistics for the specified period."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Include both active and historical alerts
        all_alerts = list(self.active_alerts.values()) + [
            alert for alert in self.alert_history
            if alert.created_at >= cutoff_date
        ]
        
        if not all_alerts:
            return {
                'total_alerts': 0,
                'period_days': days,
                'severity_breakdown': {},
                'status_breakdown': {},
                'top_rules': [],
                'alert_trends': {}
            }
        
        # Calculate statistics
        severity_counts = {severity: 0 for severity in AlertSeverity}
        status_counts = {status: 0 for status in AlertStatus}
        rule_counts = {}
        daily_counts = {}
        
        for alert in all_alerts:
            # Severity breakdown
            severity_counts[alert.severity] += 1
            
            # Status breakdown
            status_counts[alert.status] += 1
            
            # Rule breakdown
            rule_counts[alert.rule_id] = rule_counts.get(alert.rule_id, 0) + 1
            
            # Daily trend
            day_key = alert.created_at.strftime('%Y-%m-%d')
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1
        
        # Top rules
        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_alerts': len(all_alerts),
            'period_days': days,
            'severity_breakdown': {k.value: v for k, v in severity_counts.items()},
            'status_breakdown': {k.value: v for k, v in status_counts.items()},
            'top_rules': [{'rule_id': rule_id, 'count': count} for rule_id, count in top_rules],
            'alert_trends': daily_counts,
            'active_alerts_count': len(self.active_alerts),
            'average_resolution_time': self._calculate_average_resolution_time(all_alerts)
        }
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for config in self.notification_configs:
            if not config.enabled:
                continue
            
            # Check severity filter
            if config.severity_filter and alert.severity not in config.severity_filter:
                continue
            
            # Send notification
            handler = self.notification_handlers.get(config.channel)
            if handler:
                try:
                    await handler(alert, config)
                    alert.notification_attempts += 1
                    alert.last_notification_at = datetime.now(timezone.utc)
                    
                except Exception as e:
                    logger.error(f"Failed to send {config.channel.value} notification for alert {alert.alert_id}: {e}")
    
    async def _send_email_notification(self, alert: Alert, config: NotificationConfig):
        """Send email notification."""
        if 'smtp_host' not in config.config or 'recipients' not in config.config:
            logger.error("Email notification config missing required fields")
            return
        
        # Create email
        msg = MIMEMultipart()
        msg['From'] = config.config.get('sender', 'noreply@example.com')
        msg['To'] = ', '.join(config.config['recipients'])
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        # Email body
        body = f"""
Alert: {alert.title}
Severity: {alert.severity.value.upper()}
Rule: {alert.rule_id}
Time: {alert.created_at.isoformat()}
Source: {alert.source or 'Unknown'}

Message:
{alert.message}

Alert ID: {alert.alert_id}
        """.strip()
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        try:
            with smtplib.SMTP(config.config['smtp_host'], config.config.get('smtp_port', 587)) as server:
                if config.config.get('use_tls', True):
                    server.starttls()
                
                if 'username' in config.config and 'password' in config.config:
                    server.login(config.config['username'], config.config['password'])
                
                server.send_message(msg)
                
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            raise
    
    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig):
        """Send webhook notification."""
        if 'url' not in config.config:
            logger.error("Webhook notification config missing URL")
            return
        
        payload = {
            'alert_id': alert.alert_id,
            'rule_id': alert.rule_id,
            'severity': alert.severity.value,
            'title': alert.title,
            'message': alert.message,
            'status': alert.status.value,
            'created_at': alert.created_at.isoformat(),
            'source': alert.source,
            'tags': alert.tags,
            'metadata': alert.metadata
        }
        
        headers = config.config.get('headers', {})
        headers['Content-Type'] = 'application/json'
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.config['url'],
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    logger.info(f"Webhook notification sent for alert {alert.alert_id}")
                else:
                    logger.error(f"Webhook notification failed with status {response.status}")
                    raise Exception(f"Webhook returned status {response.status}")
    
    async def _send_slack_notification(self, alert: Alert, config: NotificationConfig):
        """Send Slack notification."""
        if 'webhook_url' not in config.config:
            logger.error("Slack notification config missing webhook URL")
            return
        
        # Slack color mapping
        color_map = {
            AlertSeverity.CRITICAL: '#FF0000',
            AlertSeverity.ERROR: '#FF6600',
            AlertSeverity.WARNING: '#FFCC00',
            AlertSeverity.INFO: '#0099FF'
        }
        
        payload = {
            'attachments': [
                {
                    'color': color_map[alert.severity],
                    'title': alert.title,
                    'text': alert.message,
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.value.upper(), 'short': True},
                        {'title': 'Rule', 'value': alert.rule_id, 'short': True},
                        {'title': 'Source', 'value': alert.source or 'Unknown', 'short': True},
                        {'title': 'Time', 'value': alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'), 'short': True}
                    ],
                    'footer': f"Alert ID: {alert.alert_id}"
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.config['webhook_url'],
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    logger.info(f"Slack notification sent for alert {alert.alert_id}")
                else:
                    logger.error(f"Slack notification failed with status {response.status}")
                    raise Exception(f"Slack webhook returned status {response.status}")
    
    async def _send_sms_notification(self, alert: Alert, config: NotificationConfig):
        """Send SMS notification (placeholder - implement with SMS service)."""
        # Implement with SMS service like Twilio, AWS SNS, etc.
        logger.info(f"SMS notification (placeholder) for alert {alert.alert_id}")
    
    async def _send_console_notification(self, alert: Alert, config: NotificationConfig):
        """Send console notification."""
        timestamp = alert.created_at.strftime('%Y-%m-%d %H:%M:%S')
        severity_symbol = {
            AlertSeverity.CRITICAL: '🔴',
            AlertSeverity.ERROR: '🟠',
            AlertSeverity.WARNING: '🟡',
            AlertSeverity.INFO: '🔵'
        }
        
        print(f"\n{severity_symbol[alert.severity]} ALERT [{alert.severity.value.upper()}] {timestamp}")
        print(f"Rule: {alert.rule_id}")
        print(f"Title: {alert.title}")
        print(f"Message: {alert.message}")
        print(f"Source: {alert.source or 'Unknown'}")
        print(f"Alert ID: {alert.alert_id}")
        print("-" * 50)
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                description="Alert when error rate exceeds threshold",
                condition="error_rate > threshold",
                severity=AlertSeverity.ERROR,
                threshold=5.0,  # 5% error rate
                evaluation_window=300,
                cooldown_period=600
            ),
            AlertRule(
                rule_id="high_response_time",
                name="High Response Time",
                description="Alert when average response time is too high",
                condition="avg_response_time > threshold",
                severity=AlertSeverity.WARNING,
                threshold=2000.0,  # 2 seconds
                evaluation_window=300,
                cooldown_period=300
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="Alert when memory usage exceeds threshold",
                condition="memory_usage_percent > threshold",
                severity=AlertSeverity.WARNING,
                threshold=85.0,  # 85%
                evaluation_window=300,
                cooldown_period=600
            ),
            AlertRule(
                rule_id="service_down",
                name="Service Down",
                description="Alert when critical service is down",
                condition="service_health == 'unhealthy'",
                severity=AlertSeverity.CRITICAL,
                threshold=1.0,
                evaluation_window=60,
                cooldown_period=300
            ),
            AlertRule(
                rule_id="disk_space_low",
                name="Low Disk Space",
                description="Alert when disk space is running low",
                condition="disk_usage_percent > threshold",
                severity=AlertSeverity.WARNING,
                threshold=90.0,  # 90%
                evaluation_window=300,
                cooldown_period=1800
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def _update_alert_counters(self, alert: Alert):
        """Update alert counters for statistics."""
        hour_key = alert.created_at.strftime('%Y-%m-%d-%H')
        
        if hour_key not in self.alert_counters:
            self.alert_counters[hour_key] = {
                'total': 0,
                'by_severity': {s.value: 0 for s in AlertSeverity},
                'by_rule': {}
            }
        
        counters = self.alert_counters[hour_key]
        counters['total'] += 1
        counters['by_severity'][alert.severity.value] += 1
        counters['by_rule'][alert.rule_id] = counters['by_rule'].get(alert.rule_id, 0) + 1
    
    def _calculate_average_resolution_time(self, alerts: List[Alert]) -> float:
        """Calculate average resolution time for alerts."""
        resolved_alerts = [
            alert for alert in alerts
            if alert.status == AlertStatus.RESOLVED and alert.resolved_at
        ]
        
        if not resolved_alerts:
            return 0.0
        
        total_time = sum(
            (alert.resolved_at - alert.created_at).total_seconds()
            for alert in resolved_alerts
        )
        
        return total_time / len(resolved_alerts)
    
    async def _evaluation_loop(self):
        """Background task for evaluating alert conditions."""
        while self.running:
            try:
                # This would evaluate metric conditions and trigger alerts
                # For now, it's a placeholder
                await asyncio.sleep(60)  # Evaluate every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(60)
    
    async def _escalation_loop(self):
        """Background task for alert escalation."""
        while self.running:
            try:
                now = datetime.now(timezone.utc)
                
                for alert in list(self.active_alerts.values()):
                    # Check for escalation
                    if (alert.status == AlertStatus.OPEN and 
                        alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]):
                        
                        time_since_created = (now - alert.created_at).total_seconds()
                        
                        # Escalate critical alerts after 15 minutes
                        # Escalate error alerts after 30 minutes
                        escalation_threshold = 900 if alert.severity == AlertSeverity.CRITICAL else 1800
                        
                        if time_since_created > escalation_threshold and alert.escalation_level == 0:
                            alert.escalation_level += 1
                            await self._escalate_alert(alert)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert escalation loop: {e}")
                await asyncio.sleep(300)
    
    async def _escalate_alert(self, alert: Alert):
        """Escalate an alert."""
        logger.warning(f"Escalating alert: {alert.alert_id}")
        
        # Send escalation notifications
        escalation_configs = [
            config for config in self.notification_configs
            if alert.severity in config.severity_filter or not config.severity_filter
        ]
        
        for config in escalation_configs:
            if config.escalation_delay > 0:
                # Send escalated notification
                try:
                    handler = self.notification_handlers.get(config.channel)
                    if handler:
                        await handler(alert, config)
                        
                except Exception as e:
                    logger.error(f"Failed to send escalation notification: {e}")
    
    async def _cleanup_loop(self):
        """Background task for cleaning up old alerts."""
        while self.running:
            try:
                # Remove very old alert counters (older than 30 days)
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
                cutoff_key = cutoff_date.strftime('%Y-%m-%d-%H')
                
                old_keys = [
                    key for key in self.alert_counters.keys()
                    if key < cutoff_key
                ]
                
                for key in old_keys:
                    del self.alert_counters[key]
                
                if old_keys:
                    logger.info(f"Cleaned up {len(old_keys)} old alert counter entries")
                
                await asyncio.sleep(24 * 3600)  # Run daily
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert cleanup loop: {e}")
                await asyncio.sleep(24 * 3600)


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None

def get_alert_manager() -> Optional[AlertManager]:
    """Get global alert manager instance."""
    global _alert_manager
    return _alert_manager

def initialize_alert_manager(
    notification_configs: List[NotificationConfig] = None
) -> AlertManager:
    """Initialize global alert manager."""
    global _alert_manager
    
    _alert_manager = AlertManager(notification_configs or [])
    return _alert_manager

# Convenience functions
async def trigger_alert(
    rule_id: str,
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    source: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Convenience function to trigger an alert."""
    alert_manager = get_alert_manager()
    if alert_manager:
        return await alert_manager.trigger_alert(
            rule_id, title, message, severity, source, tags, metadata
        )
    return ""

async def trigger_error_alert(
    error: Exception,
    source: str,
    additional_context: Optional[Dict[str, Any]] = None
) -> str:
    """Convenience function to trigger an error alert."""
    metadata = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'source': source
    }
    
    if additional_context:
        metadata.update(additional_context)
    
    return await trigger_alert(
        rule_id="application_error",
        title=f"Application Error in {source}",
        message=f"Error: {str(error)}",
        severity=AlertSeverity.ERROR,
        source=source,
        tags=["error", "application"],
        metadata=metadata
    )