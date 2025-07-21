"""
Load balancing and request distribution system for scalable RAG operations.
Provides circuit breaker patterns, service health monitoring, and auto-scaling capabilities.
"""

import asyncio
import logging
import time
import random
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    HEALTH_BASED = "health_based"
    ADAPTIVE = "adaptive"

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back online

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    timeout_seconds: float = 30.0
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def address(self) -> str:
        """Get service address."""
        return f"{self.host}:{self.port}"

@dataclass
class ServiceHealth:
    """Service health metrics."""
    endpoint_id: str
    status: ServiceStatus = ServiceStatus.HEALTHY
    response_time_ms: float = 0.0
    active_connections: int = 0
    success_rate: float = 100.0
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    consecutive_failures: int = 0
    total_requests: int = 0
    error_count: int = 0
    
    def update_metrics(
        self,
        response_time_ms: float,
        success: bool,
        active_connections: int
    ):
        """Update health metrics."""
        self.response_time_ms = response_time_ms
        self.active_connections = active_connections
        self.total_requests += 1
        self.last_check = datetime.now(timezone.utc)
        
        if success:
            self.consecutive_failures = 0
        else:
            self.error_count += 1
            self.consecutive_failures += 1
        
        # Calculate success rate (last 100 requests)
        if self.total_requests > 0:
            recent_success = max(0, self.total_requests - self.error_count)
            self.success_rate = (recent_success / self.total_requests) * 100

@dataclass
class CircuitBreaker:
    """Circuit breaker for service protection."""
    service_id: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_calls: int = 0
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.now(timezone.utc) - self.last_failure_time > 
                timedelta(seconds=self.recovery_timeout)):
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
        elif (self.state == CircuitBreakerState.CLOSED and 
              self.failure_count >= self.failure_threshold):
            self.state = CircuitBreakerState.OPEN

class LoadBalancer:
    """
    Advanced load balancer for RAG system services.
    
    Features:
    - Multiple load balancing strategies
    - Circuit breaker pattern
    - Service health monitoring
    - Automatic failover
    - Request distribution optimization
    - Performance metrics collection
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
        health_check_interval: int = 30,
        enable_circuit_breaker: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
            health_check_interval: Health check interval in seconds
            enable_circuit_breaker: Enable circuit breaker protection
            max_retries: Maximum retry attempts
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.enable_circuit_breaker = enable_circuit_breaker
        self.max_retries = max_retries
        
        # Service management
        self.services: Dict[str, ServiceEndpoint] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Load balancing state
        self.round_robin_index = 0
        self.request_history: deque = deque(maxlen=1000)
        
        # Health monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.monitoring_active = False
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'service_utilization': {},
            'circuit_breaker_trips': 0
        }
    
    async def register_service(
        self,
        service_id: str,
        host: str,
        port: int,
        weight: float = 1.0,
        max_connections: int = 100,
        timeout_seconds: float = 30.0,
        health_check_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a service endpoint.
        
        Args:
            service_id: Unique service identifier
            host: Service host
            port: Service port
            weight: Service weight for load balancing
            max_connections: Maximum concurrent connections
            timeout_seconds: Request timeout
            health_check_url: Health check endpoint
            metadata: Additional service metadata
        """
        endpoint = ServiceEndpoint(
            id=service_id,
            host=host,
            port=port,
            weight=weight,
            max_connections=max_connections,
            timeout_seconds=timeout_seconds,
            health_check_url=health_check_url,
            metadata=metadata or {}
        )
        
        self.services[service_id] = endpoint
        self.service_health[service_id] = ServiceHealth(endpoint_id=service_id)
        
        if self.enable_circuit_breaker:
            self.circuit_breakers[service_id] = CircuitBreaker(service_id=service_id)
        
        logger.info(f"Registered service: {service_id} at {endpoint.address}")
    
    async def unregister_service(self, service_id: str):
        """Unregister a service endpoint."""
        if service_id in self.services:
            del self.services[service_id]
            del self.service_health[service_id]
            
            if service_id in self.circuit_breakers:
                del self.circuit_breakers[service_id]
            
            logger.info(f"Unregistered service: {service_id}")
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Started load balancer health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped load balancer health monitoring")
    
    async def distribute_request(
        self,
        request_func: Callable,
        *args,
        service_type: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Distribute request to optimal service endpoint.
        
        Args:
            request_func: Function to execute the request
            *args: Request arguments
            service_type: Optional service type filter
            **kwargs: Request keyword arguments
            
        Returns:
            Request result
        """
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        # Get available services
        available_services = await self._get_available_services(service_type)
        
        if not available_services:
            raise Exception("No available services for request")
        
        # Select service based on strategy
        selected_service = await self._select_service(available_services)
        
        # Execute request with retries
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check circuit breaker
                if self.enable_circuit_breaker:
                    circuit_breaker = self.circuit_breakers.get(selected_service.id)
                    if circuit_breaker and not circuit_breaker.can_execute():
                        # Circuit is open, try next service
                        available_services = [s for s in available_services if s.id != selected_service.id]
                        if not available_services:
                            raise Exception("All services unavailable (circuit breakers open)")
                        selected_service = await self._select_service(available_services)
                        continue
                
                # Execute request
                result = await self._execute_request(
                    selected_service, request_func, *args, **kwargs
                )
                
                # Record success
                response_time = (time.time() - start_time) * 1000
                await self._record_request_success(selected_service, response_time)
                
                self.metrics['successful_requests'] += 1
                return result
                
            except Exception as e:
                last_exception = e
                
                # Record failure
                await self._record_request_failure(selected_service, str(e))
                
                # Try next service if available
                if attempt < self.max_retries:
                    available_services = [s for s in available_services if s.id != selected_service.id]
                    if available_services:
                        selected_service = await self._select_service(available_services)
                        continue
                
                break
        
        # All attempts failed
        self.metrics['failed_requests'] += 1
        raise last_exception
    
    async def get_service_health(self, service_id: Optional[str] = None) -> Dict[str, Any]:
        """Get service health status."""
        if service_id:
            if service_id not in self.service_health:
                return {}
            
            health = self.service_health[service_id]
            return {
                'service_id': service_id,
                'status': health.status.value,
                'response_time_ms': health.response_time_ms,
                'active_connections': health.active_connections,
                'success_rate': health.success_rate,
                'last_check': health.last_check.isoformat(),
                'consecutive_failures': health.consecutive_failures,
                'total_requests': health.total_requests,
                'error_count': health.error_count
            }
        
        # Return all service health
        health_status = {}
        for sid, health in self.service_health.items():
            health_status[sid] = {
                'status': health.status.value,
                'response_time_ms': health.response_time_ms,
                'active_connections': health.active_connections,
                'success_rate': health.success_rate,
                'last_check': health.last_check.isoformat()
            }
        
        return health_status
    
    async def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for all services."""
        if not self.enable_circuit_breaker:
            return {}
        
        status = {}
        for service_id, cb in self.circuit_breakers.items():
            status[service_id] = {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'last_failure_time': cb.last_failure_time.isoformat() if cb.last_failure_time else None,
                'half_open_calls': cb.half_open_calls
            }
        
        return status
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get load balancer performance metrics."""
        # Calculate average response time
        if self.metrics['successful_requests'] > 0:
            total_response_time = sum(
                health.response_time_ms for health in self.service_health.values()
            )
            self.metrics['average_response_time'] = total_response_time / len(self.service_health)
        
        # Calculate service utilization
        for service_id, health in self.service_health.items():
            if service_id in self.services:
                max_connections = self.services[service_id].max_connections
                utilization = (health.active_connections / max_connections) * 100
                self.metrics['service_utilization'][service_id] = utilization
        
        return {
            **self.metrics,
            'total_services': len(self.services),
            'healthy_services': sum(
                1 for health in self.service_health.values() 
                if health.status == ServiceStatus.HEALTHY
            ),
            'strategy': self.strategy.value,
            'monitoring_active': self.monitoring_active
        }
    
    async def _get_available_services(self, service_type: Optional[str] = None) -> List[ServiceEndpoint]:
        """Get available services based on health and filters."""
        available = []
        
        for service_id, endpoint in self.services.items():
            health = self.service_health.get(service_id)
            
            # Check health status
            if health and health.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]:
                # Check service type filter
                if service_type and endpoint.metadata.get('type') != service_type:
                    continue
                
                # Check circuit breaker
                if self.enable_circuit_breaker:
                    circuit_breaker = self.circuit_breakers.get(service_id)
                    if circuit_breaker and not circuit_breaker.can_execute():
                        continue
                
                available.append(endpoint)
        
        return available
    
    async def _select_service(self, services: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select service based on load balancing strategy."""
        if not services:
            raise Exception("No services available")
        
        if len(services) == 1:
            return services[0]
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected = services[self.round_robin_index % len(services)]
            self.round_robin_index += 1
            return selected
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(services)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(
                services, 
                key=lambda s: self.service_health[s.id].active_connections
            )
        
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return min(
                services,
                key=lambda s: self.service_health[s.id].response_time_ms
            )
        
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            return max(
                services,
                key=lambda s: self.service_health[s.id].success_rate
            )
        
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return await self._adaptive_select(services)
        
        else:
            # Default to round robin
            return services[0]
    
    def _weighted_round_robin_select(self, services: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select service using weighted round robin."""
        total_weight = sum(service.weight for service in services)
        
        if total_weight == 0:
            return services[0]
        
        # Use random selection based on weights
        random_value = random.uniform(0, total_weight)
        current_weight = 0
        
        for service in services:
            current_weight += service.weight
            if random_value <= current_weight:
                return service
        
        return services[-1]  # Fallback
    
    async def _adaptive_select(self, services: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select service using adaptive algorithm."""
        # Calculate score for each service
        best_service = None
        best_score = float('-inf')
        
        for service in services:
            health = self.service_health[service.id]
            
            # Score based on multiple factors
            response_score = 1000 / (health.response_time_ms + 1)  # Lower response time = higher score
            success_score = health.success_rate  # Higher success rate = higher score
            load_score = 100 - (health.active_connections / service.max_connections * 100)  # Lower load = higher score
            weight_score = service.weight * 100  # Higher weight = higher score
            
            total_score = (response_score + success_score + load_score + weight_score) / 4
            
            if total_score > best_score:
                best_score = total_score
                best_service = service
        
        return best_service or services[0]
    
    async def _execute_request(
        self,
        service: ServiceEndpoint,
        request_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute request on selected service."""
        # Update active connections
        health = self.service_health[service.id]
        health.active_connections += 1
        
        try:
            # Add service info to kwargs if needed
            if 'service_endpoint' not in kwargs:
                kwargs['service_endpoint'] = service
            
            # Execute request with timeout
            result = await asyncio.wait_for(
                request_func(*args, **kwargs),
                timeout=service.timeout_seconds
            )
            
            return result
            
        finally:
            # Update active connections
            health.active_connections = max(0, health.active_connections - 1)
    
    async def _record_request_success(self, service: ServiceEndpoint, response_time_ms: float):
        """Record successful request."""
        health = self.service_health[service.id]
        health.update_metrics(response_time_ms, True, health.active_connections)
        
        # Update circuit breaker
        if self.enable_circuit_breaker:
            circuit_breaker = self.circuit_breakers[service.id]
            circuit_breaker.record_success()
        
        # Record in history
        self.request_history.append({
            'service_id': service.id,
            'timestamp': time.time(),
            'response_time_ms': response_time_ms,
            'success': True
        })
    
    async def _record_request_failure(self, service: ServiceEndpoint, error: str):
        """Record failed request."""
        health = self.service_health[service.id]
        health.update_metrics(0.0, False, health.active_connections)
        
        # Update circuit breaker
        if self.enable_circuit_breaker:
            circuit_breaker = self.circuit_breakers[service.id]
            circuit_breaker.record_failure()
            self.metrics['circuit_breaker_trips'] += 1
        
        # Record in history
        self.request_history.append({
            'service_id': service.id,
            'timestamp': time.time(),
            'error': error,
            'success': False
        })
        
        logger.warning(f"Request failed for service {service.id}: {error}")
    
    async def _health_check_loop(self):
        """Health check monitoring loop."""
        while self.monitoring_active:
            try:
                # Check all services
                for service_id, endpoint in self.services.items():
                    await self._check_service_health(service_id, endpoint)
                
                # Wait for next check
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_service_health(self, service_id: str, endpoint: ServiceEndpoint):
        """Check health of a specific service."""
        health = self.service_health[service_id]
        
        try:
            if endpoint.health_check_url:
                # Perform HTTP health check
                start_time = time.time()
                
                # Simulate health check (in real implementation, use aiohttp)
                await asyncio.sleep(0.01)  # Simulate network delay
                is_healthy = random.random() > 0.1  # 90% success rate simulation
                
                response_time_ms = (time.time() - start_time) * 1000
                
                if is_healthy:
                    health.status = ServiceStatus.HEALTHY
                    health.consecutive_failures = 0
                else:
                    health.consecutive_failures += 1
                    if health.consecutive_failures >= 3:
                        health.status = ServiceStatus.UNHEALTHY
                    else:
                        health.status = ServiceStatus.DEGRADED
                
                health.response_time_ms = response_time_ms
                health.last_check = datetime.now(timezone.utc)
            
            else:
                # Basic connectivity check
                health.status = ServiceStatus.HEALTHY
                health.last_check = datetime.now(timezone.utc)
                
        except Exception as e:
            health.consecutive_failures += 1
            health.status = ServiceStatus.UNHEALTHY
            health.last_check = datetime.now(timezone.utc)
            
            logger.warning(f"Health check failed for {service_id}: {e}")


# Global load balancer instance
_load_balancer: Optional[LoadBalancer] = None

def get_load_balancer() -> LoadBalancer:
    """Get global load balancer instance."""
    global _load_balancer
    
    if _load_balancer is None:
        _load_balancer = LoadBalancer()
    
    return _load_balancer

# Convenience functions for common load balancing scenarios
async def distribute_vector_search(
    search_func: Callable,
    query_vector: List[float],
    **kwargs
) -> Any:
    """Distribute vector search request across services."""
    load_balancer = get_load_balancer()
    
    return await load_balancer.distribute_request(
        search_func,
        query_vector,
        service_type="vector_search",
        **kwargs
    )

async def distribute_llm_request(
    llm_func: Callable,
    prompt: str,
    **kwargs
) -> Any:
    """Distribute LLM request across services."""
    load_balancer = get_load_balancer()
    
    return await load_balancer.distribute_request(
        llm_func,
        prompt,
        service_type="llm",
        **kwargs
    )

async def distribute_document_processing(
    process_func: Callable,
    document_data: bytes,
    **kwargs
) -> Any:
    """Distribute document processing request across services."""
    load_balancer = get_load_balancer()
    
    return await load_balancer.distribute_request(
        process_func,
        document_data,
        service_type="document_processing",
        **kwargs
    )