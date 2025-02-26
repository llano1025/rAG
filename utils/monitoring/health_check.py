from typing import Dict, Any, List, Callable
import asyncio
import time
from enum import Enum
import logging
from datetime import datetime
import aiohttp
import psutil

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ComponentHealth:
    def __init__(self, name: str):
        self.name = name
        self.status = HealthStatus.HEALTHY
        self.last_check = datetime.now()
        self.details = {}
        self.error_message = None

class HealthChecker:
    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {}
        self.checks: Dict[str, Callable] = {}
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default system health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_usage", self._check_disk_usage)

    def register_check(self, name: str, check_func: Callable):
        """Register a new health check."""
        self.checks[name] = check_func
        self.components[name] = ComponentHealth(name)

    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system CPU and memory usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        status = HealthStatus.HEALTHY
        details = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024 ** 3)
        }
        
        if cpu_percent > 90 or memory.percent > 90:
            status = HealthStatus.DEGRADED
            if cpu_percent > 95 or memory.percent > 95:
                status = HealthStatus.UNHEALTHY
        
        return {
            "status": status,
            "details": details
        }

    async def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage."""
        disk = psutil.disk_usage("/")
        
        status = HealthStatus.HEALTHY
        details = {
            "total_gb": disk.total / (1024 ** 3),
            "used_gb": disk.used / (1024 ** 3),
            "free_gb": disk.free / (1024 ** 3),
            "usage_percent": disk.percent
        }
        
        if disk.percent > 85:
            status = HealthStatus.DEGRADED
            if disk.percent > 95:
                status = HealthStatus.UNHEALTHY
        
        return {
            "status": status,
            "details": details
        }

    async def check_external_service(self, name: str, url: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Check external service health."""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(url, timeout=timeout) as response:
                    response_time = time.time() - start_time
                    
                    status = HealthStatus.HEALTHY
                    if response.status != 200:
                        status = HealthStatus.UNHEALTHY
                    elif response_time > timeout * 0.8:
                        status = HealthStatus.DEGRADED
                        
                    return {
                        "status": status,
                        "details": {
                            "response_time": response_time,
                            "status_code": response.status
                        }
                    }
        except Exception as e:
            logger.error(f"Health check failed for {name}: {str(e)}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "details": {"error": str(e)}
            }

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY

        for name, check_func in self.checks.items():
            try:
                result = await check_func()
                self.components[name].status = result["status"]
                self.components[name].last_check = datetime.now()
                self.components[name].details = result["details"]
                self.components[name].error_message = None
                
                if result["status"] == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result["status"] == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
                results[name] = result
                
            except Exception as e:
                logger.error(f"Health check failed for {name}: {str(e)}")
                self.components[name].status = HealthStatus.UNHEALTHY
                self.components[name].error_message = str(e)
                overall_status = HealthStatus.UNHEALTHY
                results[name] = {
                    "status": HealthStatus.UNHEALTHY,
                    "details": {"error": str(e)}
                }

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": results
        }

# Example usage:
# if __name__ == "__main__":
#     async def main():
#         checker = HealthChecker()
        
#         # Register custom service check
#         checker.register_check(
#             "vector_db",
#             lambda: checker.check_external_service("vector_db", "http://localhost:6333/health")
#         )
        
#         # Run all health checks
#         results = await checker.run_all_checks()
#         print(f"System Health Status: {results['status'].value}")
#         for component, result in results['components'].items():
#             print(f"{component}: {result['status'].value}")
#             print(f"Details: {result['details']}")

#     asyncio.run(main())