#!/usr/bin/env python3
"""
Simple mock health server to test the analytics frontend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def mock_health():
    """Mock health endpoint with CPU and disk usage data."""
    return {
        "status": "healthy",
        "timestamp": "2025-07-25T13:30:00Z",
        "components": {
            "system_resources": {
                "status": "healthy",
                "details": {
                    "cpu_usage_percent": 45.2,
                    "memory_usage_percent": 62.8,
                    "memory_available_gb": 3.2
                }
            },
            "disk_usage": {
                "status": "healthy", 
                "details": {
                    "total_gb": 500.0,
                    "used_gb": 320.5,
                    "free_gb": 179.5,
                    "usage_percent": 64.1
                }
            }
        }
    }

@app.get("/api/analytics/usage-stats")
async def mock_usage_stats():
    """Mock usage stats for analytics."""
    return {
        "total_documents": 1250,
        "total_searches": 8940,
        "active_users": 45,
        "storage_used": 1024000000,  # 1GB in bytes
        "upload_trends": [
            {"date": "2025-07-20", "count": 12},
            {"date": "2025-07-21", "count": 18},
            {"date": "2025-07-22", "count": 9},
            {"date": "2025-07-23", "count": 15},
            {"date": "2025-07-24", "count": 22},
            {"date": "2025-07-25", "count": 7}
        ],
        "search_trends": [
            {"date": "2025-07-20", "count": 145},
            {"date": "2025-07-21", "count": 198},
            {"date": "2025-07-22", "count": 132},
            {"date": "2025-07-23", "count": 167},
            {"date": "2025-07-24", "count": 203},
            {"date": "2025-07-25", "count": 89}
        ]
    }

if __name__ == "__main__":
    print("🧪 Starting Mock Health Server on http://localhost:8000")
    print("📊 Analytics endpoint: http://localhost:8000/api/analytics/usage-stats")
    print("💚 Health endpoint: http://localhost:8000/api/health")
    uvicorn.run("mock_health_server:app", host="0.0.0.0", port=8000, reload=False)