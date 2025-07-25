#!/usr/bin/env python3
"""
Simple test script to verify the health check system works independently.
"""

import asyncio
import json
from utils.monitoring.health_check import HealthChecker

async def main():
    """Test the health check system."""
    print("Testing Health Check System...")
    
    try:
        checker = HealthChecker()
        health_status = await checker.run_all_checks()
        
        print(f"Overall Status: {health_status['status'].value}")
        print(f"Timestamp: {health_status['timestamp']}")
        print("\nComponents:")
        
        for name, component in health_status['components'].items():
            print(f"\n  {name}:")
            print(f"    Status: {component['status'].value}")
            print(f"    Details: {json.dumps(component['details'], indent=6)}")
            
        return health_status
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(main())
    if result:
        print("\n✅ Health check system is working!")
    else:
        print("\n❌ Health check system failed!")