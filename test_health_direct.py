#!/usr/bin/env python3
"""
Direct test of health check functionality.
"""

import asyncio
import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import health checker directly
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
        # Write result to file for frontend to test
        with open('/tmp/health_test.json', 'w') as f:
            # Convert enum values to strings for JSON serialization
            serializable_result = {}
            for key, value in result.items():
                if key == 'components':
                    serializable_result[key] = {}
                    for comp_name, comp_data in value.items():
                        serializable_result[key][comp_name] = {
                            'status': comp_data['status'].value,
                            'details': comp_data['details']
                        }
                elif hasattr(value, 'value'):
                    serializable_result[key] = value.value
                else:
                    serializable_result[key] = value
            
            json.dump(serializable_result, f, indent=2)
        print("Health data written to /tmp/health_test.json")
    else:
        print("\n❌ Health check system failed!")