#!/usr/bin/env python3
"""
Quick development environment test script.
"""

import sys
import requests
import time
import subprocess
from pathlib import Path

def test_dependencies():
    """Test if core dependencies are available."""
    print("🔄 Testing dependencies...")
    
    missing = []
    try:
        import fastapi, uvicorn, pydantic, sqlalchemy
        print("✅ Core FastAPI dependencies")
    except ImportError as e:
        missing.append(f"FastAPI: {e}")
    
    try:
        from config import get_settings
        settings = get_settings()
        print(f"✅ Configuration: {settings.APP_NAME}")
    except Exception as e:
        missing.append(f"Config: {e}")
    
    try:
        from database.connection import SessionLocal
        db = SessionLocal()
        db.close()
        print("✅ Database connection")
    except Exception as e:
        missing.append(f"Database: {e}")
    
    if missing:
        print("❌ Missing dependencies:")
        for m in missing:
            print(f"   - {m}")
        return False
    
    return True

def test_server():
    """Test if development server can start."""
    print("\n🔄 Testing development server...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server responding: {data['status']}")
            print(f"   Database: {data.get('database', 'unknown')}")
            print(f"   Users: {data.get('users', 'unknown')}")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server not responding (not started?)")
        return False
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

def main():
    """Run development environment tests."""
    print("🚀 RAG Development Environment Test\n")
    
    # Test dependencies
    if not test_dependencies():
        print("\n❌ Dependencies test failed. Please install missing packages.")
        return False
    
    # Check if server is already running
    if test_server():
        print("\n✅ Development server is already running!")
        print("   Visit: http://localhost:8000/docs")
        return True
    
    print("\n📋 Development Environment Status:")
    print("✅ Dependencies: Ready")
    print("✅ Database: Initialized")
    print("✅ Configuration: Loaded")
    print("\n🚀 To start development server:")
    print("   python3 dev_server.py")
    print("\n🔧 Alternative minimal start:")
    print("   uvicorn dev_server:create_dev_app --reload --host 0.0.0.0 --port 8000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)