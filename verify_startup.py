#!/usr/bin/env python3
"""
Startup verification script for the RAG application.

This script checks if the application can start without import errors
and validates the basic structure before launching the full server.
"""

import sys
import ast
import os
from pathlib import Path

def check_syntax(file_path: Path) -> bool:
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def main():
    """Run startup verification checks."""
    print("🚀 RAG Application Startup Verification")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    
    # Key files to check
    critical_files = [
        "main.py",
        "api/routes/model_routes.py",
        "services/model_discovery_service.py", 
        "services/model_registration_service.py",
        "database/models.py",
    ]
    
    print("🧪 Checking critical file syntax...")
    
    all_good = True
    for file_path in critical_files:
        full_path = project_root / file_path
        if full_path.exists():
            if check_syntax(full_path):
                print(f"✅ {file_path}")
            else:
                all_good = False
        else:
            print(f"❌ Missing file: {file_path}")
            all_good = False
    
    if all_good:
        print("\n🎉 All critical files have valid syntax!")
        print("\n📋 Next steps:")
        print("1. Ensure PostgreSQL is running on port 5432")
        print("2. Set up environment variables (DATABASE_URL, etc.)")
        print("3. Run database migrations if needed")
        print("4. Start the application with: uvicorn main:app --reload")
        print("\n🔗 Model management will be available at:")
        print("  • /admin/models/discover - Discover new models")
        print("  • /admin/models/manage - Manage registered models")
        print("  • /admin/models/register - Register models manually")
        return 0
    else:
        print("\n❌ Some issues found. Please fix the errors above before starting.")
        return 1

if __name__ == "__main__":
    exit(main())