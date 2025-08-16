#!/usr/bin/env python3
"""
Test script for the dynamic model registration system.

This script validates the core functionality without loading heavy dependencies.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_database_models():
    """Test database model definitions."""
    print("🧪 Testing database models...")
    
    try:
        from database.models import RegisteredModel, ModelTest, ModelProviderEnum, ModelTestStatusEnum
        
        # Test enum values
        providers = [e.value for e in ModelProviderEnum]
        expected_providers = ['openai', 'gemini', 'anthropic', 'ollama', 'lmstudio']
        
        assert all(p in providers for p in expected_providers), f"Missing providers: {set(expected_providers) - set(providers)}"
        
        statuses = [e.value for e in ModelTestStatusEnum]
        expected_statuses = ['pending', 'running', 'passed', 'failed', 'timeout']
        
        assert all(s in statuses for s in expected_statuses), f"Missing statuses: {set(expected_statuses) - set(statuses)}"
        
        print("✅ Database models validated")
        return True
        
    except Exception as e:
        print(f"❌ Database models test failed: {e}")
        return False

def test_model_config():
    """Test ModelConfig functionality."""
    print("🧪 Testing ModelConfig...")
    
    try:
        from llm.base.models import ModelConfig
        
        # Test basic config creation
        config = ModelConfig(
            model_name="test-model",
            max_tokens=1000,
            temperature=0.7,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        
        # Test serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['model_name'] == "test-model"
        assert config_dict['max_tokens'] == 1000
        
        # Test deserialization
        config2 = ModelConfig.from_dict(config_dict)
        assert config2.model_name == config.model_name
        assert config2.max_tokens == config.max_tokens
        
        print("✅ ModelConfig functionality validated")
        return True
        
    except Exception as e:
        print(f"❌ ModelConfig test failed: {e}")
        return False

def test_api_route_structure():
    """Test API route structure by examining the file."""
    print("🧪 Testing API route structure...")
    
    try:
        route_file = project_root / "api" / "routes" / "model_routes.py"
        
        if not route_file.exists():
            print(f"❌ Route file not found: {route_file}")
            return False
        
        content = route_file.read_text()
        
        # Check for expected endpoint patterns (flexible matching)
        expected_endpoints = [
            ("/providers", "GET"),
            ("/discover/{provider}", "POST"),
            ("/templates", "GET"),
            ("/test-connectivity/{provider}", "POST"),
            ("/register", "POST"),
            ("/registered", "GET"),
            ("/registered/{model_id}", "GET"),
            ("/registered/{model_id}", "PUT"),
            ("/registered/{model_id}", "DELETE"),
            ("/load-registered", "POST"),
            ("/sync", "POST"),
            ("/loaded", "GET"),
            ("/registered/{model_id}/reload", "POST"),
            ("/registered/{model_id}/test", "POST"),
            ("/registered/{model_id}/tests", "GET"),
        ]
        
        missing_endpoints = []
        for path, method in expected_endpoints:
            pattern = f"@router.{method.lower()}(\"{path}\""
            if pattern not in content:
                missing_endpoints.append(f"{method} {path}")
        
        if missing_endpoints:
            print(f"❌ Missing endpoints: {missing_endpoints}")
            return False
        
        print(f"✅ All {len(expected_endpoints)} API endpoints found")
        return True
        
    except Exception as e:
        print(f"❌ API route structure test failed: {e}")
        return False

def test_service_structure():
    """Test service file structure."""
    print("🧪 Testing service structure...")
    
    try:
        discovery_file = project_root / "services" / "model_discovery_service.py"
        registration_file = project_root / "services" / "model_registration_service.py"
        
        files_to_check = [
            (discovery_file, ["ModelDiscoveryService", "get_discovery_service"]),
            (registration_file, ["ModelRegistrationService", "get_registration_service"]),
        ]
        
        for file_path, expected_classes in files_to_check:
            if not file_path.exists():
                print(f"❌ Service file not found: {file_path}")
                return False
            
            content = file_path.read_text()
            
            for class_name in expected_classes:
                if class_name not in content:
                    print(f"❌ Missing class/function: {class_name} in {file_path.name}")
                    return False
        
        print("✅ Service structure validated")
        return True
        
    except Exception as e:
        print(f"❌ Service structure test failed: {e}")
        return False

def test_frontend_structure():
    """Test frontend file structure."""
    print("🧪 Testing frontend structure...")
    
    try:
        frontend_files = [
            project_root / "frontend" / "src" / "api" / "models.ts",
            project_root / "frontend" / "src" / "pages" / "admin" / "models" / "discover.tsx",
            project_root / "frontend" / "src" / "pages" / "admin" / "models" / "register.tsx",
            project_root / "frontend" / "src" / "pages" / "admin" / "models" / "manage.tsx",
        ]
        
        for file_path in frontend_files:
            if not file_path.exists():
                print(f"❌ Frontend file not found: {file_path}")
                return False
        
        # Check API types in models.ts
        models_content = frontend_files[0].read_text()
        expected_types = [
            "DiscoveredModel",
            "RegisteredModel",
            "Provider",
            "ModelTemplate",
            "ModelRegistrationRequest",
        ]
        
        for type_name in expected_types:
            if f"interface {type_name}" not in models_content:
                print(f"❌ Missing TypeScript interface: {type_name}")
                return False
        
        print("✅ Frontend structure validated")
        return True
        
    except Exception as e:
        print(f"❌ Frontend structure test failed: {e}")
        return False

def test_database_migration():
    """Test database migration structure."""
    print("🧪 Testing database migration...")
    
    try:
        migration_file = project_root / "database" / "migrations" / "create_model_registration_tables.py"
        
        if not migration_file.exists():
            print(f"❌ Migration file not found: {migration_file}")
            return False
        
        content = migration_file.read_text()
        
        expected_elements = [
            "CREATE_REGISTERED_MODELS_TABLE",
            "CREATE_MODEL_TESTS_TABLE", 
            "CREATE_INDEXES",
            "CREATE_UPDATE_TRIGGERS",
        ]
        
        for element in expected_elements:
            if element not in content:
                print(f"❌ Missing migration element: {element}")
                return False
        
        print("✅ Database migration structure validated")
        return True
        
    except Exception as e:
        print(f"❌ Database migration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Dynamic Model Registration System")
    print("=" * 60)
    
    tests = [
        test_database_models,
        test_model_config,
        test_api_route_structure,
        test_service_structure,
        test_frontend_structure,
        test_database_migration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The dynamic model registration system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())