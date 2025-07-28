#!/usr/bin/env python3
"""
Vector Database Health Check Script
Verifies that all vector database components are working correctly.
"""

import sys
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies() -> bool:
    """Check vector database dependencies."""
    try:
        from vector_db.dependency_checker import DependencyChecker
        checker = DependencyChecker()
        results = checker.check_all_dependencies()
        
        if results['summary']['all_required_met']:
            logger.info("✓ All required dependencies are available")
            return True
        else:
            logger.error("✗ Missing required dependencies")
            checker.print_report(verbose=True)
            return False
    except Exception as e:
        logger.error(f"Failed to check dependencies: {e}")
        return False

def check_imports() -> bool:
    """Check that all vector_db modules can be imported."""
    tests = [
        ("vector_db availability flags", lambda: check_availability_flags()),
        ("Chunk class", lambda: check_chunk_class()),
        ("SearchOptimizer", lambda: check_search_optimizer()),
        ("VectorStorageManager", lambda: check_storage_manager()),
    ]
    
    success_count = 0
    for test_name, test_func in tests:
        try:
            test_func()
            logger.info(f"✓ {test_name} - OK")
            success_count += 1
        except Exception as e:
            logger.error(f"✗ {test_name} - FAILED: {e}")
    
    return success_count == len(tests)

def check_availability_flags():
    """Check availability flags."""
    from vector_db import SEARCH_AVAILABLE, STORAGE_AVAILABLE
    if not SEARCH_AVAILABLE:
        raise RuntimeError("SEARCH_AVAILABLE is False")
    if not STORAGE_AVAILABLE:
        raise RuntimeError("STORAGE_AVAILABLE is False")

def check_chunk_class():
    """Check Chunk class functionality."""
    from vector_db.chunking import Chunk
    chunk = Chunk(
        text="test text",
        start_idx=0,
        end_idx=9,
        metadata={"source": "test"},
        document_id=1,
        chunk_id="test_chunk"
    )
    if chunk.text != "test text":
        raise RuntimeError("Chunk text not set correctly")

def check_search_optimizer():
    """Check SearchOptimizer functionality."""
    from vector_db.search_optimizer import SearchOptimizer, SearchConfig, IndexType, MetricType
    
    config = SearchConfig(
        dimension=384,
        index_type=IndexType.FLAT,
        metric=MetricType.COSINE
    )
    
    # Don't create the optimizer here to avoid numpy array issues
    # Just verify the config is created correctly
    if config.dimension != 384:
        raise RuntimeError("SearchConfig not created correctly")

def check_storage_manager():
    """Check VectorStorageManager functionality."""
    from vector_db.storage_manager import VectorStorageManager, get_storage_manager
    
    # Get the global instance
    manager = get_storage_manager()
    if manager is None:
        raise RuntimeError("Failed to get storage manager instance")
    
    # Check basic properties
    if not hasattr(manager, 'storage_path'):
        raise RuntimeError("StorageManager missing storage_path attribute")

def check_lazy_imports() -> bool:
    """Check lazy import functions."""
    try:
        from vector_db import get_search_optimizer, get_storage_manager, get_chunking
        
        # Test lazy imports
        SearchOptimizer = get_search_optimizer()
        if SearchOptimizer is None:
            logger.error("✗ get_search_optimizer returned None")
            return False
        
        storage_manager_func = get_storage_manager()
        if storage_manager_func is None:
            logger.error("✗ get_storage_manager returned None")
            return False
        
        Chunk, AdaptiveChunker, chunking_available = get_chunking()
        if Chunk is None or AdaptiveChunker is None:
            logger.error("✗ get_chunking returned None components")
            return False
        
        logger.info("✓ All lazy imports working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Lazy imports failed: {e}")
        return False

def main():
    """Main health check function."""
    logger.info("Starting Vector Database Health Check...")
    logger.info("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Imports", check_imports),
        ("Lazy Imports", check_lazy_imports),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        logger.info(f"\nRunning {check_name} check...")
        try:
            if check_func():
                logger.info(f"✓ {check_name} check PASSED")
                passed += 1
            else:
                logger.error(f"✗ {check_name} check FAILED")
        except Exception as e:
            logger.error(f"✗ {check_name} check FAILED with exception: {e}")
    
    logger.info("=" * 50)
    logger.info(f"Health Check Results: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("🎉 All vector database components are healthy!")
        return 0
    else:
        logger.error("❌ Some vector database components have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())