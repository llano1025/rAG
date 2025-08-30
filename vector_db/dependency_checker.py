"""
Dependency checker for vector_db module.
Helps diagnose missing dependencies and provides installation instructions.
"""

import logging
import sys
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DependencyChecker:
    """Check and report on vector database dependencies."""
    
    REQUIRED_PACKAGES = {
        'numpy': {
            'import_name': 'numpy',
            'pip_name': 'numpy>=1.24.0',
            'description': 'Required for vector operations and array processing'
        },
        'transformers': {
            'import_name': 'transformers',
            'pip_name': 'transformers>=4.35.0',
            'description': 'Required for text processing and tokenization'
        },
        'sentence_transformers': {
            'import_name': 'sentence_transformers',
            'pip_name': 'sentence-transformers>=2.2.0',
            'description': 'Required for generating embeddings'
        },
        'qdrant_client': {
            'import_name': 'qdrant_client',
            'pip_name': 'qdrant-client>=1.6.0',
            'description': 'Required for Qdrant vector database operations'
        },
        'scipy': {
            'import_name': 'scipy',
            'pip_name': 'scipy>=1.11.0',
            'description': 'Required for advanced vector operations'
        },
        'tokenizers': {
            'import_name': 'tokenizers',
            'pip_name': 'tokenizers>=0.15.0',
            'description': 'Required for text tokenization in chunking'
        }
    }
    
    OPTIONAL_PACKAGES = {
        'torch': {
            'import_name': 'torch',
            'pip_name': 'torch>=2.0.0',
            'description': 'Optional: Enhanced model support'
        },
        'spacy': {
            'import_name': 'spacy',
            'pip_name': 'spacy>=3.7.0',
            'description': 'Optional: Advanced NLP processing'
        }
    }
    
    def __init__(self):
        self.results = {}
        
    def check_package(self, package_name: str, import_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a package is available."""
        try:
            __import__(import_name)
            return True, None
        except ImportError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def check_all_dependencies(self) -> Dict[str, Dict]:
        """Check all required and optional dependencies."""
        results = {
            'required': {},
            'optional': {},
            'summary': {
                'required_available': 0,
                'required_total': len(self.REQUIRED_PACKAGES),
                'optional_available': 0,
                'optional_total': len(self.OPTIONAL_PACKAGES),
                'all_required_met': True
            }
        }
        
        # Check required packages
        for pkg_name, pkg_info in self.REQUIRED_PACKAGES.items():
            available, error = self.check_package(pkg_name, pkg_info['import_name'])
            results['required'][pkg_name] = {
                'available': available,
                'error': error,
                'pip_name': pkg_info['pip_name'],
                'description': pkg_info['description']
            }
            
            if available:
                results['summary']['required_available'] += 1
            else:
                results['summary']['all_required_met'] = False
        
        # Check optional packages
        for pkg_name, pkg_info in self.OPTIONAL_PACKAGES.items():
            available, error = self.check_package(pkg_name, pkg_info['import_name'])
            results['optional'][pkg_name] = {
                'available': available,
                'error': error,
                'pip_name': pkg_info['pip_name'],
                'description': pkg_info['description']
            }
            
            if available:
                results['summary']['optional_available'] += 1
        
        self.results = results
        return results
    
    def print_report(self, verbose: bool = False):
        """Print a formatted dependency report."""
        if not self.results:
            self.check_all_dependencies()
        
        print("=" * 60)
        print("VECTOR DATABASE DEPENDENCY REPORT")
        print("=" * 60)
        
        # Summary
        summary = self.results['summary']
        print(f"\nSummary:")
        print(f"  Required packages: {summary['required_available']}/{summary['required_total']}")
        print(f"  Optional packages: {summary['optional_available']}/{summary['optional_total']}")
        print(f"  All required met: {'✓ YES' if summary['all_required_met'] else '✗ NO'}")
        
        # Required packages
        print(f"\nRequired Packages:")
        print("-" * 40)
        missing_required = []
        
        for pkg_name, pkg_info in self.results['required'].items():
            status = "✓ Available" if pkg_info['available'] else "✗ Missing"
            print(f"  {pkg_name:20} {status}")
            
            if verbose and not pkg_info['available']:
                print(f"    Error: {pkg_info['error']}")
                print(f"    Install: pip install {pkg_info['pip_name']}")
                print(f"    Purpose: {pkg_info['description']}")
            
            if not pkg_info['available']:
                missing_required.append(pkg_info['pip_name'])
        
        # Optional packages
        print(f"\nOptional Packages:")
        print("-" * 40)
        missing_optional = []
        
        for pkg_name, pkg_info in self.results['optional'].items():
            status = "✓ Available" if pkg_info['available'] else "✗ Missing"
            print(f"  {pkg_name:20} {status}")
            
            if verbose and not pkg_info['available']:
                print(f"    Error: {pkg_info['error']}")
                print(f"    Install: pip install {pkg_info['pip_name']}")
                print(f"    Purpose: {pkg_info['description']}")
            
            if not pkg_info['available']:
                missing_optional.append(pkg_info['pip_name'])
        
        # Installation instructions
        if missing_required or missing_optional:
            print(f"\nInstallation Instructions:")
            print("-" * 40)
            
            if missing_required:
                print("Install required packages:")
                print(f"pip install {' '.join(missing_required)}")
            
            if missing_optional:
                print("\nInstall optional packages (recommended):")
                print(f"pip install {' '.join(missing_optional)}")
            
            print("\nOr install all at once:")
            print("pip install -r requirements.txt")
        
        print("=" * 60)
    
    def get_missing_packages(self) -> Tuple[List[str], List[str]]:
        """Get lists of missing required and optional packages."""
        if not self.results:
            self.check_all_dependencies()
        
        missing_required = []
        missing_optional = []
        
        for pkg_name, pkg_info in self.results['required'].items():
            if not pkg_info['available']:
                missing_required.append(pkg_info['pip_name'])
        
        for pkg_name, pkg_info in self.results['optional'].items():
            if not pkg_info['available']:
                missing_optional.append(pkg_info['pip_name'])
        
        return missing_required, missing_optional
    
    def is_vector_db_ready(self) -> bool:
        """Check if the vector database is ready to use."""
        if not self.results:
            self.check_all_dependencies()
        
        return self.results['summary']['all_required_met']


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check vector database dependencies')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed error messages and installation instructions')
    
    args = parser.parse_args()
    
    checker = DependencyChecker()
    checker.print_report(verbose=args.verbose)
    
    if not checker.is_vector_db_ready():
        sys.exit(1)  # Exit with error code if dependencies are missing


if __name__ == "__main__":
    main()