# llm/config/__init__.py

"""
Configuration management for LLM system.

This module provides configuration templates, provider registry,
and utilities for managing LLM model configurations.
"""

from .provider_registry import ProviderRegistry, get_provider_registry

__all__ = [
    'ProviderRegistry',
    'get_provider_registry'
]