"""
SQLAlchemy base class and common utilities.
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, Boolean
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps to models."""
    
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

class SoftDeleteMixin:
    """Mixin to add soft delete capability to models."""
    
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)