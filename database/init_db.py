"""
Database initialization script.
Creates tables and populates initial data for the RAG system.
"""

import asyncio
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from .connection import engine, SessionLocal, create_tables
from .models import (
    User, UserRole, Permission, APIKey, 
    UserRoleEnum, PermissionEnum
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)

def create_default_permissions(db: Session):
    """Create default permissions."""
    permissions_data = [
        # Document permissions
        (PermissionEnum.READ_DOCUMENTS, "Read Documents", "View and read documents", "documents", "read"),
        (PermissionEnum.WRITE_DOCUMENTS, "Write Documents", "Create and edit documents", "documents", "write"),
        (PermissionEnum.DELETE_DOCUMENTS, "Delete Documents", "Delete documents", "documents", "delete"),
        
        # Folder permissions
        (PermissionEnum.READ_FOLDERS, "Read Folders", "View folders and their contents", "folders", "read"),
        (PermissionEnum.WRITE_FOLDERS, "Write Folders", "Create and edit folders", "folders", "write"),
        (PermissionEnum.DELETE_FOLDERS, "Delete Folders", "Delete folders", "folders", "delete"),
        
        # Tag permissions
        (PermissionEnum.READ_TAGS, "Read Tags", "View tags", "tags", "read"),
        (PermissionEnum.WRITE_TAGS, "Write Tags", "Create and edit tags", "tags", "write"),
        (PermissionEnum.DELETE_TAGS, "Delete Tags", "Delete tags", "tags", "delete"),
        
        # User management permissions
        (PermissionEnum.READ_USERS, "Read Users", "View user information", "users", "read"),
        (PermissionEnum.WRITE_USERS, "Write Users", "Create and edit user accounts", "users", "write"),
        (PermissionEnum.DELETE_USERS, "Delete Users", "Delete user accounts", "users", "delete"),
        
        # Analytics permissions
        (PermissionEnum.READ_ANALYTICS, "Read Analytics", "View system analytics and reports", "analytics", "read"),
        
        # API key permissions
        (PermissionEnum.READ_API_KEYS, "Read API Keys", "View API keys", "api_keys", "read"),
        (PermissionEnum.WRITE_API_KEYS, "Write API Keys", "Create and edit API keys", "api_keys", "write"),
        (PermissionEnum.DELETE_API_KEYS, "Delete API Keys", "Delete API keys", "api_keys", "delete"),
        
        # System permissions
        (PermissionEnum.SYSTEM_ADMIN, "System Admin", "Full system administration access", "system", "admin"),
    ]
    
    for perm_name, display_name, description, resource, action in permissions_data:
        permission = db.query(Permission).filter(Permission.name == perm_name).first()
        if not permission:
            permission = Permission(
                name=perm_name,
                display_name=display_name,
                description=description,
                resource=resource,
                action=action
            )
            db.add(permission)
    
    db.commit()

def create_default_roles(db: Session):
    """Create default roles with appropriate permissions."""
    # Get all permissions
    permissions = {perm.name: perm for perm in db.query(Permission).all()}
    
    # Admin role - full access
    admin_role = db.query(UserRole).filter(UserRole.name == UserRoleEnum.ADMIN).first()
    if not admin_role:
        admin_role = UserRole(
            name=UserRoleEnum.ADMIN,
            display_name="Administrator",
            description="Full system access with all permissions"
        )
        admin_role.permissions = list(permissions.values())
        db.add(admin_role)
    
    # User role - standard user permissions
    user_role = db.query(UserRole).filter(UserRole.name == UserRoleEnum.USER).first()
    if not user_role:
        user_role = UserRole(
            name=UserRoleEnum.USER,
            display_name="User",
            description="Standard user with read/write access to documents and folders"
        )
        user_permissions = [
            permissions[PermissionEnum.READ_DOCUMENTS],
            permissions[PermissionEnum.WRITE_DOCUMENTS],
            permissions[PermissionEnum.READ_FOLDERS],
            permissions[PermissionEnum.WRITE_FOLDERS],
            permissions[PermissionEnum.READ_TAGS],
            permissions[PermissionEnum.WRITE_TAGS],
            permissions[PermissionEnum.READ_API_KEYS],
            permissions[PermissionEnum.WRITE_API_KEYS],
        ]
        user_role.permissions = user_permissions
        db.add(user_role)
    
    # Viewer role - read-only access
    viewer_role = db.query(UserRole).filter(UserRole.name == UserRoleEnum.VIEWER).first()
    if not viewer_role:
        viewer_role = UserRole(
            name=UserRoleEnum.VIEWER,
            display_name="Viewer",
            description="Read-only access to documents and folders"
        )
        viewer_permissions = [
            permissions[PermissionEnum.READ_DOCUMENTS],
            permissions[PermissionEnum.READ_FOLDERS],
            permissions[PermissionEnum.READ_TAGS],
        ]
        viewer_role.permissions = viewer_permissions
        db.add(viewer_role)
    
    db.commit()

def create_admin_user(db: Session, email: str = "admin@example.com", password: str = "admin123"):
    """Create default admin user if it doesn't exist."""
    admin_user = db.query(User).filter(User.email == email).first()
    if not admin_user:
        admin_role = db.query(UserRole).filter(UserRole.name == UserRoleEnum.ADMIN).first()
        if not admin_role:
            raise ValueError("Admin role not found. Run create_default_roles first.")
        
        admin_user = User(
            email=email,
            username="admin",
            full_name="System Administrator",
            hashed_password=hash_password(password),
            is_active=True,
            is_verified=True,
            is_superuser=True,
            roles=[admin_role]
        )
        db.add(admin_user)
        db.commit()
        print(f"Created admin user: {email}")
    else:
        print(f"Admin user already exists: {email}")

def init_database():
    """Initialize database with tables and default data."""
    print("Creating database tables...")
    create_tables()
    
    print("Initializing default data...")
    db = SessionLocal()
    try:
        create_default_permissions(db)
        print("Created default permissions")
        
        create_default_roles(db)
        print("Created default roles")
        
        create_admin_user(db)
        print("Created admin user")
        
        print("Database initialization completed successfully!")
        
    except Exception as e:
        print(f"Error during database initialization: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    init_database()