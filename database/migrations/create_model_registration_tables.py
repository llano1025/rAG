#!/usr/bin/env python3
"""
Database migration script to create model registration tables.

This script creates the necessary tables for dynamic LLM model registration:
- registered_models: Store user-registered model configurations
- model_tests: Store model testing and validation results

Run this script to create the tables in PostgreSQL database.
"""

import asyncio
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from database.models import Base, RegisteredModel, ModelTest
from database.connection import get_database_url

logger = logging.getLogger(__name__)

# SQL statements to create the new tables
CREATE_REGISTERED_MODELS_TABLE = """
CREATE TABLE IF NOT EXISTS registered_models (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    display_name VARCHAR(200),
    description TEXT,
    model_name VARCHAR(200) NOT NULL,
    provider VARCHAR(50) NOT NULL CHECK (provider IN ('openai', 'gemini', 'anthropic', 'ollama', 'lmstudio')),
    config_json TEXT NOT NULL,
    provider_config_json TEXT,
    version VARCHAR(50),
    context_window INTEGER,
    max_tokens INTEGER,
    supports_streaming BOOLEAN NOT NULL DEFAULT TRUE,
    supports_embeddings BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    fallback_priority INTEGER,
    usage_count INTEGER NOT NULL DEFAULT 0,
    last_used TIMESTAMP WITH TIME ZONE,
    total_tokens_used INTEGER NOT NULL DEFAULT 0,
    estimated_cost FLOAT NOT NULL DEFAULT 0.0,
    average_response_time FLOAT,
    success_rate FLOAT NOT NULL DEFAULT 100.0,
    error_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);
"""

CREATE_MODEL_TESTS_TABLE = """
CREATE TABLE IF NOT EXISTS model_tests (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES registered_models(id) ON DELETE CASCADE,
    test_type VARCHAR(50) NOT NULL,
    test_prompt TEXT,
    test_parameters TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'passed', 'failed', 'timeout')),
    response_text TEXT,
    response_time_ms FLOAT,
    tokens_used INTEGER,
    estimated_cost FLOAT,
    error_message TEXT,
    error_code VARCHAR(50),
    error_details TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    timeout_seconds INTEGER NOT NULL DEFAULT 30,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

# Create indexes for performance
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_registered_models_user_provider ON registered_models(user_id, provider);",
    "CREATE INDEX IF NOT EXISTS idx_registered_models_active ON registered_models(is_active);",
    "CREATE INDEX IF NOT EXISTS idx_registered_models_public ON registered_models(is_public);",
    "CREATE INDEX IF NOT EXISTS idx_registered_models_name ON registered_models(name);",
    "CREATE INDEX IF NOT EXISTS idx_model_tests_model_status ON model_tests(model_id, status);",
    "CREATE INDEX IF NOT EXISTS idx_model_tests_type ON model_tests(test_type);",
    "CREATE INDEX IF NOT EXISTS idx_model_tests_created ON model_tests(created_at);",
]

# Add update trigger for updated_at timestamp
CREATE_UPDATE_TRIGGERS = [
    """
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    """,
    """
    CREATE TRIGGER update_registered_models_updated_at 
    BEFORE UPDATE ON registered_models 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """,
    """
    CREATE TRIGGER update_model_tests_updated_at 
    BEFORE UPDATE ON model_tests 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """
]

def create_tables():
    """Create the model registration tables in PostgreSQL."""
    try:
        # Get database connection
        database_url = get_database_url()
        
        # Create engine
        engine = create_engine(database_url)
        
        print("Connecting to PostgreSQL database...")
        
        with engine.connect() as connection:
            print("Connected to database successfully")
            
            # Create tables
            print("Creating registered_models table...")
            connection.execute(text(CREATE_REGISTERED_MODELS_TABLE))
            connection.commit()
            print("registered_models table created")
            
            print("Creating model_tests table...")
            connection.execute(text(CREATE_MODEL_TESTS_TABLE))
            connection.commit()
            print("model_tests table created")
            
            # Create indexes
            print("Creating database indexes...")
            for index_sql in CREATE_INDEXES:
                connection.execute(text(index_sql))
            connection.commit()
            print("Database indexes created")
            
            # Create update triggers
            print("Creating update triggers...")
            for trigger_sql in CREATE_UPDATE_TRIGGERS:
                connection.execute(text(trigger_sql))
            connection.commit()
            print("Update triggers created")
            
            print("Model registration tables created successfully!")
            
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        raise

def verify_tables():
    """Verify that the tables were created correctly."""
    try:
        database_url = get_database_url()
        engine = create_engine(database_url)
        
        with engine.connect() as connection:
            # Check if tables exist
            result = connection.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('registered_models', 'model_tests')
                ORDER BY table_name;
            """))
            
            tables = [row[0] for row in result.fetchall()]
            
            print(f"Found tables: {tables}")
            
            if 'registered_models' in tables and 'model_tests' in tables:
                print("All model registration tables verified successfully")
                
                # Check table structure
                for table_name in ['registered_models', 'model_tests']:
                    result = connection.execute(text(f"""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}'
                        ORDER BY ordinal_position;
                    """))
                    
                    columns = result.fetchall()
                    print(f"{table_name} has {len(columns)} columns")
                
                return True
            else:
                print("Some tables are missing")
                return False
                
    except Exception as e:
        print(f"Error verifying tables: {str(e)}")
        return False

def main():
    """Main migration function."""
    print("Starting model registration tables migration...")
    print("=" * 60)
    
    try:
        # Create tables
        create_tables()
        
        print("\n" + "=" * 60)
        print("Verifying table creation...")
        
        # Verify tables
        if verify_tables():
            print("\nMigration completed successfully!")
            print("The following tables are now available:")
            print("  • registered_models - Store user-registered LLM models")
            print("  • model_tests - Store model testing and validation results")
        else:
            print("\nMigration verification failed!")
            return 1
            
    except Exception as e:
        print(f"\nMigration failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())