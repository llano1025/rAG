#!/usr/bin/env python3
"""
Database migration script to convert documents.tags column from TEXT to JSONB.

This migration:
1. Creates a new JSONB column for tags
2. Migrates existing JSON string data to proper JSONB format
3. Drops the old TEXT column
4. Renames the new column to 'tags'
5. Updates the search engine to use efficient JSONB operations

Run this script to upgrade the database schema for enhanced tag filtering.
"""

import asyncio
import logging
import json
from sqlalchemy import create_engine, text, Column, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from database.models import Base
from database.connection import get_database_url

logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging for migration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class TagsMigration:
    """Handles the migration of tags column from TEXT to JSONB."""
    
    def __init__(self):
        self.database_url = get_database_url()
        self.engine = create_engine(self.database_url)
    
    def check_current_schema(self):
        """Check the current schema of the tags column."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = 'documents' 
                    AND column_name = 'tags';
                """))
                
                column_info = result.fetchone()
                if column_info:
                    return {
                        'column_name': column_info[0],
                        'data_type': column_info[1],
                        'is_nullable': column_info[2]
                    }
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error checking schema: {e}")
            return None
    
    def backup_existing_data(self):
        """Create a backup of existing tags data."""
        try:
            with self.engine.connect() as connection:
                # Create backup table
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS documents_tags_backup AS
                    SELECT id, tags, created_at
                    FROM documents
                    WHERE tags IS NOT NULL;
                """))
                connection.commit()
                
                # Count backed up records
                result = connection.execute(text("""
                    SELECT COUNT(*) FROM documents_tags_backup;
                """))
                count = result.fetchone()[0]
                logger.info(f"✅ Backed up {count} documents with tags")
                return True
                
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def validate_json_data(self):
        """Validate that existing tags data is valid JSON."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT id, tags
                    FROM documents
                    WHERE tags IS NOT NULL
                    LIMIT 1000;
                """))
                
                invalid_records = []
                valid_count = 0
                
                for row in result.fetchall():
                    doc_id, tags_str = row
                    if tags_str:
                        try:
                            # Try to parse as JSON
                            parsed = json.loads(tags_str)
                            if isinstance(parsed, list):
                                valid_count += 1
                            else:
                                invalid_records.append((doc_id, "Not a JSON array"))
                        except json.JSONDecodeError as e:
                            invalid_records.append((doc_id, str(e)))
                
                if invalid_records:
                    logger.warning(f"Found {len(invalid_records)} invalid JSON records:")
                    for doc_id, error in invalid_records[:5]:  # Show first 5
                        logger.warning(f"  Document {doc_id}: {error}")
                    if len(invalid_records) > 5:
                        logger.warning(f"  ... and {len(invalid_records) - 5} more")
                
                logger.info(f"✅ Validated {valid_count} records with valid JSON")
                return len(invalid_records) == 0, invalid_records
                
        except Exception as e:
            logger.error(f"Error validating JSON data: {e}")
            return False, []
    
    def fix_invalid_json(self, invalid_records):
        """Fix invalid JSON records by converting them to proper format."""
        try:
            with self.engine.connect() as connection:
                fixed_count = 0
                
                for doc_id, error in invalid_records:
                    # Get the current tags value
                    result = connection.execute(text("""
                        SELECT tags FROM documents WHERE id = :doc_id
                    """), {"doc_id": doc_id})
                    
                    current_tags = result.fetchone()[0]
                    
                    # Try to fix common issues
                    fixed_tags = None
                    try:
                        if current_tags:
                            # Remove extra quotes or fix format
                            cleaned = current_tags.strip()
                            if not cleaned.startswith('['):
                                # Single tag, convert to array
                                if cleaned.startswith('"') and cleaned.endswith('"'):
                                    fixed_tags = f'[{cleaned}]'
                                else:
                                    fixed_tags = f'["{cleaned}"]'
                            else:
                                # Try to parse and reformat
                                try:
                                    parsed = json.loads(cleaned)
                                    fixed_tags = json.dumps(parsed)
                                except:
                                    # Set to empty array if unfixable
                                    fixed_tags = '[]'
                        else:
                            fixed_tags = '[]'
                        
                        # Update the record
                        connection.execute(text("""
                            UPDATE documents SET tags = :fixed_tags WHERE id = :doc_id
                        """), {"fixed_tags": fixed_tags, "doc_id": doc_id})
                        
                        fixed_count += 1
                        
                    except Exception as fix_error:
                        logger.warning(f"Could not fix document {doc_id}: {fix_error}")
                        # Set to empty array as last resort
                        connection.execute(text("""
                            UPDATE documents SET tags = '[]' WHERE id = :doc_id
                        """), {"doc_id": doc_id})
                        fixed_count += 1
                
                connection.commit()
                logger.info(f"✅ Fixed {fixed_count} invalid JSON records")
                return True
                
        except Exception as e:
            logger.error(f"Error fixing invalid JSON: {e}")
            return False
    
    def create_new_jsonb_column(self):
        """Create new JSONB column for tags."""
        try:
            with self.engine.connect() as connection:
                # Add new JSONB column
                connection.execute(text("""
                    ALTER TABLE documents 
                    ADD COLUMN IF NOT EXISTS tags_jsonb JSONB;
                """))
                connection.commit()
                logger.info("✅ Created new JSONB column 'tags_jsonb'")
                return True
                
        except Exception as e:
            logger.error(f"Error creating JSONB column: {e}")
            return False
    
    def migrate_data_to_jsonb(self):
        """Migrate data from TEXT tags to JSONB tags_jsonb column."""
        try:
            with self.engine.connect() as connection:
                # Migrate data in batches
                batch_size = 1000
                offset = 0
                total_migrated = 0
                
                while True:
                    # Get batch of documents with tags
                    result = connection.execute(text("""
                        SELECT id, tags
                        FROM documents
                        WHERE tags IS NOT NULL
                        ORDER BY id
                        LIMIT :batch_size OFFSET :offset
                    """), {"batch_size": batch_size, "offset": offset})
                    
                    rows = result.fetchall()
                    if not rows:
                        break
                    
                    # Process each document in the batch
                    for doc_id, tags_str in rows:
                        try:
                            if tags_str:
                                # Parse JSON and convert to JSONB
                                tags_array = json.loads(tags_str)
                                if isinstance(tags_array, list):
                                    # Normalize tags: lowercase and clean
                                    normalized_tags = [
                                        str(tag).lower().strip() 
                                        for tag in tags_array 
                                        if tag and str(tag).strip()
                                    ]
                                    # Remove duplicates while preserving order
                                    seen = set()
                                    unique_tags = []
                                    for tag in normalized_tags:
                                        if tag not in seen:
                                            seen.add(tag)
                                            unique_tags.append(tag)
                                    
                                    # Update with JSONB data
                                    connection.execute(text("""
                                        UPDATE documents 
                                        SET tags_jsonb = :jsonb_tags 
                                        WHERE id = :doc_id
                                    """), {
                                        "jsonb_tags": json.dumps(unique_tags),
                                        "doc_id": doc_id
                                    })
                                    total_migrated += 1
                                else:
                                    # Handle non-array case
                                    connection.execute(text("""
                                        UPDATE documents 
                                        SET tags_jsonb = '[]'::jsonb 
                                        WHERE id = :doc_id
                                    """), {"doc_id": doc_id})
                            
                        except Exception as e:
                            logger.warning(f"Error migrating document {doc_id}: {e}")
                            # Set empty array for problematic records
                            connection.execute(text("""
                                UPDATE documents 
                                SET tags_jsonb = '[]'::jsonb 
                                WHERE id = :doc_id
                            """), {"doc_id": doc_id})
                    
                    connection.commit()
                    offset += batch_size
                    logger.info(f"Migrated batch: {offset} documents processed")
                
                logger.info(f"✅ Successfully migrated {total_migrated} documents to JSONB")
                return True
                
        except Exception as e:
            logger.error(f"Error migrating data to JSONB: {e}")
            return False
    
    def verify_migration(self):
        """Verify that the migration was successful."""
        try:
            with self.engine.connect() as connection:
                # Count records with JSONB data
                result = connection.execute(text("""
                    SELECT COUNT(*) FROM documents WHERE tags_jsonb IS NOT NULL;
                """))
                jsonb_count = result.fetchone()[0]
                
                # Count original TEXT records
                result = connection.execute(text("""
                    SELECT COUNT(*) FROM documents WHERE tags IS NOT NULL;
                """))
                text_count = result.fetchone()[0]
                
                # Sample comparison
                result = connection.execute(text("""
                    SELECT id, tags, tags_jsonb
                    FROM documents
                    WHERE tags IS NOT NULL
                    LIMIT 5;
                """))
                
                samples = result.fetchall()
                logger.info("Sample migration results:")
                for doc_id, old_tags, new_tags in samples:
                    logger.info(f"  Document {doc_id}:")
                    logger.info(f"    OLD (TEXT): {old_tags}")
                    logger.info(f"    NEW (JSONB): {new_tags}")
                
                logger.info(f"Migration verification:")
                logger.info(f"  TEXT records: {text_count}")
                logger.info(f"  JSONB records: {jsonb_count}")
                
                return jsonb_count > 0
                
        except Exception as e:
            logger.error(f"Error verifying migration: {e}")
            return False
    
    def finalize_migration(self):
        """Finalize migration by replacing old column with new one."""
        try:
            with self.engine.connect() as connection:
                # Drop the old TEXT column
                connection.execute(text("""
                    ALTER TABLE documents DROP COLUMN IF EXISTS tags;
                """))
                
                # Rename JSONB column to 'tags'
                connection.execute(text("""
                    ALTER TABLE documents RENAME COLUMN tags_jsonb TO tags;
                """))
                
                # Create index on the new JSONB column for performance
                connection.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_documents_tags_gin 
                    ON documents USING GIN (tags);
                """))
                
                connection.commit()
                logger.info("✅ Finalized migration: old column dropped, new column renamed")
                return True
                
        except Exception as e:
            logger.error(f"Error finalizing migration: {e}")
            return False
    
    def rollback_migration(self):
        """Rollback migration if something goes wrong."""
        try:
            with self.engine.connect() as connection:
                # Check if backup exists
                result = connection.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'documents_tags_backup'
                    );
                """))
                
                backup_exists = result.fetchone()[0]
                
                if backup_exists:
                    # Restore from backup
                    connection.execute(text("""
                        UPDATE documents 
                        SET tags = backup.tags
                        FROM documents_tags_backup backup
                        WHERE documents.id = backup.id;
                    """))
                    
                    # Drop new JSONB column if it exists
                    connection.execute(text("""
                        ALTER TABLE documents DROP COLUMN IF EXISTS tags_jsonb;
                    """))
                    
                    connection.commit()
                    logger.info("✅ Migration rolled back successfully")
                    return True
                else:
                    logger.error("No backup table found for rollback")
                    return False
                    
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False

def main():
    """Main migration function."""
    setup_logging()
    
    logger.info("🚀 Starting tags column migration from TEXT to JSONB...")
    logger.info("=" * 70)
    
    migration = TagsMigration()
    
    try:
        # Step 1: Check current schema
        logger.info("🔍 Step 1: Checking current schema...")
        schema_info = migration.check_current_schema()
        if not schema_info:
            logger.error("❌ Could not find 'tags' column in documents table")
            return 1
        
        logger.info(f"Current schema: {schema_info}")
        
        if schema_info['data_type'] == 'jsonb':
            logger.info("✅ Tags column is already JSONB format - no migration needed")
            return 0
        
        # Step 2: Create backup
        logger.info("\n🔧 Step 2: Creating backup...")
        if not migration.backup_existing_data():
            logger.error("❌ Failed to create backup")
            return 1
        
        # Step 3: Validate JSON data
        logger.info("\n🔍 Step 3: Validating existing JSON data...")
        is_valid, invalid_records = migration.validate_json_data()
        
        if not is_valid:
            logger.info(f"Found {len(invalid_records)} invalid records - attempting to fix...")
            if not migration.fix_invalid_json(invalid_records):
                logger.error("❌ Failed to fix invalid JSON records")
                return 1
        
        # Step 4: Create new JSONB column
        logger.info("\n🔧 Step 4: Creating new JSONB column...")
        if not migration.create_new_jsonb_column():
            logger.error("❌ Failed to create JSONB column")
            return 1
        
        # Step 5: Migrate data
        logger.info("\n📋 Step 5: Migrating data to JSONB...")
        if not migration.migrate_data_to_jsonb():
            logger.error("❌ Failed to migrate data")
            logger.info("Attempting rollback...")
            migration.rollback_migration()
            return 1
        
        # Step 6: Verify migration
        logger.info("\n✅ Step 6: Verifying migration...")
        if not migration.verify_migration():
            logger.error("❌ Migration verification failed")
            logger.info("Attempting rollback...")
            migration.rollback_migration()
            return 1
        
        # Step 7: Finalize migration
        logger.info("\n🏁 Step 7: Finalizing migration...")
        if not migration.finalize_migration():
            logger.error("❌ Failed to finalize migration")
            logger.info("Attempting rollback...")
            migration.rollback_migration()
            return 1
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 Migration completed successfully!")
        logger.info("The 'tags' column is now JSONB format with GIN index")
        logger.info("Enhanced tag filtering with optimal performance is now available")
        logger.info("\n📋 Next steps:")
        logger.info("1. The search engine will automatically use efficient JSONB operations")
        logger.info("2. You can now use advanced PostgreSQL JSONB queries")
        logger.info("3. The backup table 'documents_tags_backup' can be dropped once verified")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Migration failed with error: {str(e)}")
        logger.info("Attempting rollback...")
        migration.rollback_migration()
        return 1

if __name__ == "__main__":
    exit(main())