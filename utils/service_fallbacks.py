"""
Service fallbacks for when Redis and Qdrant services are not available.
Provides embedded alternatives for development and testing.
"""

import logging
import os
from typing import Optional, Dict, Any, List
import json
import sqlite3
from pathlib import Path

import fakeredis
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

class EmbeddedRedis:
    """FakeRedis implementation for testing/development."""
    
    def __init__(self):
        self.client = fakeredis.FakeRedis()
        logger.info("Using embedded FakeRedis for caching")
    
    def get(self, key: str) -> Optional[bytes]:
        return self.client.get(key)
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        return self.client.set(key, value, ex=ex)
    
    def delete(self, *keys) -> int:
        return self.client.delete(*keys)
    
    def exists(self, key: str) -> bool:
        return self.client.exists(key)
    
    def keys(self, pattern: str = "*") -> List[str]:
        return [k.decode() if isinstance(k, bytes) else k for k in self.client.keys(pattern)]

class EmbeddedVectorDB:
    """SQLite-based vector database for testing/development."""
    
    def __init__(self, db_path: str = "./runtime/databases/vector_storage.db"):
        self.db_path = db_path
        self.init_db()
        logger.info(f"Using embedded SQLite vector database at {db_path}")
    
    def init_db(self):
        """Initialize SQLite database with vector storage tables."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    vector_size INTEGER,
                    distance TEXT DEFAULT 'Cosine'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT,
                    collection_name TEXT,
                    vector BLOB,
                    payload TEXT,
                    FOREIGN KEY (collection_name) REFERENCES collections(name),
                    PRIMARY KEY (id, collection_name)
                )
            """)
            
            conn.commit()
    
    def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine"):
        """Create a new collection."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO collections (name, vector_size, distance) VALUES (?, ?, ?)",
                (collection_name, vector_size, distance)
            )
            conn.commit()
    
    def upsert_vectors(self, collection_name: str, vectors: List[List[float]], 
                      payloads: List[Dict[str, Any]], ids: List[str]):
        """Insert or update vectors."""
        with sqlite3.connect(self.db_path) as conn:
            for vector, payload, point_id in zip(vectors, payloads, ids):
                # Serialize vector as numpy array bytes
                vector_bytes = np.array(vector, dtype=np.float32).tobytes()
                payload_json = json.dumps(payload) if payload else "{}"
                
                conn.execute(
                    "INSERT OR REPLACE INTO vectors (id, collection_name, vector, payload) VALUES (?, ?, ?, ?)",
                    (point_id, collection_name, vector_bytes, payload_json)
                )
            conn.commit()
    
    def search_vectors(self, collection_name: str, query_vector: List[float], 
                      limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors using cosine similarity."""
        query_array = np.array(query_vector, dtype=np.float32)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, vector, payload FROM vectors WHERE collection_name = ?",
                (collection_name,)
            )
            
            results = []
            for row in cursor.fetchall():
                point_id, vector_bytes, payload_json = row
                
                # Deserialize vector
                stored_vector = np.frombuffer(vector_bytes, dtype=np.float32)
                
                # Calculate cosine similarity
                dot_product = np.dot(query_array, stored_vector)
                norm_query = np.linalg.norm(query_array)
                norm_stored = np.linalg.norm(stored_vector)
                
                if norm_query > 0 and norm_stored > 0:
                    similarity = dot_product / (norm_query * norm_stored)
                    score = float(similarity)
                else:
                    score = 0.0
                
                # Parse payload
                try:
                    payload = json.loads(payload_json)
                except:
                    payload = {}
                
                results.append({
                    "id": point_id,
                    "score": score,
                    "payload": payload
                })
            
            # Sort by score (descending) and limit results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
    
    def delete_collection(self, collection_name: str):
        """Delete a collection and all its vectors."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM vectors WHERE collection_name = ?", (collection_name,))
            conn.execute("DELETE FROM collections WHERE name = ?", (collection_name,))
            conn.commit()

def get_redis_client(host: str = "localhost", port: int = 6379, **kwargs):
    """Get Redis client with fallback to embedded Redis."""
    try:
        import redis
        client = redis.Redis(host=host, port=port, **kwargs)
        # Test connection
        client.ping()
        logger.info(f"Connected to Redis at {host}:{port}")
        return client
    except Exception as e:
        logger.warning(f"Could not connect to Redis ({e}), using embedded FakeRedis")
        return EmbeddedRedis()

def get_qdrant_client(host: str = "localhost", port: int = 6333, **kwargs):
    """Get Qdrant client with fallback to embedded vector database."""
    try:
        client = QdrantClient(host=host, port=port, **kwargs)
        # Test connection by listing collections
        client.get_collections()
        logger.info(f"Connected to Qdrant at {host}:{port}")
        return client
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant ({e}), using embedded vector database")
        return EmbeddedVectorDB()

def test_service_connections():
    """Test connections to all services and report status."""
    status = {
        "redis": False,
        "qdrant": False,
        "embedded_redis": False,
        "embedded_vector_db": False
    }
    
    # Test Redis
    try:
        import redis
        client = redis.Redis(host="localhost", port=6379)
        client.ping()
        status["redis"] = True
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.info(f"❌ Redis connection failed: {e}")
        try:
            embedded_redis = EmbeddedRedis()
            embedded_redis.set("test", "value")
            status["embedded_redis"] = True
            logger.info("✅ Embedded Redis (FakeRedis) working")
        except Exception as e2:
            logger.error(f"❌ Embedded Redis failed: {e2}")
    
    # Test Qdrant
    try:
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()
        status["qdrant"] = True
        logger.info("✅ Qdrant connection successful")
    except Exception as e:
        logger.info(f"❌ Qdrant connection failed: {e}")
        try:
            embedded_db = EmbeddedVectorDB()
            embedded_db.create_collection("test", 384)
            status["embedded_vector_db"] = True
            logger.info("✅ Embedded Vector Database (SQLite) working")
        except Exception as e2:
            logger.error(f"❌ Embedded Vector Database failed: {e2}")
    
    return status

if __name__ == "__main__":
    # Test all service connections
    logging.basicConfig(level=logging.INFO)
    status = test_service_connections()
    print("\nService Status Summary:")
    for service, available in status.items():
        emoji = "✅" if available else "❌"
        print(f"{emoji} {service}: {'Available' if available else 'Not Available'}")