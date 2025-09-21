import os
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    APP_NAME: str = Field(default="RAG Document Processing API", env="APP_NAME")
    VERSION: str = Field(default="1.0.0", env="VERSION")
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Security settings
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    ENCRYPTION_KEY: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    
    # CORS settings
    ALLOWED_ORIGINS: str = Field(default="*", env="ALLOWED_ORIGINS")
    TRUSTED_HOSTS: Optional[str] = Field(default=None, env="TRUSTED_HOSTS")
    
    # Database settings
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Redis settings
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_TTL: int = Field(default=3600, env="REDIS_TTL")  # 1 hour default
    
    # Vector Database settings (Qdrant)
    QDRANT_HOST: str = Field(default="localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(default=6333, env="QDRANT_PORT")
    QDRANT_API_KEY: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = Field(default="documents", env="QDRANT_COLLECTION_NAME")
    
    # LLM Provider settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4o-2024-08-06", env="OPENAI_MODEL")
    OPENAI_MAX_TOKENS: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    OPENAI_TEMPERATURE: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    OPENAI_TOP_P: float = Field(default=1.0, env="OPENAI_TOP_P")
    
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="llama2", env="OLLAMA_MODEL")
    
    LMSTUDIO_BASE_URL: str = Field(default="http://localhost:1234", env="LMSTUDIO_BASE_URL")
    LMSTUDIO_MODEL: str = Field(default="local-model", env="LMSTUDIO_MODEL")
    
    # Google Gemini settings
    GEMINI_API_KEY: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", env="GEMINI_MODEL")
    GEMINI_MAX_TOKENS: int = Field(default=8192, env="GEMINI_MAX_TOKENS")
    GEMINI_TEMPERATURE: float = Field(default=0.7, env="GEMINI_TEMPERATURE")
    GEMINI_TOP_P: float = Field(default=0.95, env="GEMINI_TOP_P")
    
    # Anthropic Claude settings
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = Field(default="claude-sonnet-4-20250514", env="ANTHROPIC_MODEL")
    ANTHROPIC_MAX_TOKENS: int = Field(default=8192, env="ANTHROPIC_MAX_TOKENS")
    ANTHROPIC_TEMPERATURE: float = Field(default=0.7, env="ANTHROPIC_TEMPERATURE")
    ANTHROPIC_TOP_P: float = Field(default=1.0, env="ANTHROPIC_TOP_P")
    
    # Embedding settings
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(default=384, env="EMBEDDING_DIMENSION")
    BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
    
    # Document processing settings
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    ALLOWED_FILE_TYPES: str = Field(
        default=".pdf,.docx,.txt,.md,.html,.json,.csv,.jpg,.png,.jpge",
        env="ALLOWED_FILE_TYPES"
    )
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # OCR settings
    OCR_ENABLED: bool = Field(default=True, env="OCR_ENABLED")
    OCR_LANGUAGE: str = Field(default="eng", env="OCR_LANGUAGE")
    OCR_DEFAULT_METHOD: str = Field(default="tesseract", env="OCR_DEFAULT_METHOD")
    
    # Vision LLM settings for OCR
    VISION_LLM_ENABLED: bool = Field(default=False, env="VISION_LLM_ENABLED")
    VISION_LLM_MODEL: str = Field(default="gemini-2.5-flash", env="VISION_LLM_MODEL")
    VISION_LLM_MAX_TOKENS: int = Field(default=4096, env="VISION_LLM_MAX_TOKENS")
    VISION_LLM_TEMPERATURE: float = Field(default=0.1, env="VISION_LLM_TEMPERATURE")
    VISION_LLM_TIMEOUT: int = Field(default=30, env="VISION_LLM_TIMEOUT")  # seconds
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Monitoring and logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # File storage - Updated for organized runtime structure
    UPLOAD_DIRECTORY: str = Field(default="./runtime/storage/uploads", env="UPLOAD_DIRECTORY")
    TEMP_DIRECTORY: str = Field(default="./runtime/storage/temp", env="TEMP_DIRECTORY")
    PROCESSED_DIRECTORY: str = Field(default="./runtime/storage/processed", env="PROCESSED_DIRECTORY")
    CACHE_DIRECTORY: str = Field(default="./runtime/storage/cache", env="CACHE_DIRECTORY")
    
    # Audit and security - Updated for organized runtime structure  
    ENABLE_AUDIT_LOG: bool = Field(default=True, env="ENABLE_AUDIT_LOG")
    AUDIT_LOG_PATH: str = Field(default="./runtime/logs/audit.log", env="AUDIT_LOG_PATH")
    LOG_DIRECTORY: str = Field(default="./runtime/logs", env="LOG_DIRECTORY")
    PII_DETECTION_ENABLED: bool = Field(default=True, env="PII_DETECTION_ENABLED")
    
    # Worker settings
    WORKER_CONCURRENCY: int = Field(default=4, env="WORKER_CONCURRENCY")
    BACKGROUND_TASK_TIMEOUT: int = Field(default=300, env="BACKGROUND_TASK_TIMEOUT")  # 5 minutes
    
    # Environment and fallback settings
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    USE_EMBEDDED_FALLBACKS: bool = Field(default=True, env="USE_EMBEDDED_FALLBACKS")
    
    # Phase 7: Performance Configuration
    BATCH_PROCESSOR_MAX_WORKERS: int = Field(default=10, env="BATCH_PROCESSOR_MAX_WORKERS")
    BATCH_PROCESSOR_MAX_MEMORY_MB: int = Field(default=1024, env="BATCH_PROCESSOR_MAX_MEMORY_MB")
    BATCH_PROCESSOR_DEFAULT_BATCH_SIZE: int = Field(default=50, env="BATCH_PROCESSOR_DEFAULT_BATCH_SIZE")
    
    # Query Optimization Configuration
    QUERY_CACHE_TTL_SECONDS: int = Field(default=3600, env="QUERY_CACHE_TTL_SECONDS")
    QUERY_CACHE_MAX_SIZE_MB: int = Field(default=500, env="QUERY_CACHE_MAX_SIZE_MB")
    ENABLE_QUERY_PLANNING: bool = Field(default=True, env="ENABLE_QUERY_PLANNING")
    
    # Load Balancer Configuration
    LOAD_BALANCER_STRATEGY: str = Field(default="adaptive", env="LOAD_BALANCER_STRATEGY")
    LOAD_BALANCER_HEALTH_CHECK_INTERVAL: int = Field(default=30, env="LOAD_BALANCER_HEALTH_CHECK_INTERVAL")
    ENABLE_CIRCUIT_BREAKER: bool = Field(default=True, env="ENABLE_CIRCUIT_BREAKER")
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    
    # Async Processing Configuration
    ASYNC_WORKER_POOL_MAX_WORKERS: int = Field(default=10, env="ASYNC_WORKER_POOL_MAX_WORKERS")
    ASYNC_WORKER_POOL_QUEUE_SIZE: int = Field(default=1000, env="ASYNC_WORKER_POOL_QUEUE_SIZE")
    ENABLE_AUTO_SCALING: bool = Field(default=True, env="ENABLE_AUTO_SCALING")
    
    # Phase 8: Monitoring Configuration
    USAGE_STATISTICS_ENABLED: bool = Field(default=True, env="USAGE_STATISTICS_ENABLED")
    USAGE_STATS_FLUSH_INTERVAL: int = Field(default=300, env="USAGE_STATS_FLUSH_INTERVAL")  # 5 minutes
    USAGE_STATS_MAX_EVENTS: int = Field(default=10000, env="USAGE_STATS_MAX_EVENTS")

    # Model Storage Configuration
    MODEL_STORAGE_BASE_PATH: str = Field(default="./runtime/models", env="MODEL_STORAGE_BASE_PATH")
    MODEL_STORAGE_ENABLE_AUTO_DOWNLOAD: bool = Field(default=True, env="MODEL_STORAGE_ENABLE_AUTO_DOWNLOAD")
    MODEL_STORAGE_MAX_CONCURRENT_DOWNLOADS: int = Field(default=2, env="MODEL_STORAGE_MAX_CONCURRENT_DOWNLOADS")
    MODEL_STORAGE_LIMIT_GB: Optional[float] = Field(default=None, env="MODEL_STORAGE_LIMIT_GB")
    MODEL_STORAGE_CLEANUP_DAYS: int = Field(default=90, env="MODEL_STORAGE_CLEANUP_DAYS")

    # Model Preloader Configuration
    MODEL_PRELOAD_POLICY: str = Field(default="all", env="MODEL_PRELOAD_POLICY")  # none, essential, all, selective
    MODEL_PRELOAD_TIMEOUT_MINUTES: int = Field(default=30, env="MODEL_PRELOAD_TIMEOUT_MINUTES")
    MODEL_PRELOAD_ON_STARTUP: bool = Field(default=True, env="MODEL_PRELOAD_ON_STARTUP")
    MODEL_PRELOAD_BACKGROUND: bool = Field(default=True, env="MODEL_PRELOAD_BACKGROUND")
    MODEL_PRELOAD_ESSENTIAL_USE_CASES: str = Field(default="general,semantic_search,question_answering", env="MODEL_PRELOAD_ESSENTIAL_USE_CASES")
    
    # Backup Configuration - Updated for organized runtime structure
    BACKUP_ENABLED: bool = Field(default=True, env="BACKUP_ENABLED")
    BACKUP_DIRECTORY: str = Field(default="./runtime/backups", env="BACKUP_DIRECTORY")
    BACKUP_RETENTION_DAYS: int = Field(default=30, env="BACKUP_RETENTION_DAYS")
    BACKUP_COMPRESSION: bool = Field(default=True, env="BACKUP_COMPRESSION")
    BACKUP_SCHEDULE_DAILY: str = Field(default="0 2 * * *", env="BACKUP_SCHEDULE_DAILY")  # Daily at 2 AM
    
    # Alert Configuration
    ALERTS_ENABLED: bool = Field(default=True, env="ALERTS_ENABLED")
    ALERT_EMAIL_SMTP_HOST: str = Field(default="localhost", env="ALERT_EMAIL_SMTP_HOST")
    ALERT_EMAIL_FROM: str = Field(default="alerts@rag-system.com", env="ALERT_EMAIL_FROM")
    ALERT_WEBHOOK_URL: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    ALERT_SLACK_WEBHOOK_URL: Optional[str] = Field(default=None, env="ALERT_SLACK_WEBHOOK_URL")
    
    # Dashboard Configuration
    DASHBOARD_ENABLED: bool = Field(default=True, env="DASHBOARD_ENABLED")
    DASHBOARD_REFRESH_INTERVAL: int = Field(default=30, env="DASHBOARD_REFRESH_INTERVAL")
    DASHBOARD_MAX_HISTORY: int = Field(default=100, env="DASHBOARD_MAX_HISTORY")
    DASHBOARD_THEME: str = Field(default="auto", env="DASHBOARD_THEME")
    
    # Export Configuration - Updated for organized runtime structure
    EXPORT_DIRECTORY: str = Field(default="./runtime/storage/exports", env="EXPORT_DIRECTORY") 
    EXPORT_MAX_FILE_SIZE_MB: int = Field(default=500, env="EXPORT_MAX_FILE_SIZE_MB")
    EXPORT_COMPRESSION_DEFAULT: bool = Field(default=True, env="EXPORT_COMPRESSION_DEFAULT")
    
    # Vector Database Storage - Updated for organized runtime structure
    QDRANT_STORAGE_DIRECTORY: str = Field(default="./runtime/storage/qdrant_storage", env="QDRANT_STORAGE_DIRECTORY")
    EMBEDDINGS_CACHE_DIRECTORY: str = Field(default="./runtime/storage/embeddings", env="EMBEDDINGS_CACHE_DIRECTORY")
    VECTOR_INDICES_DIRECTORY: str = Field(default="./runtime/storage/vector_indices", env="VECTOR_INDICES_DIRECTORY")
    
    # Database Storage - Updated for organized runtime structure  
    DATABASE_DIRECTORY: str = Field(default="./runtime/databases", env="DATABASE_DIRECTORY")
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if not v or len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
    @validator("ALLOWED_FILE_TYPES")
    def validate_file_types(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Development settings
class DevelopmentSettings(Settings):
    """Development-specific settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    HOST: str = "127.0.0.1"
    CORS_ALLOW_ALL: bool = True

# Production settings
class ProductionSettings(Settings):
    """Production-specific settings."""
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    HOST: str = "0.0.0.0"
    
    @validator("SECRET_KEY")
    def validate_production_secret_key(cls, v):
        if not v or len(v) < 64:
            raise ValueError("Production SECRET_KEY must be at least 64 characters long")
        return v

# Testing settings
class TestSettings(Settings):
    """Test-specific settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DATABASE_URL: str = "sqlite:///:memory:"
    REDIS_DB: int = 15  # Use separate Redis DB for tests
    UPLOAD_DIRECTORY: str = "./runtime/storage/test_uploads"
    TEMP_DIRECTORY: str = "./runtime/storage/test_temp"

def get_settings_for_environment(env: str = None) -> Settings:
    """Get settings based on environment."""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestSettings()
    else:
        return DevelopmentSettings()