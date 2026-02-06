# -*- coding: utf-8 -*-
"""
Application Configuration
Type-safe configuration management using Pydantic Settings
"""

import os
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings class"""

    # Flask basic configuration
    SECRET_KEY: str = Field(
        default_factory=lambda: os.urandom(32).hex(),
        description="Flask session encryption key"
    )
    DEBUG: bool = Field(default=False, description="Debug mode")
    TESTING: bool = Field(default=False, description="Test mode")
    FLASK_ENV: str = Field(default="development", description="Flask environment")

    # Server configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=5000, description="Server port")

    # OCI configuration
    OCI_CONFIG_FILE: str = Field(default="~/.oci/config", description="OCI config file path")
    OCI_PROFILE: str = Field(default="DEFAULT", description="OCI config profile")
    OCI_BUCKET: str = Field(default="chatbot-images", description="Default bucket name")
    OCI_REGION: Optional[str] = Field(default=None, description="OCI region")
    OCI_OBJECT_STORAGE_REGION: Optional[str] = Field(default=None, description="OCI Object Storage region (if different from OCI_REGION)")
    OCI_COMPARTMENT_OCID: Optional[str] = Field(default=None, description="OCI compartment OCID")

    # Oracle database configuration
    DB_USERNAME: str = Field(default="ADMIN", description="Database username")
    DB_PASSWORD: str = Field(default="", description="Database password")
    DB_DSN: str = Field(default="localhost:1521/XEPDB1", description="Database connection string (DSN)")

    # Oracle Generative AI configuration
    OCI_COHERE_EMBED_MODEL: str = Field(default="cohere.embed-v4.0", description="Oracle Generative AI embedding model")
    OCI_EMBEDDING_DIMENSION: int = Field(default=1536, description="Embedding vector dimension")
    OCI_EMBEDDING_INPUT_TYPE: str = Field(default="IMAGE", description="Embedding input type")
    OCI_EMBEDDING_TRUNCATE: str = Field(default="NONE", description="Truncation strategy")
    OCI_EMBEDDING_BATCH_SIZE: int = Field(default=1, description="Batch processing size")
    OCI_EMBEDDING_MAX_RETRIES: int = Field(default=3, description="Maximum retry count")
    OCI_EMBEDDING_RETRY_DELAY: int = Field(default=10, description="Retry delay (seconds)")
    
    # OCI Generic API configuration
    OCI_API_MAX_RETRIES: int = Field(default=5, description="Maximum retry count for OCI APIs")
    OCI_API_RETRY_BASE_DELAY: int = Field(default=2, description="Base retry delay (seconds) for exponential backoff")

    # Security configuration
    MAX_CONTENT_LENGTH: int = Field(default=16 * 1024 * 1024, description="Maximum upload size (16MB)")
    ALLOWED_EXTENSIONS: List[str] = Field(
        default=['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg', 'ppt', 'pptx', 'pdf', 'doc', 'docx', 'xls', 'xlsx'],
        description="Allowed file extensions"
    )
    ALLOW_PRIVATE_IPS: bool = Field(default=True, description="Allow image fetching from private IP addresses")

    # Session configuration
    SESSION_COOKIE_SECURE: bool = Field(default=True, description="HTTPS required")
    SESSION_COOKIE_HTTPONLY: bool = Field(default=True, description="JavaScript disabled")
    SESSION_COOKIE_SAMESITE: str = Field(default="Lax", description="SameSite setting")
    PERMANENT_SESSION_LIFETIME: int = Field(default=3600, description="Session lifetime (seconds)")

    # CORS configuration
    CORS_ORIGINS: List[str] = Field(default=["*"], description="CORS allowed origins")
    CORS_METHODS: List[str] = Field(default=["GET", "POST", "OPTIONS"], description="CORS allowed methods")

    # Rate limiting configuration
    RATELIMIT_STORAGE_URL: str = Field(default="memory://", description="Rate limit storage")
    RATELIMIT_DEFAULT: str = Field(default="100 per hour", description="Default rate limit")
    RATELIMIT_UPLOAD: str = Field(default="10 per minute", description="Upload rate limit")

    # Logging configuration
    LOG_LEVEL: str = Field(default="INFO", description="Log level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )

    # Sentry configuration (error monitoring)
    SENTRY_DSN: Optional[str] = Field(default=None, description="Sentry DSN")
    SENTRY_ENVIRONMENT: str = Field(default="development", description="Sentry environment")

    # Security headers configuration
    FORCE_HTTPS: bool = Field(default=False, description="Force HTTPS")
    CONTENT_SECURITY_POLICY: str = Field(
        default="default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'",
        description="Content Security Policy"
    )

    @field_validator('ALLOWED_EXTENSIONS')
    @classmethod
    def validate_extensions(cls, v):
        """Validate file extensions"""
        # Ensure all extensions are lowercase and don't contain dots
        validated = []
        for ext in v:
            clean_ext = ext.lower().strip().strip('.')
            if clean_ext:  # Ensure not empty string
                validated.append(clean_ext)
        
        # Add logging for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Validated allowed extensions: {validated}")
        
        return validated

    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"  # Ignore additional fields
    }


# Global settings instance
settings = Settings()


class DevelopmentConfig:
    """Development environment configuration"""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False
    FORCE_HTTPS = False


class ProductionConfig:
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True
    FORCE_HTTPS = True


class TestingConfig:
    """Test environment configuration"""
    DEBUG = False
    TESTING = True
    SESSION_COOKIE_SECURE = False
    FORCE_HTTPS = False
    OCI_BUCKET = "test-bucket"


# Environment-specific configuration mapping
config_mapping = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
}


def get_config(env: str = None) -> object:
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')

    return config_mapping.get(env, DevelopmentConfig)
