# -*- coding: utf-8 -*-
"""
アプリケーション設定
Pydantic Settings を使用した型安全な設定管理
"""

import os
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """アプリケーション設定クラス"""

    # Flask 基本設定
    SECRET_KEY: str = Field(
        default_factory=lambda: os.urandom(32).hex(),
        description="Flask セッション暗号化キー"
    )
    DEBUG: bool = Field(default=False, description="デバッグモード")
    TESTING: bool = Field(default=False, description="テストモード")
    FLASK_ENV: str = Field(default="development", description="Flask環境")

    # サーバー設定
    HOST: str = Field(default="0.0.0.0", description="サーバーホスト")
    PORT: int = Field(default=5000, description="サーバーポート")

    # OCI 設定
    OCI_CONFIG_FILE: str = Field(default="~/.oci/config", description="OCI設定ファイルパス")
    OCI_PROFILE: str = Field(default="DEFAULT", description="OCI設定プロファイル")
    OCI_BUCKET: str = Field(default="chatbot-images", description="デフォルトバケット名")
    OCI_REGION: Optional[str] = Field(default=None, description="OCI リージョン")
    OCI_COMPARTMENT_OCID: Optional[str] = Field(default=None, description="OCI コンパートメントOCID")

    # Oracle データベース設定
    DB_USERNAME: str = Field(default="ADMIN", description="データベースユーザー名")
    DB_PASSWORD: str = Field(default="", description="データベースパスワード")
    DB_DSN: str = Field(default="localhost:1521/XEPDB1", description="データベース接続文字列（DSN）")

    # Oracle Generative AI設定
    OCI_COHERE_EMBED_MODEL: str = Field(default="cohere.embed-v4.0", description="Oracle Generative AI埋め込みモデル")
    OCI_EMBEDDING_DIMENSION: int = Field(default=1536, description="埋め込みベクトル次元数")
    OCI_EMBEDDING_INPUT_TYPE: str = Field(default="IMAGE", description="埋め込み入力タイプ")
    OCI_EMBEDDING_TRUNCATE: str = Field(default="NONE", description="切り詰め戦略")
    OCI_EMBEDDING_BATCH_SIZE: int = Field(default=1, description="バッチ処理サイズ")
    OCI_EMBEDDING_MAX_RETRIES: int = Field(default=3, description="最大リトライ回数")
    OCI_EMBEDDING_RETRY_DELAY: int = Field(default=10, description="リトライ遅延（秒）")

    # セキュリティ設定
    MAX_CONTENT_LENGTH: int = Field(default=16 * 1024 * 1024, description="最大アップロードサイズ (16MB)")
    ALLOWED_EXTENSIONS: List[str] = Field(
        default=['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg'],
        description="許可されたファイル拡張子"
    )
    ALLOW_PRIVATE_IPS: bool = Field(default=True, description="プライベートIPアドレスからの画像取得を許可")

    # セッション設定
    SESSION_COOKIE_SECURE: bool = Field(default=True, description="HTTPS必須")
    SESSION_COOKIE_HTTPONLY: bool = Field(default=True, description="JavaScript無効")
    SESSION_COOKIE_SAMESITE: str = Field(default="Lax", description="SameSite設定")
    PERMANENT_SESSION_LIFETIME: int = Field(default=3600, description="セッション有効期間(秒)")

    # CORS設定
    CORS_ORIGINS: List[str] = Field(default=["*"], description="CORS許可オリジン")
    CORS_METHODS: List[str] = Field(default=["GET", "POST", "OPTIONS"], description="CORS許可メソッド")

    # レート制限設定
    RATELIMIT_STORAGE_URL: str = Field(default="memory://", description="レート制限ストレージ")
    RATELIMIT_DEFAULT: str = Field(default="100 per hour", description="デフォルトレート制限")
    RATELIMIT_UPLOAD: str = Field(default="10 per minute", description="アップロードレート制限")

    # ログ設定
    LOG_LEVEL: str = Field(default="INFO", description="ログレベル")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="ログフォーマット"
    )

    # Sentry設定（エラー監視）
    SENTRY_DSN: Optional[str] = Field(default=None, description="Sentry DSN")
    SENTRY_ENVIRONMENT: str = Field(default="development", description="Sentry環境")

    # セキュリティヘッダー設定
    FORCE_HTTPS: bool = Field(default=False, description="HTTPS強制")
    CONTENT_SECURITY_POLICY: str = Field(
        default="default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'",
        description="Content Security Policy"
    )

    @field_validator('ALLOWED_EXTENSIONS')
    @classmethod
    def validate_extensions(cls, v):
        """ファイル拡張子の検証"""
        return [ext.lower().strip('.') for ext in v]

    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v):
        """ログレベルの検証"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'ログレベルは {valid_levels} のいずれかである必要があります')
        return v.upper()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"  # 追加フィールドを無視
    }


# グローバル設定インスタンス
settings = Settings()


class DevelopmentConfig:
    """開発環境設定"""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False
    FORCE_HTTPS = False


class ProductionConfig:
    """本番環境設定"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True
    FORCE_HTTPS = True


class TestingConfig:
    """テスト環境設定"""
    DEBUG = False
    TESTING = True
    SESSION_COOKIE_SECURE = False
    FORCE_HTTPS = False
    OCI_BUCKET = "test-bucket"


# 環境別設定マッピング
config_mapping = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
}


def get_config(env: str = None) -> object:
    """環境に応じた設定を取得"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')

    return config_mapping.get(env, DevelopmentConfig)
