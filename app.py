#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle Cloud Storage 画像プロキシアプリケーション
OCIオブジェクトストレージの画像を認証付きで表示・アップロードするFlaskアプリ

業界ベストプラクティスに基づく実装:
- 型安全な設定管理 (Pydantic)
- 構造化ログ (structlog)
- セキュリティヘッダー (Flask-Talisman)
- レート制限 (Flask-Limiter)
- CORS対応 (Flask-CORS)
- エラー監視 (Sentry)
"""

import os
import uuid
import structlog
from datetime import datetime
from typing import Optional, Tuple
import time
import io
import requests
from urllib.parse import urlparse
import mimetypes

from flask import Flask, Response, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman

import oci
from oci.object_storage import ObjectStorageClient
from oci.exceptions import ServiceError

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

import oracledb
import numpy as np
from PIL import Image
import base64
import array

from config import settings, get_config

# 標準ライブラリのloggingモジュールを設定
import logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT
)

# 構造化ログの設定
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(ensure_ascii=False)
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class OCIClient:
    """OCI Object Storage クライアントラッパー"""

    def __init__(self):
        self.client: Optional[ObjectStorageClient] = None
        self.namespace: Optional[str] = None
        self._initialize()

    def _initialize(self):
        """OCI クライアントの初期化"""
        try:
            # OCI設定の読み込み
            config_file = os.path.expanduser(settings.OCI_CONFIG_FILE)
            if not os.path.exists(config_file):
                logger.warning("OCI設定ファイルが見つかりません", config_file=config_file)
                return

            config = oci.config.from_file(
                file_location=config_file,
                profile_name=settings.OCI_PROFILE
            )

            # リージョンの設定
            if settings.OCI_REGION:
                config['region'] = settings.OCI_REGION

            self.client = ObjectStorageClient(config)
            self.namespace = self.client.get_namespace().data

            logger.info("OCI接続成功",
                       namespace=self.namespace,
                       region=config.get('region'))

        except Exception as e:
            logger.error("OCI設定の初期化に失敗", error=str(e))
            self.client = None
            self.namespace = None

    def is_connected(self) -> bool:
        """接続状態の確認"""
        return self.client is not None and self.namespace is not None

    def get_object(self, bucket_name: str, object_name: str):
        """オブジェクトの取得"""
        if not self.is_connected():
            raise RuntimeError("OCI クライアントが初期化されていません")

        return self.client.get_object(
            namespace_name=self.namespace,
            bucket_name=bucket_name,
            object_name=object_name
        )

    def put_object(self, bucket_name: str, object_name: str, data, content_type: str = None):
        """オブジェクトのアップロード"""
        if not self.is_connected():
            raise RuntimeError("OCI クライアントが初期化されていません")

        return self.client.put_object(
            namespace_name=self.namespace,
            bucket_name=bucket_name,
            object_name=object_name,
            put_object_body=data,
            content_type=content_type
        )


# グローバル OCI クライアント
oci_client = OCIClient()


class DatabaseClient:
    """Oracle データベースクライアント"""
    
    def __init__(self):
        self.connection = None
        self._initialize()
    
    def _initialize(self):
        """データベース接続を初期化"""
        try:
            self.connection = oracledb.connect(
                user=settings.DB_USERNAME,
                password=settings.DB_PASSWORD,
                dsn=settings.DB_DSN
            )
            logger.info("Oracle データベース接続成功", dsn=settings.DB_DSN)
        except Exception as e:
            logger.error("Oracle データベース接続失敗", error=str(e))
            self.connection = None
    
    def is_connected(self) -> bool:
        """データベース接続状態をチェック"""
        try:
            if self.connection is None:
                return False
            # 簡単なクエリでテスト
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM DUAL")
                cursor.fetchone()
            return True
        except Exception:
            return False
    
    def insert_embedding(self, bucket: str, object_name: str, content_type: str, 
                        file_size: int, embedding: np.ndarray) -> Optional[int]:
        """埋め込みベクトルをデータベースに挿入"""
        try:
            if not self.is_connected():
                logger.error("データベース接続が無効")
                return None
            
            # NumPy配列をFLOAT32配列に変換
            embedding_array = array.array("f", embedding.tolist())
            
            with self.connection.cursor() as cursor:
                # IMG_EMBEDDINGSテーブルに挿入
                cursor.execute("""
                    INSERT INTO IMG_EMBEDDINGS 
                    (BUCKET, OBJECT_NAME, CONTENT_TYPE, FILE_SIZE, EMBEDDING)
                    VALUES (:bucket, :object_name, :content_type, :file_size, :embedding)
                    RETURNING ID INTO :id
                """, {
                    'bucket': bucket,
                    'object_name': object_name,
                    'content_type': content_type,
                    'file_size': file_size,
                    'embedding': embedding_array,
                    'id': cursor.var(oracledb.NUMBER)
                })
                
                # 挿入されたIDを取得
                id_var = cursor.bindvars['id']
                embedding_id = id_var.getvalue()
                self.connection.commit()
                
                # embedding_idがリストの場合は最初の要素を取得
                if isinstance(embedding_id, list) and len(embedding_id) > 0:
                    embedding_id = embedding_id[0]
                
                logger.info("埋め込みベクトル挿入成功", 
                           embedding_id=embedding_id, 
                           object_name=object_name)
                return embedding_id
                
        except Exception as e:
            logger.error("埋め込みベクトル挿入失敗", error=str(e))
            if self.connection:
                self.connection.rollback()
            return None
    
    def search_similar_images(self, query_embedding: np.ndarray, limit: int = 10, threshold: float = 0.7) -> Optional[list]:
        """埋め込みベクトルに基づいて類似画像を検索
        
        Args:
            query_embedding: 検索用の埋め込みベクトル
            limit: 返す結果の最大数
            threshold: 類似度の閾値（0.0-1.0）
            
        Returns:
            類似画像のリスト（bucket, object_name, vector_distance）
        """
        try:
            if not self.is_connected():
                logger.error("データベース接続が無効")
                return None
            
            # NumPy配列をFLOAT32配列に変換
            embedding_array = array.array("f", query_embedding.tolist())
            
            with self.connection.cursor() as cursor:
                # ベクトル類似度検索クエリを実行
                sql = """
                SELECT 
                    ie.ID as embed_id, 
                    ie.BUCKET as bucket,
                    ie.OBJECT_NAME as object_name,
                    VECTOR_DISTANCE(ie.EMBEDDING, :query_embedding, COSINE) as vector_distance 
                FROM  
                    IMG_EMBEDDINGS ie 
                WHERE  
                    VECTOR_DISTANCE(ie.EMBEDDING, :query_embedding, COSINE) <= :threshold 
                ORDER BY  
                    vector_distance
                FETCH FIRST :limit ROWS ONLY
                """
                
                cursor.execute(sql, {
                    'query_embedding': embedding_array,
                    'threshold': threshold,
                    'limit': limit
                })
                
                results = []
                for row in cursor:
                    results.append({
                        'embed_id': row[0],
                        'bucket': row[1],
                        'object_name': row[2],
                        'vector_distance': float(row[3])
                    })
                
                logger.info("ベクトル検索成功", 
                           results_count=len(results),
                           threshold=threshold)
                return results
                
        except Exception as e:
            logger.error("ベクトル検索失敗", error=str(e))
            return None


class ImageEmbedder:
    """Oracle Generative AI を使用した画像埋め込みクライアント"""
    
    def __init__(self):
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Oracle Generative AI クライアントを初期化"""
        try:
            config = oci.config.from_file('~/.oci/config', "DEFAULT")
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                service_endpoint=f"https://inference.generativeai.{settings.OCI_REGION}.oci.oraclecloud.com",
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240)
            )
            logger.info("Oracle Generative AI クライアント初期化成功")
        except Exception as e:
            logger.error("Oracle Generative AI クライアント初期化失敗", error=str(e))
            self.client = None
    
    def is_initialized(self) -> bool:
        """クライアント初期化状態をチェック"""
        return self.client is not None
    
    def _image_to_base64(self, image_data: io.BytesIO, content_type: str = "image/png") -> str:
        """画像データをbase64 data URIエンコード"""
        image_data.seek(0)
        image_bytes = image_data.read()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:{content_type};base64,{base64_string}"
    
    def generate_embedding(self, image_data: io.BytesIO, content_type: str = "image/png") -> Optional[np.ndarray]:
        """画像から埋め込みベクトルを生成"""
        try:
            if not self.is_initialized():
                logger.error("Oracle Generative AI クライアントが初期化されていません")
                return None
            
            logger.info(f"{settings.OCI_EMBEDDING_INPUT_TYPE=}")
            logger.info(f"{settings.OCI_COMPARTMENT_OCID=}")

            # 画像をbase64エンコード
            base64_image = self._image_to_base64(image_data, content_type)
            
            # 埋め込みリクエストを作成
            embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
            embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                model_id=settings.OCI_COHERE_EMBED_MODEL
            )
            embed_text_detail.input_type = settings.OCI_EMBEDDING_INPUT_TYPE
            embed_text_detail.inputs = [base64_image]
            embed_text_detail.truncate = settings.OCI_EMBEDDING_TRUNCATE
            embed_text_detail.compartment_id = settings.OCI_COMPARTMENT_OCID
            
            # リトライロジック
            max_retries = settings.OCI_EMBEDDING_MAX_RETRIES
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = self.client.embed_text(embed_text_detail)
                    
                    if response.data.embeddings:
                        embedding = response.data.embeddings[0]
                        embedding_array = np.array(embedding, dtype=np.float32)
                        
                        logger.info("画像埋め込みベクトル生成成功", 
                                   embedding_shape=embedding_array.shape,
                                   embedding_norm=float(np.linalg.norm(embedding_array)))
                        return embedding_array
                    else:
                        logger.error("埋め込みベクトルが空です")
                        return None
                        
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"画像埋め込み生成エラー: {e}. リトライ中 ({retry_count}/{max_retries})...")
                    if retry_count < max_retries:
                        time.sleep(settings.OCI_EMBEDDING_RETRY_DELAY * retry_count)
                    else:
                        logger.error("画像埋め込み生成の最大リトライ回数に達しました")
                        return None
            
            return None
            
        except Exception as e:
            logger.error("画像埋め込み生成中に予期しないエラー", error=str(e))
            return None


class TextEmbedder:
    """Oracle Generative AI を使用したテキスト埋め込みクライアント"""
    
    def __init__(self):
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Oracle Generative AI クライアントを初期化"""
        try:
            config = oci.config.from_file('~/.oci/config', "DEFAULT")
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                service_endpoint=f"https://inference.generativeai.{settings.OCI_REGION}.oci.oraclecloud.com",
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240)
            )
            logger.info("Oracle Generative AI テキストクライアント初期化成功")
        except Exception as e:
            logger.error("Oracle Generative AI テキストクライアント初期化失敗", error=str(e))
            self.client = None
    
    def is_initialized(self) -> bool:
        """クライアント初期化状態をチェック"""
        return self.client is not None
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """テキストから埋め込みベクトルを生成（3回リトライ付き）"""
        try:
            if not self.is_initialized():
                logger.error("Oracle Generative AI クライアントが初期化されていません")
                return None
            
            # 埋め込みリクエストを作成
            embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
            embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                model_id=settings.OCI_COHERE_EMBED_MODEL
            )
            embed_text_detail.input_type = "SEARCH_QUERY"  # テキスト検索用
            embed_text_detail.inputs = [text]
            embed_text_detail.truncate = settings.OCI_EMBEDDING_TRUNCATE
            embed_text_detail.compartment_id = settings.OCI_COMPARTMENT_OCID
            
            # リトライロジック（3回）
            max_retries = settings.OCI_EMBEDDING_MAX_RETRIES
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    logger.info(f"テキスト埋め込み生成試行 {retry_count + 1}/{max_retries}", text=text[:100])
                    response = self.client.embed_text(embed_text_detail)
                    
                    if response.data.embeddings:
                        embedding = response.data.embeddings[0]
                        embedding_array = np.array(embedding, dtype=np.float32)
                        
                        logger.info("テキスト埋め込みベクトル生成成功", 
                                   text_length=len(text),
                                   embedding_shape=embedding_array.shape,
                                   embedding_norm=float(np.linalg.norm(embedding_array)))
                        return embedding_array
                    else:
                        logger.error("埋め込みベクトルが空です")
                        return None
                        
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"テキスト埋め込み生成エラー: {e}. リトライ中 ({retry_count}/{max_retries})...")
                    if retry_count < max_retries:
                        time.sleep(settings.OCI_EMBEDDING_RETRY_DELAY * retry_count)
                    else:
                        logger.error("テキスト埋め込み生成の最大リトライ回数に達しました")
                        return None
            
            return None
            
        except Exception as e:
            logger.error("テキスト埋め込み生成中に予期しないエラー", error=str(e), text=text[:100])
            return None


# グローバルクライアント
db_client = DatabaseClient()
image_embedder = ImageEmbedder()
text_embedder = TextEmbedder()


def allowed_file(filename: str) -> bool:
    """許可されたファイル拡張子かチェック"""
    if not filename or '.' not in filename:
        logger.debug(f"ファイル名が無効: {filename}")
        return False

    extension = filename.rsplit('.', 1)[1].lower()
    allowed = extension in settings.ALLOWED_EXTENSIONS
    
    # デバッグ情報を追加
    logger.debug(f"ファイル拡張子チェック: filename={filename}, extension={extension}, allowed_extensions={settings.ALLOWED_EXTENSIONS}, result={allowed}")
    
    return allowed


def create_response(success: bool, message: str, data: Optional[dict] = None, status_code: int = 200) -> Tuple[dict, int]:
    """統一されたAPIレスポンス形式を作成
    
    Args:
        success: 処理の成功/失敗
        message: レスポンスメッセージ
        data: 追加データ（オプション）
        status_code: HTTPステータスコード
        
    Returns:
        Tuple[dict, int]: (レスポンスデータ, ステータスコード)
    """
    response = {
        'success': success,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    
    if data is not None:
        response['data'] = data
        
    return response, status_code


def is_valid_image_url(url: str) -> Tuple[bool, str]:
    """有効な画像URLかチェック（SSRF攻撃防止）
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        parsed = urlparse(url)
        logger.info("URL検証開始", url=url, scheme=parsed.scheme, hostname=parsed.hostname)
        
        # HTTPSまたはHTTPのみ許可
        if parsed.scheme not in ['http', 'https']:
            error_msg = f"サポートされていないプロトコル: {parsed.scheme}"
            logger.warning("URL検証失敗", url=url, reason=error_msg)
            return False, error_msg
            
        # ローカルIPアドレスやプライベートIPアドレスを拒否
        import ipaddress
        import socket
        
        hostname = parsed.hostname
        if not hostname:
            error_msg = "ホスト名が見つかりません"
            logger.warning("URL検証失敗", url=url, reason=error_msg)
            return False, error_msg
            
        # IPアドレスの場合はプライベートIPを拒否
        try:
            ip = ipaddress.ip_address(hostname)
            logger.info("IP検証", ip=str(ip), is_private=ip.is_private, is_loopback=ip.is_loopback, is_link_local=ip.is_link_local)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                if not settings.ALLOW_PRIVATE_IPS:
                    error_msg = f"プライベートIPアドレスは許可されていません: {ip} (ALLOW_PRIVATE_IPS=Falseのため)"
                    logger.warning("URL検証失敗", url=url, ip=str(ip), reason=error_msg, allow_private_ips=settings.ALLOW_PRIVATE_IPS)
                    return False, error_msg
                else:
                    logger.info("プライベートIP許可", ip=str(ip), url=url, allow_private_ips=settings.ALLOW_PRIVATE_IPS)
        except ValueError:
            # ホスト名の場合はDNS解決してIPをチェック
            try:
                ip_str = socket.gethostbyname(hostname)
                ip = ipaddress.ip_address(ip_str)
                logger.info("DNS解決後IP検証", hostname=hostname, resolved_ip=ip_str, is_private=ip.is_private)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    if not settings.ALLOW_PRIVATE_IPS:
                        error_msg = f"ホスト名 {hostname} がプライベートIPに解決されました: {ip} (ALLOW_PRIVATE_IPS=Falseのため)"
                        logger.warning("URL検証失敗", url=url, hostname=hostname, resolved_ip=str(ip), reason=error_msg, allow_private_ips=settings.ALLOW_PRIVATE_IPS)
                        return False, error_msg
                    else:
                        logger.info("プライベートIP許可", hostname=hostname, resolved_ip=str(ip), url=url, allow_private_ips=settings.ALLOW_PRIVATE_IPS)
            except (socket.gaierror, ValueError) as e:
                error_msg = f"DNS解決に失敗: {str(e)}"
                logger.warning("URL検証失敗", url=url, hostname=hostname, reason=error_msg)
                return False, error_msg
                
        logger.info("URL検証成功", url=url)
        return True, ""
    except Exception as e:
        error_msg = f"URL検証中にエラー: {str(e)}"
        logger.error("URL検証エラー", url=url, error=str(e))
        return False, error_msg


def download_image_from_url(url: str, max_size: int = None) -> Tuple[io.BytesIO, str, str]:
    """URLから画像をダウンロード
    
    Returns:
        Tuple[io.BytesIO, str, str]: (画像データ, content_type, ファイル名)
    """
    if max_size is None:
        max_size = settings.MAX_CONTENT_LENGTH
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; ImageProxy/1.0)'
    }
    
    response = requests.get(
        url, 
        headers=headers, 
        stream=True, 
        timeout=30,
        allow_redirects=True
    )
    response.raise_for_status()
    
    # Content-Typeチェック
    content_type = response.headers.get('content-type', '')
    if not content_type.startswith('image/'):
        raise ValueError(f"無効なコンテンツタイプ: {content_type}")
    
    # ファイルサイズチェック
    content_length = response.headers.get('content-length')
    if content_length and int(content_length) > max_size:
        raise ValueError(f"ファイルサイズが大きすぎます: {content_length} bytes")
    
    # 画像データをダウンロード
    image_data = io.BytesIO()
    downloaded_size = 0
    
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                raise ValueError(f"ファイルサイズが大きすぎます: {downloaded_size} bytes")
            image_data.write(chunk)
    
    image_data.seek(0)
    
    # ファイル名を生成（URLから推測またはContent-Typeから生成）
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    if not filename or '.' not in filename:
        # Content-Typeから拡張子を推測
        extension = mimetypes.guess_extension(content_type)
        if extension:
            filename = f"image{extension}"
        else:
            filename = "image.jpg"  # デフォルト
    
    return image_data, content_type, filename


def create_app(config_name: str = None) -> Flask:
    """アプリケーションファクトリ"""
    app = Flask(__name__)

    # 設定の読み込み
    if config_name:
        config_class = get_config(config_name)
        app.config.from_object(config_class)

    # 基本設定
    app.config.update(
        SECRET_KEY=settings.SECRET_KEY,
        MAX_CONTENT_LENGTH=settings.MAX_CONTENT_LENGTH,
        SESSION_COOKIE_SECURE=settings.SESSION_COOKIE_SECURE,
        SESSION_COOKIE_HTTPONLY=settings.SESSION_COOKIE_HTTPONLY,
        SESSION_COOKIE_SAMESITE=settings.SESSION_COOKIE_SAMESITE,
        PERMANENT_SESSION_LIFETIME=settings.PERMANENT_SESSION_LIFETIME,
    )
    
    # 記録当前配置以便調試
    logger.info(f"应用启动 - 允许的文件扩展名: {settings.ALLOWED_EXTENSIONS}")
    logger.info(f"应用启动 - 最大文件大小: {settings.MAX_CONTENT_LENGTH} bytes")
    logger.info(f"应用启动 - 日志级别: {settings.LOG_LEVEL}")

    # Sentry初期化（エラー監視）
    if settings.SENTRY_DSN and settings.SENTRY_DSN.strip():
        try:
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                integrations=[FlaskIntegration()],
                environment=settings.SENTRY_ENVIRONMENT,
                traces_sample_rate=0.1,
            )
            logger.info("Sentry初期化完了", environment=settings.SENTRY_ENVIRONMENT)
        except Exception as e:
            logger.warning("Sentry初期化に失敗", error=str(e))
    else:
        logger.info("Sentry DSNが設定されていません。エラー監視は無効です。")

    # CORS設定
    CORS(app,
         origins=settings.CORS_ORIGINS,
         methods=settings.CORS_METHODS,
         allow_headers=['Content-Type', 'Authorization'])

    # レート制限設定
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        storage_uri=settings.RATELIMIT_STORAGE_URL,
        default_limits=[settings.RATELIMIT_DEFAULT]
    )

    # セキュリティヘッダー設定
    try:
        Talisman(app,
                 force_https=settings.FORCE_HTTPS,
                 csp=settings.CONTENT_SECURITY_POLICY)
        logger.info("Talisman セキュリティヘッダー設定完了")
    except Exception as e:
        logger.warning("Talisman 設定に失敗、基本的なセキュリティヘッダーを手動設定", error=str(e))

        @app.after_request
        def add_security_headers(response):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            if settings.FORCE_HTTPS:
                response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            return response

    @app.route('/')
    def index():
        """ヘルスチェック用エンドポイント"""
        is_connected = oci_client.is_connected()
        return jsonify({
            'status': 'running',
            'oci_connected': is_connected,
            'message': 'OK' if is_connected else 'OCI接続が初期化されていません',
            'endpoints': {
                'image_proxy': '/img/<bucket>/<object_name>',
                'upload': '/upload (POST)',
                'health': '/health',
                'test': '/test'
            }
        })

    @app.route('/test')
    def test_page():
        """テストページを提供"""
        return send_from_directory('.', 'test.html')

    @app.route('/test.html')
    def test_upload_page():
        """テストアップロードページを提供"""
        return send_from_directory('.', 'test.html')

    @app.route('/img/<bucket>/<path:obj>')
    @limiter.limit("50 per minute")
    def serve_image(bucket, obj):
        """
        OCI Object Storageから画像を取得して返すプロキシエンドポイント

        Args:
            bucket: バケット名
            obj: オブジェクト名（パス含む）

        Returns:
            画像データまたはエラーレスポンス
        """
        try:
            # OCI接続チェック
            if not oci_client.is_connected():
                logger.error("画像取得失敗 - OCI接続エラー")
                return jsonify({'error': 'OCI接続エラー'}), 500

            logger.info("画像取得開始", bucket=bucket, object=obj)

            # オブジェクトを取得
            response = oci_client.get_object(bucket, obj)

            # Content-Typeを取得（デフォルトは画像として扱う）
            content_type = response.headers.get('Content-Type', 'image/jpeg')

            logger.info("画像取得成功", object=obj, content_type=content_type)

            return Response(
                response.data.content,
                mimetype=content_type,
                headers={
                    'Cache-Control': 'max-age=3600',  # 1時間キャッシュ
                    'Content-Disposition': f'inline; filename="{obj.split("/")[-1]}"'
                }
            )

        except ServiceError as e:
            if e.status == 404:
                logger.warning("画像が見つかりません", bucket=bucket, object=obj)
                return jsonify({'error': '画像が見つかりません'}), 404
            else:
                logger.error("OCI サービスエラー", error=str(e))
                return jsonify({'error': 'OCI サービスエラー'}), 500
        except Exception as e:
            logger.error("画像取得中に予期しないエラー", error=str(e))
            return jsonify({'error': '画像取得に失敗しました'}), 500

    @app.route('/upload', methods=['POST'])
    @limiter.limit(settings.RATELIMIT_UPLOAD)
    def upload_image():
        """
        画像をOCI Object Storageにアップロードするエンドポイント

        Request:
            - file: アップロードする画像ファイル (ファイルアップロード)
            - url: 画像のHTTP URL (URLからダウンロード)
            - bucket (optional): アップロード先バケット名
            - folder (optional): アップロード先フォルダ
            - filename (optional): カスタムファイル名（拡張子を含む）
            
        Note: fileとurlのどちらか一方を指定してください
        Note: filenameが指定されない場合、ユニークなファイル名が自動生成されます

        Returns:
            アップロード結果とアクセス用URL
        """
        try:
            # OCI接続チェック
            if not oci_client.is_connected():
                logger.error("アップロード失敗 - OCI接続エラー")
                return jsonify({'error': 'OCI接続エラー'}), 500

            # リクエストデータの取得（JSON または form data）
            if request.is_json:
                data = request.get_json()
                bucket = data.get('bucket', settings.OCI_BUCKET)
                folder = data.get('folder', '')
                image_url = data.get('url')
                filename = data.get('filename')
            else:
                bucket = request.form.get('bucket', settings.OCI_BUCKET)
                folder = request.form.get('folder', '')
                image_url = request.form.get('url')
                filename = request.form.get('filename')
            
            logger.info(f"{image_url=}")
            logger.info(f"{filename=}")
            
            if image_url:
                # URLからの画像ダウンロード
                logger.info("URL画像ダウンロード開始", url=image_url)
                
                # URL検証
                is_valid, error_message = is_valid_image_url(image_url)
                if not is_valid:
                    logger.error("URL検証失敗", url=image_url, error=error_message)
                    return jsonify({'error': f'無効なURLまたは安全でないURLです: {error_message}'}), 400
                
                try:
                    # 画像をダウンロード
                    image_data, content_type, downloaded_filename = download_image_from_url(image_url)
                    file_size = len(image_data.getvalue())
                    logger.info(f"{downloaded_filename=}")
                    logger.info(f"{file_size=}")
                    
                    # ファイル拡張子チェック
                    if not allowed_file(downloaded_filename):
                        return jsonify({
                            'error': f'許可されていないファイル形式です。許可形式: {", ".join(settings.ALLOWED_EXTENSIONS)}'
                        }), 400
                    
                    logger.info("URL画像ダウンロード成功", 
                               url=image_url, 
                               size=file_size, 
                               content_type=content_type)
                    
                except requests.RequestException as e:
                    logger.error("画像ダウンロードエラー", url=image_url, error=str(e))
                    return jsonify({'error': '画像のダウンロードに失敗しました'}), 400
                except ValueError as e:
                    logger.error("画像検証エラー", url=image_url, error=str(e))
                    return jsonify({'error': str(e)}), 400
                
                # ユニークなオブジェクト名を生成
                if filename and '.' in filename:
                    # カスタムファイル名が指定された場合
                    unique_filename = filename
                else:
                    # デフォルトのユニークファイル名を生成
                    file_extension = downloaded_filename.rsplit('.', 1)[1].lower()
                    unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
                
            else:
                # 従来のファイルアップロード
                if 'file' not in request.files:
                    return jsonify({'error': 'ファイルまたはURLを指定してください'}), 400

                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'ファイル名が空です'}), 400

                # ファイル拡張子チェック
                if not allowed_file(file.filename):
                    return jsonify({
                        'error': f'許可されていないファイル形式です。許可形式: {", ".join(settings.ALLOWED_EXTENSIONS)}'
                    }), 400

                # ファイルサイズチェック
                file.seek(0, 2)  # ファイル末尾に移動
                file_size = file.tell()
                file.seek(0)  # ファイル先頭に戻る

                if file_size > settings.MAX_CONTENT_LENGTH:
                    max_size_mb = settings.MAX_CONTENT_LENGTH // (1024*1024)
                    return jsonify({
                        'error': f'ファイルサイズが大きすぎます。最大サイズ: {max_size_mb}MB'
                    }), 400

                # ユニークなオブジェクト名を生成
                file_extension = file.filename.rsplit('.', 1)[1].lower()
                if filename and '.' in filename:
                    # カスタムファイル名が指定された場合
                    unique_filename = filename
                else:
                    # デフォルトのユニークファイル名を生成
                    unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
                
                # ファイルデータとコンテンツタイプを設定
                image_data = file.stream
                content_type = file.content_type or 'application/octet-stream'

            # フォルダが指定されている場合はパスに含める
            if folder:
                object_name = f"{folder.strip('/')}/{unique_filename}"
            else:
                object_name = unique_filename

            logger.info("アップロード開始",
                       bucket=bucket,
                       object=object_name,
                       size=file_size,
                       source="url" if image_url else "file")

            # OCI Object Storageにアップロード
            oci_client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=image_data,
                content_type=content_type
            )
            time.sleep(7)

            # プロキシURL生成
            proxy_url = f"/img/{bucket}/{object_name}"

            logger.info("アップロード成功", object=object_name)

            return jsonify({
                'success': True,
                'message': 'アップロードが完了しました',
                'data': {
                    'object_name': object_name,
                    'bucket': bucket,
                    'proxy_url': proxy_url,
                    'file_size': file_size,
                    'content_type': content_type,
                    'source': 'url' if image_url else 'file',
                    'source_url': image_url if image_url else None,
                    'uploaded_at': datetime.now().isoformat()
                }
            })

        except requests.RequestException as e:
            logger.error("HTTP リクエストエラー", error=str(e))
            return jsonify({'error': '画像のダウンロードに失敗しました'}), 400
        except requests.Timeout as e:
            logger.error("HTTP タイムアウトエラー", error=str(e))
            return jsonify({'error': '画像のダウンロードがタイムアウトしました'}), 408
        except ValueError as e:
            logger.error("画像検証エラー", error=str(e))
            return jsonify({'error': str(e)}), 400
        except ServiceError as e:
            logger.error("OCI サービスエラー", error=str(e))
            return jsonify({'error': 'アップロードに失敗しました（OCI エラー）'}), 500
        except Exception as e:
            logger.error("アップロード中に予期しないエラー", error=str(e))
            return jsonify({'error': 'アップロードに失敗しました'}), 500

    @app.route('/vectorize', methods=['POST'])
    @limiter.limit("10 per minute")
    def vectorize_image():
        """
        画像をベクトル化するエンドポイント
        
        Request:
            - bucket: バケット名
            - object_name: オブジェクト名
            または
            - file: 画像ファイル (直接アップロード)
            - url: 画像のHTTP URL
            
        Returns:
            ベクトル化結果と埋め込みベクトルID
        """
        try:
            # クライアント接続チェック
            if not image_embedder.is_initialized():
                logger.error("ベクトル化失敗 - Oracle Generative AI接続エラー")
                return jsonify({'error': 'Oracle Generative AI接続エラー'}), 500
                
            if not db_client.is_connected():
                logger.error("ベクトル化失敗 - データベース接続エラー")
                return jsonify({'error': 'データベース接続エラー'}), 500

            # リクエストデータの取得
            if request.is_json:
                data = request.get_json()
                bucket = data.get('bucket', settings.OCI_BUCKET)
                object_name = data.get('filename')
                image_url = data.get('url')
            else:
                bucket = request.form.get('bucket', settings.OCI_BUCKET)
                object_name = request.form.get('filename')
                image_url = request.form.get('url')

            image_data = None
            content_type = None
            file_size = 0

            if bucket and object_name:
                # OCI Object Storageから画像を取得
                try:
                    if not oci_client.is_connected():
                        return jsonify({'error': 'OCI接続エラー'}), 500
                        
                    response = oci_client.get_object(bucket, object_name)
                    image_data = io.BytesIO(response.data.content)
                    content_type = response.headers.get('Content-Type', 'image/jpeg')
                    file_size = len(response.data.content)
                    
                    logger.info("OCI画像取得成功", bucket=bucket, object_name=object_name, size=file_size)
                    
                except ServiceError as e:
                    if e.status == 404:
                        return jsonify({'error': '指定された画像が見つかりません'}), 404
                    else:
                        logger.error("OCI画像取得エラー", error=str(e))
                        return jsonify({'error': 'OCI画像取得エラー'}), 500
                        
            elif image_url:
                # URLから画像をダウンロード
                try:
                    # URL検証
                    is_valid, error_message = is_valid_image_url(image_url)
                    if not is_valid:
                        return jsonify({'error': f'無効なURL: {error_message}'}), 400
                    
                    image_data, content_type, filename = download_image_from_url(image_url)
                    file_size = len(image_data.getvalue())
                    
                    # ファイル拡張子チェック
                    if not allowed_file(filename):
                        return jsonify({
                            'error': f'許可されていないファイル形式です。許可形式: {", ".join(settings.ALLOWED_EXTENSIONS)}'
                        }), 400
                    
                    # オブジェクト名を設定（filenameがある場合はそれを使用、なければ一時的な名前を生成）
                    if filename:
                        object_name = filename
                    else:
                        file_extension = filename.rsplit('.', 1)[1].lower() if filename else 'jpg'
                        object_name = f"temp_{uuid.uuid4().hex}.{file_extension}"
                    
                    logger.info("URL画像取得成功", url=image_url, size=file_size)
                    
                except Exception as e:
                    logger.error("URL画像取得エラー", url=image_url, error=str(e))
                    return jsonify({'error': '画像の取得に失敗しました'}), 400
                    
            elif 'file' in request.files:
                # 直接ファイルアップロード
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'ファイル名が空です'}), 400
                
                # ファイル拡張子チェック
                if not allowed_file(file.filename):
                    return jsonify({
                        'error': f'許可されていないファイル形式です。許可形式: {", ".join(settings.ALLOWED_EXTENSIONS)}'
                    }), 400
                
                # ファイルサイズチェック
                file.seek(0, 2)
                file_size = file.tell()
                file.seek(0)
                
                if file_size > settings.MAX_CONTENT_LENGTH:
                    max_size_mb = settings.MAX_CONTENT_LENGTH // (1024*1024)
                    return jsonify({
                        'error': f'ファイルサイズが大きすぎます。最大サイズ: {max_size_mb}MB'
                    }), 400
                
                image_data = io.BytesIO(file.read())
                content_type = file.content_type or 'application/octet-stream'
                
                # オブジェクト名を設定（filenameがある場合はそれを使用、なければ一時的な名前を生成）
                if file.filename:
                    object_name = file.filename
                else:
                    file_extension = file.filename.rsplit('.', 1)[1].lower() if file.filename else 'jpg'
                    object_name = f"temp_{uuid.uuid4().hex}.{file_extension}"
                
                logger.info("ファイル画像取得成功", filename=file.filename, size=file_size)
                
            else:
                return jsonify({'error': 'bucket/object_name、url、またはfileのいずれかを指定してください'}), 400

            # ベクトル化実行
            logger.info("画像ベクトル化開始", bucket=bucket, object_name=object_name)
            
            embedding = image_embedder.generate_embedding(image_data, content_type)
            if embedding is None:
                logger.error("画像ベクトル化失敗", object_name=object_name)
                return jsonify({'error': '画像のベクトル化に失敗しました'}), 500
            
            # データベースに保存
            embedding_id = db_client.insert_embedding(
                bucket=bucket,
                object_name=object_name,
                content_type=content_type,
                file_size=file_size,
                embedding=embedding
            )
            
            if embedding_id is None:
                logger.error("埋め込みベクトル保存失敗", object_name=object_name)
                return jsonify({'error': '埋め込みベクトルの保存に失敗しました'}), 500
            
            logger.info("画像ベクトル化成功", embedding_id=embedding_id, object_name=object_name)
            
            return jsonify({
                'success': True,
                'message': '画像のベクトル化が完了しました',
                'data': {
                    'embedding_id': embedding_id,
                    'bucket': bucket,
                    'object_name': object_name,
                    'file_size': file_size,
                    'content_type': content_type,
                    'embedding_shape': embedding.shape,
                    'embedding_norm': float(np.linalg.norm(embedding)),
                    'vectorized_at': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error("ベクトル化処理中に予期しないエラー", error=str(e))
            return jsonify({'error': 'ベクトル化処理に失敗しました'}), 500

    @app.route('/search', methods=['POST'])
    @limiter.limit("20 per minute")
    def search_similar_images():
        """
        テキストクエリに基づいて類似画像を検索するエンドポイント
        
        Request:
            - query: 検索クエリテキスト (必須)
            - limit: 返す結果の最大数 (オプション、デフォルト: 10)
            - threshold: 類似度の閾値 (オプション、デフォルト: 0.7)
            
        Returns:
            類似画像のリスト（bucket, object_name, vector_distance）
        """
        try:
            # データベース接続チェック
            if not db_client.is_connected():
                logger.error("検索失敗 - データベース接続エラー")
                return jsonify({'error': 'データベース接続エラー'}), 500
            
            # リクエストデータの取得
            if request.is_json:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'JSONデータが必要です'}), 400
                query = data.get('query')
                limit = data.get('limit', 10)
                threshold = data.get('threshold', 0.7)
            else:
                query = request.form.get('query')
                limit = int(request.form.get('limit', 10))
                threshold = float(request.form.get('threshold', 0.7))
            
            # クエリパラメータの検証
            if not query or not query.strip():
                return jsonify({'error': 'クエリテキストが必要です'}), 400
            
            if not isinstance(limit, int) or limit <= 0 or limit > 100:
                return jsonify({'error': 'limitは1から100の間の整数である必要があります'}), 400
            
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                return jsonify({'error': 'thresholdは0.0から1.0の間の数値である必要があります'}), 400
            
            logger.info("画像検索開始", query=query, limit=limit, threshold=threshold)
            
            # テキストから埋め込みベクトルを生成（3回リトライ）
            query_embedding = text_embedder.generate_embedding(query.strip())
            
            if query_embedding is None:
                logger.error("テキスト埋め込み生成失敗", query=query)
                return jsonify({'error': 'テキストの埋め込み生成に失敗しました'}), 500
            
            # 類似画像検索を実行
            results = db_client.search_similar_images(
                query_embedding=query_embedding,
                limit=limit,
                threshold=threshold
            )
            
            if results is None:
                logger.error("画像検索失敗", query=query)
                return jsonify({'error': '画像検索に失敗しました'}), 500
            
            # プロキシURLを追加
            for result in results:
                result['proxy_url'] = f"/img/{result['bucket']}/{result['object_name']}"
            
            logger.info("画像検索成功", 
                       query=query, 
                       results_count=len(results),
                       threshold=threshold)
            
            return jsonify({
                'success': True,
                'message': f'検索が完了しました。{len(results)}件の類似画像が見つかりました。',
                'data': {
                    'query': query,
                    'limit': limit,
                    'threshold': threshold,
                    'results_count': len(results),
                    'results': results,
                    'searched_at': datetime.now().isoformat()
                }
            })
            
        except ValueError as e:
            logger.error("パラメータエラー", query=query, error=str(e))
            return jsonify({'error': f'パラメータエラー: {str(e)}'}), 400
        except Exception as e:
            logger.error("検索処理中に予期しないエラー", query=query, error=str(e))
            return jsonify({'error': '検索処理に失敗しました'}), 500

    @app.route('/api/upload/ppt', methods=['POST'])
    @limiter.limit(settings.RATELIMIT_UPLOAD)
    def upload_ppt():
        """
        PPT/PPTXファイルをOCI Object Storageにアップロードするエンドポイント
        
        Request:
            - file: アップロードするPPT/PPTXファイル (必須)
            - bucket (optional): アップロード先バケット名
            - folder (optional): アップロード先フォルダ
            - filename (optional): カスタムファイル名（拡張子を含む）
            
        Returns:
            アップロード結果とアクセス用URL
        """
        logger.info("PPT/PPTXファイルアップロードリクエストを受信しました")
        
        try:
            # OCI接続チェック
            if not oci_client.is_connected():
                logger.error("アップロード失敗 - OCI接続エラー")
                response, status_code = create_response(
                    success=False,
                    message='OCI接続エラー',
                    status_code=500
                )
                return jsonify(response), status_code
            
            # ファイルがリクエストに含まれているかチェック
            if 'file' not in request.files:
                logger.warning("ファイルがリクエストに含まれていません")
                response, status_code = create_response(
                    success=False,
                    message='ファイルが指定されていません',
                    status_code=400
                )
                return jsonify(response), status_code
            
            file = request.files['file']
            
            # ファイルが選択されているかチェック
            if file.filename == '':
                logger.warning("ファイルが選択されていません")
                response, status_code = create_response(
                    success=False,
                    message='ファイルが選択されていません',
                    status_code=400
                )
                return jsonify(response), status_code
            
            # ファイル拡張子をチェック
            if not allowed_file(file.filename):
                logger.warning(f"サポートされていないファイル形式: {file.filename}")
                response, status_code = create_response(
                    success=False,
                    message=f'サポートされていないファイル形式です。許可される形式: {", ".join(settings.ALLOWED_EXTENSIONS)}',
                    status_code=400
                )
                return jsonify(response), status_code
            
            # PPT/PPTX専用チェック
            file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
            if file_extension not in ['ppt', 'pptx']:
                logger.warning(f"PPT/PPTX以外のファイル形式: {file.filename}")
                response, status_code = create_response(
                    success=False,
                    message='PPT/PPTXファイルのみアップロード可能です',
                    status_code=400
                )
                return jsonify(response), status_code
            
            # ファイルサイズチェック
            file.seek(0, 2)  # ファイル末尾に移動
            file_size = file.tell()
            file.seek(0)  # ファイル先頭に戻る
            
            if file_size > settings.MAX_CONTENT_LENGTH:
                max_size_mb = settings.MAX_CONTENT_LENGTH // (1024*1024)
                logger.warning(f"ファイルサイズが大きすぎます: {file_size} bytes")
                response, status_code = create_response(
                    success=False,
                    message=f'ファイルサイズが大きすぎます。最大サイズ: {max_size_mb}MB',
                    status_code=400
                )
                return jsonify(response), status_code
            
            # リクエストパラメータの取得
            bucket = request.form.get('bucket', settings.OCI_BUCKET)
            folder = request.form.get('folder', '')
            custom_filename = request.form.get('filename')
            
            # ユニークなオブジェクト名を生成
            if custom_filename and '.' in custom_filename:
                # カスタムファイル名が指定された場合
                unique_filename = custom_filename
            else:
                # デフォルトのユニークファイル名を生成
                unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
            
            # フォルダが指定されている場合はパスに含める
            if folder:
                object_name = f"{folder.strip('/')}/{unique_filename}"
            else:
                object_name = unique_filename
            
            # ファイルデータとコンテンツタイプを設定
            file_data = file.stream
            content_type = file.content_type or 'application/vnd.ms-powerpoint'
            
            # PPTXの場合はより適切なMIMEタイプを設定
            if file_extension == 'pptx':
                content_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            elif file_extension == 'ppt':
                content_type = 'application/vnd.ms-powerpoint'
            
            logger.info("PPT/PPTXアップロード開始",
                       bucket=bucket,
                       object=object_name,
                       size=file_size,
                       content_type=content_type)
            
            # OCI Object Storageにアップロード
            oci_client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=file_data,
                content_type=content_type
            )
            
            # プロキシURL生成
            proxy_url = f"/img/{bucket}/{object_name}"
            
            logger.info("PPT/PPTXアップロード成功", object=object_name)
            
            response, status_code = create_response(
                success=True,
                message='PPT/PPTXファイルのアップロードが完了しました',
                data={
                    'object_name': object_name,
                    'bucket': bucket,
                    'proxy_url': proxy_url,
                    'file_size': file_size,
                    'content_type': content_type,
                    'file_extension': file_extension,
                    'uploaded_at': datetime.now().isoformat()
                }
            )
            return jsonify(response), status_code
            
        except ServiceError as e:
            logger.error("OCI サービスエラー", error=str(e))
            response, status_code = create_response(
                success=False,
                message='アップロードに失敗しました（OCI エラー）',
                status_code=500
            )
            return jsonify(response), status_code
        except Exception as e:
            logger.error("PPT/PPTXアップロード中に予期しないエラー", error=str(e))
            response, status_code = create_response(
                success=False,
                message='アップロードに失敗しました',
                status_code=500
            )
            return jsonify(response), status_code

    @app.route('/health')
    def health_check():
        """ヘルスチェック用エンドポイント"""
        is_connected = oci_client.is_connected()
        db_connected = db_client.is_connected()
        embedder_initialized = image_embedder.is_initialized()
        
        return jsonify({
            'status': 'healthy' if (is_connected and db_connected and embedder_initialized) else 'unhealthy',
            'oci_connection': 'OK' if is_connected else 'OCI接続が初期化されていません',
            'database_connection': 'OK' if db_connected else 'データベース接続エラー',
            'embedder_status': 'OK' if embedder_initialized else 'Oracle Generative AI接続エラー',
            'timestamp': datetime.now().isoformat()
        }), 200 if (is_connected and db_connected and embedder_initialized) else 503

    # エラーハンドラー
    @app.errorhandler(413)
    def too_large(e):
        """ファイルサイズ制限エラーハンドラ"""
        logger.warning("ファイルサイズ制限エラー")
        return jsonify({'error': 'ファイルサイズが大きすぎます'}), 413

    @app.errorhandler(404)
    def not_found(e):
        """404エラーハンドラ"""
        return jsonify({'error': 'エンドポイントが見つかりません'}), 404

    @app.errorhandler(500)
    def internal_error(e):
        """500エラーハンドラ"""
        logger.error("内部サーバーエラー", error=str(e))
        return jsonify({'error': '内部サーバーエラーが発生しました'}), 500

    @app.errorhandler(ServiceError)
    def handle_oci_error(e):
        """OCI サービスエラーハンドラ"""
        logger.error("OCI サービスエラー",
                    status=e.status,
                    code=e.code,
                    message=e.message)
        return jsonify({
            'error': 'OCI サービスエラーが発生しました',
            'details': e.message if settings.DEBUG else None
        }), 500

    return app


def create_production_app() -> Flask:
    """本番環境用アプリケーション作成"""
    return create_app('production')


def create_development_app() -> Flask:
    """開発環境用アプリケーション作成"""
    return create_app('development')


def create_testing_app() -> Flask:
    """テスト環境用アプリケーション作成"""
    return create_app('testing')


# デフォルトアプリケーション（開発用）
app = create_development_app()


if __name__ == '__main__':
    # 開発環境での実行
    env = os.getenv('FLASK_ENV', 'development')
    port = int(os.getenv('PORT', settings.PORT))
    debug = settings.DEBUG

    logger.info("アプリケーション開始",
               environment=env,
               port=port,
               debug=debug)

    app.run(host=settings.HOST, port=port, debug=debug)
