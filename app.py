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
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
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


def allowed_file(filename: str) -> bool:
    """許可されたファイル拡張子かチェック"""
    if not filename or '.' not in filename:
        return False

    extension = filename.rsplit('.', 1)[1].lower()
    return extension in settings.ALLOWED_EXTENSIONS


def is_valid_image_url(url: str) -> bool:
    """有効な画像URLかチェック（SSRF攻撃防止）"""
    try:
        parsed = urlparse(url)
        # HTTPSまたはHTTPのみ許可
        if parsed.scheme not in ['http', 'https']:
            return False
        # ローカルIPアドレスやプライベートIPアドレスを拒否
        import ipaddress
        import socket
        
        hostname = parsed.hostname
        if not hostname:
            return False
            
        # IPアドレスの場合はプライベートIPを拒否
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            # ホスト名の場合はDNS解決してIPをチェック
            try:
                ip_str = socket.gethostbyname(hostname)
                ip = ipaddress.ip_address(ip_str)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False
            except (socket.gaierror, ValueError):
                return False
                
        return True
    except Exception:
        return False


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
            
        Note: fileとurlのどちらか一方を指定してください

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
            else:
                bucket = request.form.get('bucket', settings.OCI_BUCKET)
                folder = request.form.get('folder', '')
                image_url = request.form.get('url')
            
            logger.info(f"{image_url=}")
            
            if image_url:
                # URLからの画像ダウンロード
                logger.info("URL画像ダウンロード開始", url=image_url)
                
                # URL検証
                if not is_valid_image_url(image_url):
                    return jsonify({'error': '無効なURLまたは安全でないURLです'}), 400
                
                try:
                    # 画像をダウンロード
                    image_data, content_type, filename = download_image_from_url(image_url)
                    file_size = len(image_data.getvalue())
                    logger.info(f"{filename=}")
                    logger.info(f"{file_size=}")
                    
                    # ファイル拡張子チェック
                    if not allowed_file(filename):
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
                file_extension = filename.rsplit('.', 1)[1].lower()
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

    @app.route('/health')
    def health_check():
        """ヘルスチェック用エンドポイント"""
        is_connected = oci_client.is_connected()
        return jsonify({
            'status': 'healthy' if is_connected else 'unhealthy',
            'oci_connection': 'OK' if is_connected else 'OCI接続が初期化されていません',
            'timestamp': datetime.now().isoformat()
        }), 200 if is_connected else 503

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
