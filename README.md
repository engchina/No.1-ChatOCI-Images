# 🖼️ ChatOCI Images - Oracle Cloud Storage 画像プロキシアプリ

Oracle Cloud Infrastructure (OCI) Object Storage に保存された画像を認証付きで表示・アップロードするための Flask アプリケーションです。

**業界ベストプラクティスに基づく実装:**
- 🔒 **セキュリティファースト**: Flask-Talisman によるセキュリティヘッダー、CORS 対応
- 🚀 **パフォーマンス**: レート制限、キャッシュ、構造化ログ
- 🛡️ **信頼性**: 型安全な設定管理、包括的なエラーハンドリング
- 📊 **監視**: Sentry 統合、ヘルスチェック、メトリクス
- 🐳 **デプロイ**: Docker 対応、本番環境向け設定

## 📋 機能

### コア機能
- **画像プロキシ**: OCI Object Storage の画像を認証付きで安全に表示
- **画像アップロード**: Web インターフェースから OCI Object Storage へ画像をアップロード
- **セキュリティ**: OCI 認証情報をバックエンドで管理し、フロントエンドに露出しない

### セキュリティ機能
- **レート制限**: Flask-Limiter による API 呼び出し制限
- **セキュリティヘッダー**: CSP、HSTS、XSS 保護など
- **CORS 対応**: クロスオリジンリクエストの適切な制御
- **入力検証**: ファイル形式・サイズの厳格な検証

### 運用機能
- **構造化ログ**: JSON 形式の詳細なログ出力
- **エラー監視**: Sentry による本番環境でのエラー追跡
- **ヘルスチェック**: アプリケーションと OCI 接続の状態監視
- **設定管理**: Pydantic による型安全な設定管理

## 🚀 API エンドポイント

### 1. 画像プロキシ
```
GET /img/<bucket>/<object_name>
```
OCI Object Storage から画像を取得して返します。

**例:**
```
GET /img/chatbot-images/avatars/user123.jpg
```

### 2. 画像アップロード
```
POST /upload
```
画像ファイルを OCI Object Storage にアップロードします。

**パラメータ:**
- `file`: アップロードする画像ファイル（必須）
- `bucket`: アップロード先バケット名（オプション、デフォルト: chatbot-images）
- `folder`: アップロード先フォルダ（オプション）

**レスポンス例:**
```json
{
  "success": true,
  "message": "アップロードが完了しました",
  "data": {
    "object_name": "abc123def.jpg",
    "bucket": "chatbot-images",
    "proxy_url": "/img/chatbot-images/abc123def.jpg",
    "file_size": 1024000,
    "content_type": "image/jpeg",
    "uploaded_at": "2024-01-01T12:00:00"
  }
}
```

## 🛠️ セットアップ

### 1. 依存関係のインストール

#### Conda 環境の作成（推奨）
```bash
conda create -n no.1-chatoci-images python=3.12 -y
conda activate no.1-chatoci-images
pip install -r requirements.txt
# pip list --format=freeze > requirements.txt
```

#### Python 仮想環境の作成（代替方法）
```bash
# Python 3.12 を使用して仮想環境を作成
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate     # Windows

pip install -r requirements.txt
# pip list --format=freeze > requirements.txt
```

### 2. 環境設定

#### 環境変数ファイルの作成
```bash
cp .env.example .env
# .env ファイルを編集して適切な値を設定
```

#### 主要な設定項目
```bash
# OCI 設定
OCI_CONFIG_FILE=~/.oci/config
OCI_PROFILE=DEFAULT
OCI_BUCKET=your-bucket-name
OCI_REGION=us-chicago-1

# セキュリティ
SECRET_KEY=your-super-secret-key
SESSION_COOKIE_SECURE=true  # HTTPS環境では true

# 本番環境設定
FLASK_ENV=production
SENTRY_DSN=your-sentry-dsn  # エラー監視（オプション）
```

### 3. OCI 設定

#### OCI 設定ファイルの作成
`~/.oci/config` ファイルを作成：
```ini
[DEFAULT]
user=ocid1.user.oc1..aaaaaaaa...
fingerprint=12:34:56:78:90:ab:cd:ef...
key_file=~/.oci/oci_api_key.pem
tenancy=ocid1.tenancy.oc1..aaaaaaaa...
region=us-chicago-1
```

#### OCI IAM ポリシー設定
チャットボット用のグループとユーザーを作成し、以下のポリシーを設定：

```
Allow group chatbot-app to read objects in compartment <compartment-name>
Allow group chatbot-app to manage objects in compartment <compartment-name> where request.permission='OBJECT_CREATE'
```

### 4. アプリケーション起動

#### 開発環境
```bash
# 直接起動
python app.py

# または WSGI サーバー使用
python wsgi.py
```

#### 本番環境
```bash
# Gunicorn での起動
gunicorn --config gunicorn.conf.py wsgi:application

# または Docker 使用
docker-compose up -d
```

#### Docker での起動
```bash
# 開発環境
docker-compose up

# 本番環境
docker-compose -f docker-compose.prod.yml up -d
```

## 🧪 テスト

### アップロードテスト
ブラウザで `test.html` を開いてアップロード機能をテストできます：

```bash
# 開発サーバー起動後
open http://localhost:5000/test
# または
open http://localhost:5000/test.html
```

### API テスト
```bash
# ヘルスチェック
curl http://localhost:5000/health

# 画像アップロード
curl -X POST -F "file=@test.jpg" -F "bucket=my-bucket" http://localhost:5000/upload

# 画像取得
curl http://localhost:5000/img/my-bucket/test.jpg
```

## 📁 ファイル構成

```
.
├── app.py                    # メインアプリケーション（Flask アプリファクトリ）
├── config.py                 # 設定管理（Pydantic Settings）
├── wsgi.py                   # WSGI エントリーポイント
├── requirements.txt          # Python 依存関係
├── test_app.py              # テストスイート
├── test.html         # アップロードテスト用HTML
├── .env.example             # 環境変数設定例
├── Dockerfile               # Docker イメージ定義
├── docker-compose.yml       # Docker Compose 設定
├── nginx.conf               # Nginx 設定（リバースプロキシ）
└── README.md               # このファイル
```

### アーキテクチャ概要

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask App     │    │   OCI Object    │
│   (Browser)     │◄──►│   (Python)      │◄──►│   Storage       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Monitoring    │
                       │   (Sentry)      │
                       └─────────────────┘
```

## ⚙️ 設定オプション

### 環境変数
- `OCI_CONFIG_FILE`: OCI設定ファイルのパス（デフォルト: ~/.oci/config）
- `OCI_PROFILE`: OCI設定プロファイル名（デフォルト: DEFAULT）
- `OCI_BUCKET`: デフォルトバケット名（デフォルト: chatbot-images）
- `PORT`: サーバーポート（デフォルト: 5000）
- `FLASK_DEBUG`: デバッグモード（デフォルト: False）

### アップロード制限
- **最大ファイルサイズ**: 10MB
- **許可形式**: PNG, JPG, JPEG, GIF, WebP, BMP

## 🔒 セキュリティ考慮事項

1. **認証情報の保護**: OCI 認証情報はサーバーサイドでのみ管理
2. **ファイル検証**: アップロードファイルの形式とサイズを検証
3. **エラー情報**: 詳細なエラー情報の露出を制限
4. **CORS**: 必要に応じて CORS 設定を追加

## 🐛 トラブルシューティング

### OCI 接続エラー
```
OCI接続が初期化されていません
```
- OCI 設定ファイルのパスと内容を確認
- API キーファイルの権限を確認（600推奨）
- ネットワーク接続を確認

### アップロードエラー
```
ファイルサイズが大きすぎます
```
- ファイルサイズを10MB以下に縮小
- 許可されたファイル形式を確認

### 画像表示エラー
```
画像が見つかりません
```
- バケット名とオブジェクト名を確認
- OCI IAM ポリシーの読み取り権限を確認

## 📝 ライセンス


このプロジェクトは MIT ライセンスの下で公開されています。
