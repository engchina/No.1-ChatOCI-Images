#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle Cloud Storage Image Proxy Application
Flask app to display and upload images from OCI Object Storage with authentication

Implementation based on industry best practices:
- Type-safe configuration management (Pydantic)
- Structured logging (structlog)
- Security headers (Flask-Talisman)
- Rate limiting (Flask-Limiter)
- CORS support (Flask-CORS)
- Error monitoring (Sentry)
"""

import os
import uuid
import structlog
from datetime import datetime
from typing import Optional, Tuple
import time
import io
import requests
from urllib.parse import urlparse, quote, unquote
import mimetypes
import subprocess
import shutil
import tempfile

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

# Configure standard library logging module
import logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT
)

# Configure structured logging
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
    """OCI Object Storage client wrapper"""

    def __init__(self):
        self.client: Optional[ObjectStorageClient] = None
        self.namespace: Optional[str] = None
        self._initialize()

    def _initialize(self):
        """Initialize OCI client"""
        try:
            # Load OCI configuration
            config_file = os.path.expanduser(settings.OCI_CONFIG_FILE)
            if not os.path.exists(config_file):
                logger.warning("OCI config file not found", config_file=config_file)
                return

            config = oci.config.from_file(
                file_location=config_file,
                profile_name=settings.OCI_PROFILE
            )

            # Configure region
            if settings.OCI_REGION:
                config['region'] = settings.OCI_REGION

            self.client = ObjectStorageClient(config)
            self.namespace = self.client.get_namespace().data

            logger.info("OCI connection successful",
                       namespace=self.namespace,
                       region=config.get('region'))

        except Exception as e:
            logger.error("Failed to initialize OCI configuration", error=str(e))
            self.client = None
            self.namespace = None

    def is_connected(self) -> bool:
        """Check connection status"""
        return self.client is not None and self.namespace is not None

    def get_object(self, bucket_name: str, object_name: str):
        """Get object"""
        if not self.is_connected():
            raise RuntimeError("OCI client is not initialized")

        return self.client.get_object(
            namespace_name=self.namespace,
            bucket_name=bucket_name,
            object_name=object_name
        )

    def put_object(self, bucket_name: str, object_name: str, data, content_type: str = None, metadata: dict = None):
        """Upload object
        
        Args:
            bucket_name: Bucket name
            object_name: Object name (supports Japanese and spaces)
            data: File data
            content_type: Content type
            metadata: Custom metadata (e.g., original filename)
        """
        if not self.is_connected():
            raise RuntimeError("OCI client is not initialized")

        # Prepare metadata
        opc_meta = metadata if metadata else {}
        
        return self.client.put_object(
            namespace_name=self.namespace,
            bucket_name=bucket_name,
            object_name=object_name,
            put_object_body=data,
            content_type=content_type,
            opc_meta=opc_meta
        )


# Global OCI client
oci_client = OCIClient()


class DatabaseClient:
    """Oracle database client"""
    
    def __init__(self):
        self.connection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database connection"""
        try:
            self.connection = oracledb.connect(
                user=settings.DB_USERNAME,
                password=settings.DB_PASSWORD,
                dsn=settings.DB_DSN
            )
            logger.info("Oracle database connection successful", dsn=settings.DB_DSN)
            # Check and create table if not exists
            self._ensure_table_exists()
        except Exception as e:
            logger.error("Oracle database connection failed", error=str(e))
            self.connection = None
    
    def _reconnect_with_retry(self) -> bool:
        """Attempt to reconnect to database with retry logic
        
        Returns:
            bool: True if reconnection successful, False otherwise
        """
        max_retries = settings.OCI_EMBEDDING_MAX_RETRIES
        retry_delay = settings.OCI_EMBEDDING_RETRY_DELAY
        
        for retry in range(max_retries):
            try:
                logger.info(f"Database reconnection attempt {retry + 1}/{max_retries}")
                self._initialize()
                
                if self.is_connected():
                    logger.info(f"Database reconnection successful after {retry + 1} attempts")
                    return True
                    
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
                    
            except Exception as e:
                logger.warning(f"Database reconnection attempt {retry + 1} failed", error=str(e))
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.error(f"Database reconnection failed after {max_retries} attempts")
        return False
    
    def is_connected(self) -> bool:
        """Check database connection status"""
        try:
            if self.connection is None:
                return False
            # Test with a simple query
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM DUAL")
                cursor.fetchone()
            return True
        except Exception:
            return False
    
    def _table_exists(self, table_name: str = "IMG_EMBEDDINGS") -> bool:
        """Check if table exists"""
        try:
            if not self.is_connected():
                return False
            
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM USER_TABLES 
                    WHERE TABLE_NAME = :table_name
                """, {'table_name': table_name.upper()})
                
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error("Table existence check failed", error=str(e), table=table_name)
            return False
    
    def _create_table(self):
        """Create IMG_EMBEDDINGS table"""
        try:
            if not self.is_connected():
                logger.error("Cannot create table - Database connection is invalid")
                return False
            
            logger.info("Creating IMG_EMBEDDINGS table...")
            
            with self.connection.cursor() as cursor:
                # Create table
                cursor.execute("""
                    CREATE TABLE IMG_EMBEDDINGS (
                        ID NUMBER GENERATED BY DEFAULT AS IDENTITY 
                            MINVALUE 1 
                            MAXVALUE 9999999999999999999999999999 
                            INCREMENT BY 1 
                            START WITH 1 
                            CACHE 20 
                            NOORDER NOCYCLE NOKEEP NOSCALE,
                        BUCKET VARCHAR2(128 BYTE) COLLATE USING_NLS_COMP,
                        OBJECT_NAME VARCHAR2(1024 BYTE) COLLATE USING_NLS_COMP,
                        CONTENT_TYPE VARCHAR2(128 BYTE) COLLATE USING_NLS_COMP,
                        FILE_SIZE NUMBER,
                        UPLOADED_AT TIMESTAMP(6) DEFAULT SYSTIMESTAMP,
                        EMBEDDING VECTOR(1536, FLOAT32)
                    ) DEFAULT COLLATION USING_NLS_COMP
                """)
                
                # Add NOT NULL constraints
                cursor.execute("ALTER TABLE IMG_EMBEDDINGS MODIFY (ID NOT NULL ENABLE)")
                cursor.execute("ALTER TABLE IMG_EMBEDDINGS MODIFY (BUCKET NOT NULL ENABLE)")
                cursor.execute("ALTER TABLE IMG_EMBEDDINGS MODIFY (OBJECT_NAME NOT NULL ENABLE)")
                cursor.execute("ALTER TABLE IMG_EMBEDDINGS MODIFY (EMBEDDING NOT NULL ENABLE)")
                
                # Add primary key
                cursor.execute("ALTER TABLE IMG_EMBEDDINGS ADD PRIMARY KEY (ID) USING INDEX ENABLE")
                
                self.connection.commit()
                logger.info("IMG_EMBEDDINGS table created successfully")
                return True
                
        except Exception as e:
            logger.error("IMG_EMBEDDINGS table creation failed", error=str(e))
            if self.connection:
                self.connection.rollback()
            return False
    
    def _ensure_table_exists(self):
        """Ensure IMG_EMBEDDINGS table exists, create if not"""
        try:
            if not self._table_exists():
                logger.info("IMG_EMBEDDINGS table does not exist, creating...")
                self._create_table()
            else:
                logger.info("IMG_EMBEDDINGS table already exists")
        except Exception as e:
            logger.error("Error ensuring table exists", error=str(e))
    
    def insert_embedding(self, bucket: str, object_name: str, content_type: str, 
                        file_size: int, embedding: np.ndarray) -> Optional[int]:
        """Insert embedding vector into database"""
        try:
            # Check connection and retry if needed
            if not self.is_connected():
                logger.warning("Database connection is invalid, attempting to reconnect")
                if not self._reconnect_with_retry():
                    logger.error("Database connection failed after retry attempts")
                    return None
            
            # Ensure table exists before inserting
            if not self._table_exists():
                logger.warning("IMG_EMBEDDINGS table not found, attempting to create...")
                if not self._create_table():
                    logger.error("Failed to create IMG_EMBEDDINGS table")
                    return None
            
            # Convert NumPy array to FLOAT32 array
            embedding_array = array.array("f", embedding.tolist())
            
            with self.connection.cursor() as cursor:
                # Insert into IMG_EMBEDDINGS table
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
                
                # Get inserted ID
                id_var = cursor.bindvars['id']
                embedding_id = id_var.getvalue()
                self.connection.commit()
                
                # If embedding_id is a list, get the first element
                if isinstance(embedding_id, list) and len(embedding_id) > 0:
                    embedding_id = embedding_id[0]
                
                logger.info("Embedding vector insertion successful", 
                           embedding_id=embedding_id, 
                           object_name=object_name)
                return embedding_id
                
        except Exception as e:
            logger.error("Embedding vector insertion failed", error=str(e))
            if self.connection:
                self.connection.rollback()
            return None
    
    def search_similar_images(self, query_embedding: np.ndarray, limit: int = 10, threshold: float = 0.7) -> Optional[list]:
        """Search for similar images based on embedding vector
        
        Args:
            query_embedding: Embedding vector for search
            limit: Maximum number of results to return
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            List of similar images (bucket, object_name, vector_distance)
        """
        try:
            # Check connection and retry if needed
            if not self.is_connected():
                logger.warning("Database connection is invalid, attempting to reconnect")
                if not self._reconnect_with_retry():
                    logger.error("Database connection failed after retry attempts")
                    return None
            
            # Ensure table exists before searching
            if not self._table_exists():
                logger.warning("IMG_EMBEDDINGS table not found, attempting to create...")
                if not self._create_table():
                    logger.error("Failed to create IMG_EMBEDDINGS table")
                    return None
            
            # Convert NumPy array to FLOAT32 array
            embedding_array = array.array("f", query_embedding.tolist())
            
            with self.connection.cursor() as cursor:
                # Execute vector similarity search query
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
                
                logger.info("Vector search successful", 
                           results_count=len(results),
                           threshold=threshold)
                return results
                
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return None


class ImageEmbedder:
    """Image embedding client using Oracle Generative AI"""
    
    def __init__(self):
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Oracle Generative AI client"""
        try:
            config = oci.config.from_file('~/.oci/config', "DEFAULT")
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                service_endpoint=f"https://inference.generativeai.{settings.OCI_REGION}.oci.oraclecloud.com",
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240)
            )
            logger.info("Oracle Generative AI client initialization successful")
        except Exception as e:
            logger.error("Oracle Generative AI client initialization failed", error=str(e))
            self.client = None
    
    def is_initialized(self) -> bool:
        """クライアント初期化状態をチェック"""
        return self.client is not None
    
    def _image_to_base64(self, image_data: io.BytesIO, content_type: str = "image/png") -> str:
        """Encode image data to base64 data URI"""
        image_data.seek(0)
        image_bytes = image_data.read()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:{content_type};base64,{base64_string}"
    
    def generate_embedding(self, image_data: io.BytesIO, content_type: str = "image/png") -> Optional[np.ndarray]:
        """Generate embedding vector from image"""
        try:
            if not self.is_initialized():
                logger.error("Oracle Generative AI クライアントが初期化されていません")
                return None
            
            logger.info(f"{settings.OCI_EMBEDDING_INPUT_TYPE=}")
            logger.info(f"{settings.OCI_COMPARTMENT_OCID=}")

            # Encode image to base64
            base64_image = self._image_to_base64(image_data, content_type)
            
            # Create embedding request
            embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
            embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                model_id=settings.OCI_COHERE_EMBED_MODEL
            )
            embed_text_detail.input_type = settings.OCI_EMBEDDING_INPUT_TYPE
            embed_text_detail.inputs = [base64_image]
            embed_text_detail.truncate = settings.OCI_EMBEDDING_TRUNCATE
            embed_text_detail.compartment_id = settings.OCI_COMPARTMENT_OCID
            
            # Retry logic
            max_retries = settings.OCI_EMBEDDING_MAX_RETRIES
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = self.client.embed_text(embed_text_detail)
                    
                    if response.data.embeddings:
                        embedding = response.data.embeddings[0]
                        embedding_array = np.array(embedding, dtype=np.float32)
                        
                        logger.info("Image embedding vector generation successful", 
                                   embedding_shape=embedding_array.shape,
                                   embedding_norm=float(np.linalg.norm(embedding_array)))
                        return embedding_array
                    else:
                        logger.error("埋め込みベクトルが空です")
                        return None
                        
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Image embedding generation error: {e}. Retrying ({retry_count}/{max_retries})...")
                    if retry_count < max_retries:
                        time.sleep(settings.OCI_EMBEDDING_RETRY_DELAY * retry_count)
                    else:
                        logger.error("Maximum retry count reached for image embedding generation")
                        return None
            
            return None
            
        except Exception as e:
            logger.error("Unexpected error during image embedding generation", error=str(e))
            return None


class TextEmbedder:
    """Text embedding client using Oracle Generative AI"""
    
    def __init__(self):
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Oracle Generative AI client"""
        try:
            config = oci.config.from_file('~/.oci/config', "DEFAULT")
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                service_endpoint=f"https://inference.generativeai.{settings.OCI_REGION}.oci.oraclecloud.com",
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240)
            )
            logger.info("Oracle Generative AI text client initialization successful")
        except Exception as e:
            logger.error("Oracle Generative AI text client initialization failed", error=str(e))
            self.client = None
    
    def is_initialized(self) -> bool:
        """Check client initialization status"""
        return self.client is not None
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding vector from text (with 3 retries)"""
        try:
            if not self.is_initialized():
                logger.error("Oracle Generative AI client is not initialized")
                return None
            
            # Create embedding request
            embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
            embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                model_id=settings.OCI_COHERE_EMBED_MODEL
            )
            embed_text_detail.input_type = "SEARCH_QUERY"  # For text search
            embed_text_detail.inputs = [text]
            embed_text_detail.truncate = settings.OCI_EMBEDDING_TRUNCATE
            embed_text_detail.compartment_id = settings.OCI_COMPARTMENT_OCID
            
            # Retry logic (3 times)
            max_retries = settings.OCI_EMBEDDING_MAX_RETRIES
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    logger.info(f"Text embedding generation attempt {retry_count + 1}/{max_retries}", text=text[:100])
                    response = self.client.embed_text(embed_text_detail)
                    
                    if response.data.embeddings:
                        embedding = response.data.embeddings[0]
                        embedding_array = np.array(embedding, dtype=np.float32)
                        
                        logger.info("Text embedding vector generation successful", 
                                   text_length=len(text),
                                   embedding_shape=embedding_array.shape,
                                   embedding_norm=float(np.linalg.norm(embedding_array)))
                        return embedding_array
                    else:
                        logger.error("Embedding vector is empty")
                        return None
                        
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Text embedding generation error: {e}. Retrying ({retry_count}/{max_retries})...")
                    if retry_count < max_retries:
                        time.sleep(settings.OCI_EMBEDDING_RETRY_DELAY * retry_count)
                    else:
                        logger.error("Maximum retry count reached for text embedding generation")
                        return None
            
            return None
            
        except Exception as e:
            logger.error("Unexpected error during text embedding generation", error=str(e), text=text[:100])
            return None


# Global clients
db_client = DatabaseClient()
image_embedder = ImageEmbedder()
text_embedder = TextEmbedder()


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    if not filename or '.' not in filename:
        logger.debug(f"Invalid filename: {filename}")
        return False

    extension = filename.rsplit('.', 1)[1].lower()
    allowed = extension in settings.ALLOWED_EXTENSIONS
    
    # Add debug information
    logger.debug(f"File extension check: filename={filename}, extension={extension}, allowed_extensions={settings.ALLOWED_EXTENSIONS}, result={allowed}")
    
    return allowed


def convert_office_to_pdf(input_path: str, output_dir: str) -> str:
    """Convert Office file to PDF using LibreOffice
    
    Args:
        input_path: Path to input Office file
        output_dir: Output directory for PDF file
        
    Returns:
        str: Path to converted PDF file
        
    Raises:
        RuntimeError: If conversion fails
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # LibreOffice conversion command
    cmd = [
        'soffice',
        '--headless',
        '--convert-to', 'pdf',
        '--outdir', str(output_dir),
        str(input_path)
    ]
    
    try:
        logger.info("Starting LibreOffice conversion", input=input_path, output_dir=output_dir)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,  # 5 minute timeout
            check=True
        )
        logger.info("LibreOffice conversion completed", input=input_path)
    except subprocess.TimeoutExpired:
        logger.error("LibreOffice conversion timeout", input=input_path)
        raise RuntimeError("LibreOffice conversion timed out")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error("LibreOffice conversion error", input=input_path, error=error_msg)
        raise RuntimeError(f"LibreOffice conversion error: {error_msg}")
    except FileNotFoundError:
        logger.error("LibreOffice not found. Please install LibreOffice.")
        raise RuntimeError("LibreOffice is not installed or not in PATH")
    
    # Check if conversion was successful
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    pdf_path = os.path.join(output_dir, base_name + ".pdf")
    
    if not os.path.exists(pdf_path):
        logger.error("PDF file not created after conversion", expected_path=pdf_path)
        raise RuntimeError("Office file could not be converted to PDF")
    
    logger.info("PDF file created successfully", pdf_path=pdf_path)
    return pdf_path


def convert_image_to_pdf_file(input_path: str, output_dir: str) -> str:
    """Convert image to PDF using img2pdf
    
    Args:
        input_path: Path to input image file
        output_dir: Output directory for PDF file
        
    Returns:
        str: Path to converted PDF file
        
    Raises:
        RuntimeError: If conversion fails
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(input_path)
    pic_name, _ = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{pic_name}.pdf")
    
    try:
        import img2pdf
        from PIL import Image
        
        logger.info("Starting image to PDF conversion", input=input_path)
        
        # Validate image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Save as temporary JPEG
            temp_path = os.path.join(output_dir, f"temp_{pic_name}.jpg")
            img.save(temp_path, 'JPEG', quality=95)
            
            # Convert to PDF
            with open(output_path, "wb") as f:
                f.write(img2pdf.convert([temp_path]))
            
            # Remove temporary file
            os.remove(temp_path)
        
        logger.info("Image to PDF conversion completed", output=output_path)
        return output_path
        
    except Exception as e:
        logger.error("Image to PDF conversion error", input=input_path, error=str(e))
        raise RuntimeError(f"Image to PDF conversion error: {str(e)}")


def create_response(success: bool, message: str, data: Optional[dict] = None, status_code: int = 200) -> Tuple[dict, int]:
    """Create unified API response format
    
    Args:
        success: Success/failure of operation
        message: Response message
        data: Additional data (optional)
        status_code: HTTP status code
        
    Returns:
        Tuple[dict, int]: (response_data, status_code)
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
    """Check if URL is a valid image URL (prevent SSRF attacks)
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        parsed = urlparse(url)
        logger.info("URL validation started", url=url, scheme=parsed.scheme, hostname=parsed.hostname)
        
        # Only HTTPS or HTTP allowed
        if parsed.scheme not in ['http', 'https']:
            error_msg = f"Unsupported protocol: {parsed.scheme}"
            logger.warning("URL validation failed", url=url, reason=error_msg)
            return False, error_msg
            
        # Reject local IP addresses and private IP addresses
        import ipaddress
        import socket
        
        hostname = parsed.hostname
        if not hostname:
            error_msg = "Hostname not found"
            logger.warning("URL validation failed", url=url, reason=error_msg)
            return False, error_msg
            
        # Reject private IPs if IP address
        try:
            ip = ipaddress.ip_address(hostname)
            logger.info("IP validation", ip=str(ip), is_private=ip.is_private, is_loopback=ip.is_loopback, is_link_local=ip.is_link_local)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                if not settings.ALLOW_PRIVATE_IPS:
                    error_msg = f"Private IP addresses are not allowed: {ip} (ALLOW_PRIVATE_IPS=False)"
                    logger.warning("URL validation failed", url=url, ip=str(ip), reason=error_msg, allow_private_ips=settings.ALLOW_PRIVATE_IPS)
                    return False, error_msg
                else:
                    logger.info("Private IP allowed", ip=str(ip), url=url, allow_private_ips=settings.ALLOW_PRIVATE_IPS)
        except ValueError:
            # For hostname, resolve DNS and check IP
            try:
                ip_str = socket.gethostbyname(hostname)
                ip = ipaddress.ip_address(ip_str)
                logger.info("IP validation after DNS resolution", hostname=hostname, resolved_ip=ip_str, is_private=ip.is_private)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    if not settings.ALLOW_PRIVATE_IPS:
                        error_msg = f"Hostname {hostname} resolved to private IP: {ip} (ALLOW_PRIVATE_IPS=False)"
                        logger.warning("URL validation failed", url=url, hostname=hostname, resolved_ip=str(ip), reason=error_msg, allow_private_ips=settings.ALLOW_PRIVATE_IPS)
                        return False, error_msg
                    else:
                        logger.info("Private IP allowed", hostname=hostname, resolved_ip=str(ip), url=url, allow_private_ips=settings.ALLOW_PRIVATE_IPS)
            except (socket.gaierror, ValueError) as e:
                error_msg = f"DNS resolution failed: {str(e)}"
                logger.warning("URL validation failed", url=url, hostname=hostname, reason=error_msg)
                return False, error_msg
                
        logger.info("URL validation successful", url=url)
        return True, ""
    except Exception as e:
        error_msg = f"Error during URL validation: {str(e)}"
        logger.error("URL validation error", url=url, error=str(e))
        return False, error_msg


def download_image_from_url(url: str, max_size: int = None) -> Tuple[io.BytesIO, str, str]:
    """Download image from URL
    
    Returns:
        Tuple[io.BytesIO, str, str]: (image_data, content_type, filename)
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
    
    # Content-Type check
    content_type = response.headers.get('content-type', '')
    if not content_type.startswith('image/'):
        raise ValueError(f"Invalid content type: {content_type}")
    
    # File size check
    content_length = response.headers.get('content-length')
    if content_length and int(content_length) > max_size:
        raise ValueError(f"File size too large: {content_length} bytes")
    
    # Download image data
    image_data = io.BytesIO()
    downloaded_size = 0
    
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                raise ValueError(f"File size too large: {downloaded_size} bytes")
            image_data.write(chunk)
    
    image_data.seek(0)
    
    # Generate filename (infer from URL or Content-Type)
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    if not filename or '.' not in filename:
        # Infer extension from Content-Type
        extension = mimetypes.guess_extension(content_type)
        if extension:
            filename = f"image{extension}"
        else:
            filename = "image.jpg"  # Default
    
    return image_data, content_type, filename


def create_app(config_name: str = None) -> Flask:
    """Application factory"""
    app = Flask(__name__)

    # Load configuration
    if config_name:
        config_class = get_config(config_name)
        app.config.from_object(config_class)

    # Basic configuration
    app.config.update(
        SECRET_KEY=settings.SECRET_KEY,
        MAX_CONTENT_LENGTH=settings.MAX_CONTENT_LENGTH,
        SESSION_COOKIE_SECURE=settings.SESSION_COOKIE_SECURE,
        SESSION_COOKIE_HTTPONLY=settings.SESSION_COOKIE_HTTPONLY,
        SESSION_COOKIE_SAMESITE=settings.SESSION_COOKIE_SAMESITE,
        PERMANENT_SESSION_LIFETIME=settings.PERMANENT_SESSION_LIFETIME,
    )
    
    # Record current configuration for debugging
    logger.info(f"Application started - allowed file extensions: {settings.ALLOWED_EXTENSIONS}")
    logger.info(f"Application started - maximum file size: {settings.MAX_CONTENT_LENGTH} bytes")
    logger.info(f"Application started - log level: {settings.LOG_LEVEL}")

    # Initialize Sentry (error monitoring)
    if settings.SENTRY_DSN and settings.SENTRY_DSN.strip():
        try:
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                integrations=[FlaskIntegration()],
                environment=settings.SENTRY_ENVIRONMENT,
                traces_sample_rate=0.1,
            )
            logger.info("Sentry initialization complete", environment=settings.SENTRY_ENVIRONMENT)
        except Exception as e:
            logger.warning("Sentry initialization failed", error=str(e))
    else:
        logger.info("Sentry DSN not configured. Error monitoring is disabled.")

    # CORS configuration
    CORS(app,
         origins=settings.CORS_ORIGINS,
         methods=settings.CORS_METHODS,
         allow_headers=['Content-Type', 'Authorization'])

    # Rate limiting configuration
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        storage_uri=settings.RATELIMIT_STORAGE_URL,
        default_limits=[settings.RATELIMIT_DEFAULT]
    )

    # Security headers configuration
    try:
        Talisman(app,
                 force_https=settings.FORCE_HTTPS,
                 csp=settings.CONTENT_SECURITY_POLICY)
        logger.info("Talisman security headers configuration complete")
    except Exception as e:
        logger.warning("Talisman configuration failed, manually setting basic security headers", error=str(e))

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
        """Health check endpoint"""
        is_connected = oci_client.is_connected()
        return jsonify({
            'status': 'running',
            'oci_connected': is_connected,
            'message': 'OK' if is_connected else 'OCI connection is not initialized',
            'endpoints': {
                'image_proxy': '/img/<bucket>/<object_name>',
                'upload': '/upload (POST)',
                'upload_document': '/api/upload/document (POST)',
                'convert_office': '/api/convert/office (POST)',
                'health': '/health',
                'test': '/test'
            }
        })

    @app.route('/test')
    def test_page():
        """Serve test page"""
        return send_from_directory('.', 'test.html')

    @app.route('/test.html')
    def test_upload_page():
        """Serve test upload page"""
        return send_from_directory('.', 'test.html')

    @app.route('/img/<bucket>/<path:obj>')
    @limiter.limit("50 per minute")
    def serve_image(bucket, obj):
        """
        Proxy endpoint to retrieve and return images from OCI Object Storage
        Supports Japanese filenames and spaces through URL encoding

        Args:
            bucket: Bucket name
            obj: Object name (URL-encoded if contains Japanese/spaces)

        Returns:
            Image data or error response
        """
        try:
            # OCI接続チェック
            if not oci_client.is_connected():
                logger.error("Image retrieval failed - OCI connection error")
                return jsonify({'error': 'OCI connection error'}), 500

            # URL decode the object name to handle Japanese and spaces
            decoded_obj = unquote(obj)
            logger.info("Image retrieval started", bucket=bucket, object=decoded_obj)

            # Get object
            response = oci_client.get_object(bucket, decoded_obj)

            # Get Content-Type (default to image)
            content_type = response.headers.get('Content-Type', 'image/jpeg')
            
            # Get original filename from metadata or use object name
            metadata = response.headers
            original_filename = metadata.get('opc-meta-original-filename', decoded_obj.split("/")[-1])

            logger.info("Image retrieval successful", object=decoded_obj, content_type=content_type)

            # Encode filename for Content-Disposition header (RFC 5987)
            # Support both ASCII and non-ASCII filenames
            try:
                filename_ascii = original_filename.encode('ascii')
                content_disposition = f'inline; filename="{original_filename}"'
            except UnicodeEncodeError:
                # Use RFC 5987 encoding for non-ASCII filenames
                filename_encoded = quote(original_filename)
                content_disposition = f"inline; filename*=UTF-8''{filename_encoded}"

            return Response(
                response.data.content,
                mimetype=content_type,
                headers={
                    'Cache-Control': 'max-age=3600',  # Cache for 1 hour
                    'Content-Disposition': content_disposition
                }
            )

        except ServiceError as e:
            if e.status == 404:
                logger.warning("Image not found", bucket=bucket, object=obj)
                return jsonify({'error': 'Image not found'}), 404
            else:
                logger.error("OCI service error", error=str(e))
                return jsonify({'error': 'OCI service error'}), 500
        except Exception as e:
            logger.error("Unexpected error during image retrieval", error=str(e))
            return jsonify({'error': 'Image retrieval failed'}), 500

    @app.route('/upload', methods=['POST'])
    @limiter.limit(settings.RATELIMIT_UPLOAD)
    def upload_image():
        """
        Endpoint to upload images to OCI Object Storage

        Request:
            - file: Image file to upload (file upload)
            - url: HTTP URL of image (download from URL)
            - bucket (optional): Destination bucket name
            - folder (optional): Destination folder
            - filename (optional): Custom filename (including extension)
            
        Note: Specify either file or url
        Note: If filename is not specified, a unique filename will be auto-generated

        Returns:
            Upload result and access URL
        """
        try:
            # Check OCI connection
            if not oci_client.is_connected():
                logger.error("Upload failed - OCI connection error")
                return jsonify({'error': 'OCI connection error'}), 500

            # Get request data (JSON or form data)
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
                # Download image from URL
                logger.info("URL image download started", url=image_url)
                
                # URL validation
                is_valid, error_message = is_valid_image_url(image_url)
                if not is_valid:
                    logger.error("URL validation failed", url=image_url, error=error_message)
                    return jsonify({'error': f'Invalid or unsafe URL: {error_message}'}), 400
                
                try:
                    # Download image
                    image_data, content_type, downloaded_filename = download_image_from_url(image_url)
                    file_size = len(image_data.getvalue())
                    logger.info(f"{downloaded_filename=}")
                    logger.info(f"{file_size=}")
                    
                   # File extension check
                    if not allowed_file(downloaded_filename):
                        return jsonify({
                            'error': f'File format not allowed. Allowed formats: {", ".join(settings.ALLOWED_EXTENSIONS)}'
                        }), 400
                    
                    logger.info("URL image download successful", 
                               url=image_url, 
                               size=file_size, 
                               content_type=content_type)
                    
                except requests.RequestException as e:
                    logger.error("Image download error", url=image_url, error=str(e))
                    return jsonify({'error': 'Image download failed'}), 400
                except ValueError as e:
                    logger.error("Image validation error", url=image_url, error=str(e))
                    return jsonify({'error': str(e)}), 400
                
                # Generate unique object name
                if filename and '.' in filename:
                    # When custom filename is specified
                    unique_filename = filename
                else:
                    # Generate default unique filename
                    file_extension = downloaded_filename.rsplit('.', 1)[1].lower()
                    unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
                
            else:
                # Traditional file upload
                if 'file' not in request.files:
                    return jsonify({'error': 'Please specify file or URL'}), 400

                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'Filename is empty'}), 400

                # File extension check
                if not allowed_file(file.filename):
                    return jsonify({
                        'error': f'File format not allowed. Allowed formats: {", ".join(settings.ALLOWED_EXTENSIONS)}'
                    }), 400

                # File size check
                file.seek(0, 2)  # Move to end of file
                file_size = file.tell()
                file.seek(0)  # Return to beginning of file

                if file_size > settings.MAX_CONTENT_LENGTH:
                    max_size_mb = settings.MAX_CONTENT_LENGTH // (1024*1024)
                    return jsonify({
                        'error': f'File size too large. Maximum size: {max_size_mb}MB'
                    }), 400

                # Generate unique object name
                file_extension = file.filename.rsplit('.', 1)[1].lower()
                if filename and '.' in filename:
                    # When custom filename is specified
                    unique_filename = filename
                else:
                    # Generate default unique filename
                    unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
                
                # Set file data and content type
                image_data = file.stream
                content_type = file.content_type or 'application/octet-stream'

            # Include in path if folder is specified
            if folder:
                object_name = f"{folder.strip('/')}/{unique_filename}"
            else:
                object_name = unique_filename

            logger.info("Upload started",
                       bucket=bucket,
                       object=object_name,
                       size=file_size,
                       source="url" if image_url else "file")

            # Prepare metadata with original filename
            # Extract original filename from object_name (format: 原ファイル名_序列号.png or custom filename)
            original_basename = unique_filename  # This is the actual filename stored
            upload_metadata = {
                'original-filename': original_basename,
                'upload-source': 'url' if image_url else 'file',
                'uploaded-at': datetime.now().isoformat()
            }
            
            # Upload to OCI Object Storage with metadata
            oci_client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=image_data,
                content_type=content_type,
                metadata=upload_metadata
            )
            time.sleep(7)

            # Generate proxy URL (URL encode the object name for Japanese/spaces support)
            encoded_object_name = quote(object_name, safe='')
            proxy_url = f"/img/{bucket}/{encoded_object_name}"

            logger.info("Upload successful", object=object_name)

            return jsonify({
                'success': True,
                'message': 'Upload completed',
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
            logger.error("HTTP request error", error=str(e))
            return jsonify({'error': 'Image download failed'}), 400
        except requests.Timeout as e:
            logger.error("HTTP timeout error", error=str(e))
            return jsonify({'error': 'Image download timed out'}), 408
        except ValueError as e:
            logger.error("Image validation error", error=str(e))
            return jsonify({'error': str(e)}), 400
        except ServiceError as e:
            logger.error("OCI service error", error=str(e))
            return jsonify({'error': 'Upload failed (OCI error)'}), 500
        except Exception as e:
            logger.error("Unexpected error during upload", error=str(e))
            return jsonify({'error': 'Upload failed'}), 500

    @app.route('/vectorize', methods=['POST'])
    @limiter.limit("10 per minute")
    def vectorize_image():
        """
        Endpoint to vectorize images
        
        Request:
            - bucket: Bucket name
            - object_name: Object name
            or
            - file: Image file (direct upload)
            - url: HTTP URL of image
            
        Returns:
            Vectorization result and embedding vector ID
        """
        try:
            # Check client connections
            if not image_embedder.is_initialized():
                logger.error("Vectorization failed - Oracle Generative AI connection error")
                return jsonify({'error': 'Oracle Generative AI connection error'}), 500
                
            # Check database connection and retry if needed
            if not db_client.is_connected():
                logger.warning("Database connection check failed in /vectorize endpoint, attempting to reconnect")
                if not db_client._reconnect_with_retry():
                    logger.error("Vectorization failed - Database connection error after retry attempts")
                    return jsonify({'error': 'Database connection error'}), 500
            
            # Ensure IMG_EMBEDDINGS table exists
            if not db_client._table_exists():
                logger.warning("IMG_EMBEDDINGS table not found in /vectorize endpoint, attempting to create...")
                if not db_client._create_table():
                    logger.error("Failed to create IMG_EMBEDDINGS table")
                    return jsonify({'error': 'Database table creation failed'}), 500

            # Get request data
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
                # Get image from OCI Object Storage
                try:
                    if not oci_client.is_connected():
                        return jsonify({'error': 'OCI connection error'}), 500
                        
                    response = oci_client.get_object(bucket, object_name)
                    image_data = io.BytesIO(response.data.content)
                    content_type = response.headers.get('Content-Type', 'image/jpeg')
                    file_size = len(response.data.content)
                    
                    logger.info("OCI image retrieval successful", bucket=bucket, object_name=object_name, size=file_size)
                    
                except ServiceError as e:
                    if e.status == 404:
                        return jsonify({'error': 'Specified image not found'}), 404
                    else:
                        logger.error("OCI image retrieval error", error=str(e))
                        return jsonify({'error': 'OCI image retrieval error'}), 500
                        
            elif image_url:
                # Download image from URL
                try:
                    # URL validation
                    is_valid, error_message = is_valid_image_url(image_url)
                    if not is_valid:
                        return jsonify({'error': f'Invalid URL: {error_message}'}), 400
                    
                    image_data, content_type, filename = download_image_from_url(image_url)
                    file_size = len(image_data.getvalue())
                    
                    # File extension check
                    if not allowed_file(filename):
                        return jsonify({
                            'error': f'File format not allowed. Allowed formats: {", ".join(settings.ALLOWED_EXTENSIONS)}'
                        }), 400
                    
                    # Set object name (use filename if available, otherwise generate temporary name)
                    if filename:
                        object_name = filename
                    else:
                        file_extension = filename.rsplit('.', 1)[1].lower() if filename else 'jpg'
                        object_name = f"temp_{uuid.uuid4().hex}.{file_extension}"
                    
                    logger.info("URL image retrieval successful", url=image_url, size=file_size)
                    
                except Exception as e:
                    logger.error("URL image retrieval error", url=image_url, error=str(e))
                    return jsonify({'error': 'Image retrieval failed'}), 400
                    
            elif 'file' in request.files:
                # Direct file upload
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'Filename is empty'}), 400
                
                # File extension check
                if not allowed_file(file.filename):
                    return jsonify({
                        'error': f'File format not allowed. Allowed formats: {", ".join(settings.ALLOWED_EXTENSIONS)}'
                    }), 400
                
                # File size check
                file.seek(0, 2)
                file_size = file.tell()
                file.seek(0)
                
                if file_size > settings.MAX_CONTENT_LENGTH:
                    max_size_mb = settings.MAX_CONTENT_LENGTH // (1024*1024)
                    return jsonify({
                        'error': f'File size too large. Maximum size: {max_size_mb}MB'
                    }), 400
                
                image_data = io.BytesIO(file.read())
                content_type = file.content_type or 'application/octet-stream'
                
                # Set object name (use filename if available, otherwise generate temporary name)
                if file.filename:
                    object_name = file.filename
                else:
                    file_extension = file.filename.rsplit('.', 1)[1].lower() if file.filename else 'jpg'
                    object_name = f"temp_{uuid.uuid4().hex}.{file_extension}"
                
                logger.info("File image retrieval successful", filename=file.filename, size=file_size)
                
            else:
                return jsonify({'error': 'Please specify bucket/object_name, url, or file'}), 400

            # Execute vectorization
            logger.info("Image vectorization started", bucket=bucket, object_name=object_name)
            
            embedding = image_embedder.generate_embedding(image_data, content_type)
            if embedding is None:
                logger.error("Image vectorization failed", object_name=object_name)
                return jsonify({'error': 'Image vectorization failed'}), 500
            
            # Save to database
            embedding_id = db_client.insert_embedding(
                bucket=bucket,
                object_name=object_name,
                content_type=content_type,
                file_size=file_size,
                embedding=embedding
            )
            
            if embedding_id is None:
                logger.error("Embedding vector save failed", object_name=object_name)
                return jsonify({'error': 'Embedding vector save failed'}), 500
            
            logger.info("Image vectorization successful", embedding_id=embedding_id, object_name=object_name)
            
            return jsonify({
                'success': True,
                'message': 'Image vectorization completed',
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
            logger.error("Unexpected error during vectorization processing", error=str(e))
            return jsonify({'error': 'Vectorization processing failed'}), 500

    @app.route('/search', methods=['POST'])
    @limiter.limit("20 per minute")
    def search_similar_images():
        """
        Endpoint to search for similar images based on text query
        
        Request:
            - query: Search query text (required)
            - limit: Maximum number of results to return (optional, default: 10)
            - threshold: Similarity threshold (optional, default: 0.7)
            
        Returns:
            List of similar images (bucket, object_name, vector_distance)
        """
        try:
            # Check database connection and retry if needed
            if not db_client.is_connected():
                logger.warning("Database connection check failed in /search endpoint, attempting to reconnect")
                if not db_client._reconnect_with_retry():
                    logger.error("Search failed - Database connection error after retry attempts")
                    return jsonify({'error': 'Database connection error'}), 500
            
            # Get request data
            if request.is_json:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'JSON data required'}), 400
                query = data.get('query')
                limit = data.get('limit', 10)
                threshold = data.get('threshold', 0.7)
            else:
                query = request.form.get('query')
                limit = int(request.form.get('limit', 10))
                threshold = float(request.form.get('threshold', 0.7))
            
            # Validate query parameters
            if not query or not query.strip():
                return jsonify({'error': 'Query text required'}), 400
            
            if not isinstance(limit, int) or limit <= 0 or limit > 100:
                return jsonify({'error': 'limit must be an integer between 1 and 100'}), 400
            
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                return jsonify({'error': 'threshold must be a number between 0.0 and 1.0'}), 400
            
            logger.info("Image search started", query=query, limit=limit, threshold=threshold)
            
            # Generate embedding vector from text (with 3 retries)
            query_embedding = text_embedder.generate_embedding(query.strip())
            
            if query_embedding is None:
                logger.error("Text embedding generation failed", query=query)
                return jsonify({'error': 'Text embedding generation failed'}), 500
            
            # Execute similar image search
            results = db_client.search_similar_images(
                query_embedding=query_embedding,
                limit=limit,
                threshold=threshold
            )
            
            if results is None:
                logger.error("Image search failed", query=query)
                return jsonify({'error': 'Image search failed'}), 500
            
            # Add proxy URL
            for result in results:
                result['proxy_url'] = f"/img/{result['bucket']}/{result['object_name']}"
            
            logger.info("Image search successful", 
                       query=query, 
                       results_count=len(results),
                       threshold=threshold)
            
            return jsonify({
                'success': True,
                'message': f'Search completed. Found {len(results)} similar images.',
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
            logger.error("Parameter error", query=query, error=str(e))
            return jsonify({'error': f'Parameter error: {str(e)}'}), 400
        except Exception as e:
            logger.error("Unexpected error during search processing", query=query, error=str(e))
            return jsonify({'error': 'Search processing failed'}), 500

    @app.route('/api/upload/document', methods=['POST'])
    @limiter.limit(settings.RATELIMIT_UPLOAD)
    def upload_document():
        """
        Endpoint to upload PPT/PPTX/PDF files to OCI Object Storage
        
        Request:
            - file: PPT/PPTX/PDF file to upload (required)
            - bucket (optional): Destination bucket name
            - folder (optional): Destination folder
            - filename (optional): Custom filename (including extension)
            
        Returns:
            Upload result and access URL
        """
        logger.info("Received PPT/PPTX/PDF file upload request")
        
        try:
            # Check OCI connection
            if not oci_client.is_connected():
                logger.error("Upload failed - OCI connection error")
                response, status_code = create_response(
                    success=False,
                    message='OCI connection error',
                    status_code=500
                )
                return jsonify(response), status_code
            
            # Check if file is included in request
            if 'file' not in request.files:
                logger.warning("File not included in request")
                response, status_code = create_response(
                    success=False,
                    message='File not specified',
                    status_code=400
                )
                return jsonify(response), status_code
            
            file = request.files['file']
            
            # Check if file is selected
            if file.filename == '':
                logger.warning("File not selected")
                response, status_code = create_response(
                    success=False,
                    message='File not selected',
                    status_code=400
                )
                return jsonify(response), status_code
            
            # Check file extension
            if not allowed_file(file.filename):
                logger.warning(f"Unsupported file format: {file.filename}")
                response, status_code = create_response(
                    success=False,
                    message=f'Unsupported file format. Allowed formats: {", ".join(settings.ALLOWED_EXTENSIONS)}',
                    status_code=400
                )
                return jsonify(response), status_code
            
            # PPT/PPTX/PDF exclusive check
            file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
            if file_extension not in ['ppt', 'pptx', 'pdf']:
                logger.warning(f"Non-PPT/PPTX/PDF file format: {file.filename}")
                response, status_code = create_response(
                    success=False,
                    message='Only PPT/PPTX/PDF files can be uploaded',
                    status_code=400
                )
                return jsonify(response), status_code
            
            # File size check
            file.seek(0, 2)  # Move to end of file
            file_size = file.tell()
            file.seek(0)  # Return to beginning of file
            
            if file_size > settings.MAX_CONTENT_LENGTH:
                max_size_mb = settings.MAX_CONTENT_LENGTH // (1024*1024)
                logger.warning(f"File size too large: {file_size} bytes")
                response, status_code = create_response(
                    success=False,
                    message=f'File size too large. Maximum size: {max_size_mb}MB',
                    status_code=400
                )
                return jsonify(response), status_code
            
            # Get request parameters
            bucket = request.form.get('bucket', settings.OCI_BUCKET)
            folder = request.form.get('folder', '')
            custom_filename = request.form.get('filename')
            
            # Generate unique object name
            if custom_filename and '.' in custom_filename:
                # When custom filename is specified
                unique_filename = custom_filename
            else:
                # Generate default unique filename
                unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
            
            # Include in path if folder is specified
            if folder:
                object_name = f"{folder.strip('/')}/{unique_filename}"
            else:
                object_name = unique_filename
            
            # Set file data and content type
            file_data = file.stream
            content_type = file.content_type or 'application/vnd.ms-powerpoint'
            
            # Set more appropriate MIME type based on extension
            if file_extension == 'pptx':
                content_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            elif file_extension == 'ppt':
                content_type = 'application/vnd.ms-powerpoint'
            elif file_extension == 'pdf':
                content_type = 'application/pdf'
            
            # Prepare upload metadata
            upload_metadata = {
                'original-filename': unique_filename,
                'upload-source': 'document',
                'uploaded-at': datetime.now().isoformat()
            }
            
            logger.info("PPT/PPTX/PDF upload started",
                       bucket=bucket,
                       object=object_name,
                       size=file_size,
                       content_type=content_type)
            
            # Upload to OCI Object Storage with metadata
            oci_client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=file_data,
                content_type=content_type,
                metadata=upload_metadata
            )
            
            # Generate proxy URL (URL encode the object name for Japanese/spaces support)
            encoded_object_name = quote(object_name, safe='')
            proxy_url = f"/img/{bucket}/{encoded_object_name}"
            
            logger.info("PPT/PPTX/PDF upload successful", object=object_name)
            
            response, status_code = create_response(
                success=True,
                message='PPT/PPTX/PDF file upload completed',
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
            logger.error("OCI service error", error=str(e))
            response, status_code = create_response(
                success=False,
                message='Upload failed (OCI error)',
                status_code=500
            )
            return jsonify(response), status_code
        except Exception as e:
            logger.error("Unexpected error during PPT/PPTX/PDF upload", error=str(e))
            response, status_code = create_response(
                success=False,
                message='Upload failed',
                status_code=500
            )
            return jsonify(response), status_code

    @app.route('/api/convert/office', methods=['POST'])
    @limiter.limit("10 per minute")
    def convert_office_endpoint():
        """
        Endpoint to convert Office files to PDF
        
        Request:
            - file: Office file to convert (required)
            
        Returns:
            PDF file (binary data)
        """
        logger.info("Received Office file conversion request")
        
        # Check if file is included in request
        if 'file' not in request.files:
            logger.warning("File not included in request")
            response, status_code = create_response(
                success=False,
                message='File not specified',
                status_code=400
            )
            return jsonify(response), status_code
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            logger.warning("File not selected")
            response, status_code = create_response(
                success=False,
                message='File not selected',
                status_code=400
            )
            return jsonify(response), status_code
        
        # Check file extension
        if not allowed_file(file.filename):
            logger.warning(f"Unsupported file format: {file.filename}")
            response, status_code = create_response(
                success=False,
                message=f'Unsupported file format. Allowed formats: {", ".join(settings.ALLOWED_EXTENSIONS)}',
                status_code=400
            )
            return jsonify(response), status_code
        
        # Office file exclusive check
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        allowed_office_extensions = ['docx', 'pptx', 'xlsx', 'doc', 'ppt', 'xls']
        
        if file_extension not in allowed_office_extensions:
            logger.warning(f"Non-Office file format: {file.filename}")
            response, status_code = create_response(
                success=False,
                message=f'Unsupported file format. Allowed formats: {", ".join(allowed_office_extensions)}',
                status_code=400
            )
            return jsonify(response), status_code
        
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        input_path = None
        pdf_path = None
        
        try:
            # Save uploaded file to temporary location
            input_filename = f"{uuid.uuid4().hex}_{file.filename}"
            input_path = os.path.join(temp_dir, input_filename)
            file.save(input_path)
            
            logger.info("File saved to temporary location", path=input_path)
            
            # Convert to PDF
            pdf_path = convert_office_to_pdf(input_path, temp_dir)
            
            # Get original filename without extension
            original_name = os.path.splitext(file.filename)[0]
            download_name = f"{original_name}.pdf"
            
            logger.info("Sending PDF file", pdf_path=pdf_path, download_name=download_name)
            
            # Return PDF file
            return send_from_directory(
                directory=os.path.dirname(pdf_path),
                path=os.path.basename(pdf_path),
                as_attachment=True,
                download_name=download_name,
                mimetype='application/pdf'
            )
            
        except FileNotFoundError as e:
            logger.error("File not found error", error=str(e))
            response, status_code = create_response(
                success=False,
                message=str(e),
                status_code=404
            )
            return jsonify(response), status_code
        except RuntimeError as e:
            logger.error("PDF conversion error", error=str(e))
            response, status_code = create_response(
                success=False,
                message=f'PDF conversion error: {str(e)}',
                status_code=500
            )
            return jsonify(response), status_code
        except Exception as e:
            logger.error("Unexpected error during conversion", error=str(e))
            response, status_code = create_response(
                success=False,
                message='Unexpected error occurred',
                status_code=500
            )
            return jsonify(response), status_code
        finally:
            # Clean up temporary files
            try:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info("Temporary directory cleaned up", temp_dir=temp_dir)
            except Exception as e:
                logger.warning("Failed to clean up temporary directory", temp_dir=temp_dir, error=str(e))

    @app.route('/health')
    def health_check():
        """Health check endpoint with auto-recovery"""
        max_retries = settings.OCI_EMBEDDING_MAX_RETRIES
        retry_delay = settings.OCI_EMBEDDING_RETRY_DELAY
        
        # Initialize status tracking
        status_details = {
            'oci_connection': {'status': 'checking', 'message': '', 'retries': 0},
            'database_connection': {'status': 'checking', 'message': '', 'retries': 0},
            'embedder_status': {'status': 'checking', 'message': '', 'retries': 0}
        }
        
        # Check and retry OCI connection
        is_connected = oci_client.is_connected()
        if not is_connected:
            logger.warning("OCI connection check failed, attempting to reconnect")
            for retry in range(max_retries):
                status_details['oci_connection']['retries'] = retry + 1
                logger.info(f"OCI reconnection attempt {retry + 1}/{max_retries}")
                oci_client._initialize()
                time.sleep(retry_delay)
                is_connected = oci_client.is_connected()
                if is_connected:
                    logger.info(f"OCI reconnection successful after {retry + 1} attempts")
                    break
        
        status_details['oci_connection']['status'] = 'OK' if is_connected else 'FAILED'
        status_details['oci_connection']['message'] = 'Connected' if is_connected else f'Connection failed after {max_retries} retries'
        
        # Check and retry database connection
        db_connected = db_client.is_connected()
        if not db_connected:
            logger.warning("Database connection check failed, attempting to reconnect")
            for retry in range(max_retries):
                status_details['database_connection']['retries'] = retry + 1
                logger.info(f"Database reconnection attempt {retry + 1}/{max_retries}")
                db_client._initialize()
                time.sleep(retry_delay)
                db_connected = db_client.is_connected()
                if db_connected:
                    logger.info(f"Database reconnection successful after {retry + 1} attempts")
                    break
        
        status_details['database_connection']['status'] = 'OK' if db_connected else 'FAILED'
        status_details['database_connection']['message'] = 'Connected' if db_connected else f'Connection failed after {max_retries} retries'
        
        # Check and retry embedder initialization
        embedder_initialized = image_embedder.is_initialized()
        if not embedder_initialized:
            logger.warning("Embedder initialization check failed, attempting to reinitialize")
            for retry in range(max_retries):
                status_details['embedder_status']['retries'] = retry + 1
                logger.info(f"Embedder reinitialization attempt {retry + 1}/{max_retries}")
                image_embedder._initialize()
                time.sleep(retry_delay)
                embedder_initialized = image_embedder.is_initialized()
                if embedder_initialized:
                    logger.info(f"Embedder reinitialization successful after {retry + 1} attempts")
                    break
        
        status_details['embedder_status']['status'] = 'OK' if embedder_initialized else 'FAILED'
        status_details['embedder_status']['message'] = 'Initialized' if embedder_initialized else f'Initialization failed after {max_retries} retries'
        
        # Determine overall health status
        all_healthy = is_connected and db_connected and embedder_initialized
        overall_status = 'healthy' if all_healthy else 'unhealthy'
        status_code = 200 if all_healthy else 503
        
        logger.info(f"Health check completed: {overall_status}", 
                   oci_connected=is_connected,
                   db_connected=db_connected,
                   embedder_initialized=embedder_initialized)
        
        return jsonify({
            'status': overall_status,
            'components': status_details,
            'timestamp': datetime.now().isoformat()
        }), status_code

    # Error handlers
    @app.errorhandler(413)
    def too_large(e):
        """File size limit error handler"""
        logger.warning("File size limit error")
        return jsonify({'error': 'File size too large'}), 413

    @app.errorhandler(404)
    def not_found(e):
        """404 error handler"""
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(500)
    def internal_error(e):
        """500 error handler"""
        logger.error("Internal server error", error=str(e))
        return jsonify({'error': 'Internal server error occurred'}), 500

    @app.errorhandler(ServiceError)
    def handle_oci_error(e):
        """OCI service error handler"""
        logger.error("OCI service error",
                    status=e.status,
                    code=e.code,
                    message=e.message)
        return jsonify({
            'error': 'OCI service error occurred',
            'details': e.message if settings.DEBUG else None
        }), 500

    return app


def create_production_app() -> Flask:
    """Create application for production environment"""
    return create_app('production')


def create_development_app() -> Flask:
    """Create application for development environment"""
    return create_app('development')


def create_testing_app() -> Flask:
    """Create application for testing environment"""
    return create_app('testing')


# Default application (for development)
app = create_development_app()


if __name__ == '__main__':
    # Run in development environment
    env = os.getenv('FLASK_ENV', 'development')
    port = int(os.getenv('PORT', settings.PORT))
    debug = settings.DEBUG

    logger.info("Application started",
               environment=env,
               port=port,
               debug=debug)

    app.run(host=settings.HOST, port=port, debug=debug)
