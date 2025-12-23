# -*- coding: utf-8 -*-
"""
OCI Image Proxy Application Configuration Example
Copy this file to config.py and modify the settings
"""

import os

# Flask configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# OCI configuration
OCI_CONFIG_FILE = os.getenv('OCI_CONFIG_FILE', '~/.oci/config')
OCI_PROFILE = os.getenv('OCI_PROFILE', 'DEFAULT')
OCI_BUCKET = os.getenv('OCI_BUCKET', 'chatbot-images')

# Upload configuration
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
