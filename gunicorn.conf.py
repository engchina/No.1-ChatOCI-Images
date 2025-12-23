# -*- coding: utf-8 -*-
"""
Gunicorn Configuration File
Optimized WSGI server configuration for production environment
"""

import os
import multiprocessing

# Server configuration
bind = f"0.0.0.0:{os.getenv('PORT', 8000)}"
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 120
keepalive = 2

# Logging configuration
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process configuration
preload_app = True
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# Security configuration
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance configuration
worker_tmp_dir = "/dev/shm"  # Memory-based temp directory (if available)

# Development environment configuration
if os.getenv('FLASK_ENV') == 'development':
    reload = True
    workers = 1
    loglevel = 'debug'

def when_ready(server):
    """Callback when server starts"""
    server.log.info("Server started")

def worker_int(worker):
    """Callback when worker process is interrupted"""
    worker.log.info("Worker process interrupted: %s", worker.pid)

def pre_fork(server, worker):
    """Callback before worker process creation"""
    server.log.info("Creating worker process: %s", worker.pid)

def post_fork(server, worker):
    """Callback after worker process creation"""
    server.log.info("Worker process created: %s", worker.pid)

def pre_exec(server):
    """Callback before server execution"""
    server.log.info("Executing server...")

def on_exit(server):
    """Callback when server exits"""
    server.log.info("Server exited")
