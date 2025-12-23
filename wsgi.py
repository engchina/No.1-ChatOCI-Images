# -*- coding: utf-8 -*-
"""
WSGI Entry Point
For application startup in production environment
"""

import os
from app import create_app, create_production_app, create_development_app

# Create application based on environment
env = os.getenv('FLASK_ENV', 'production')

if env == 'production':
    application = create_production_app()
elif env == 'development':
    application = create_development_app()
else:
    application = create_app(env)

# Alias for Gunicorn
app = application

if __name__ == "__main__":
    # Start with development server
    port = int(os.getenv('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=(env == 'development'))
