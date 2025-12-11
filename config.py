import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    
    # Flask settings
    JSON_SORT_KEYS = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Perplexity API settings
    PERPLEXITY_TIMEOUT = 30  # seconds
    
    # YOLO model settings
    YOLO_MODEL = 'yolo11l.pt'  # Small YOLOv8 model
    
    # Demo mode for testing
    DEMO_MODE = os.environ.get('DEMO_MODE') == 'true'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DEMO_MODE = True


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
