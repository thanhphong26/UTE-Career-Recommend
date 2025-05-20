import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class DatabaseConfig:
    SQLALCHEMY_DATABASE_URL = os.getenv(
        'DATABASE_URL', 
        'mysql+pymysql://root:NnbUcPSBYNPJZCPuOdClwftCHMseuqUs@turntable.proxy.rlwy.net:37130/ute_career_bridge'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    POOL_SIZE = int(os.getenv('DATABASE_POOL_SIZE', 10))
    POOL_TIMEOUT = int(os.getenv('DATABASE_POOL_TIMEOUT', 30))

class AppConfig:
    APP_NAME = os.getenv('APP_NAME', "Job Recommendation Microservice")
    APP_VERSION = os.getenv('APP_VERSION', "1.0.0")
    HOST = os.getenv('APP_HOST', "0.0.0.0")
    PORT = int(os.getenv('APP_PORT', 8000))
    
    # Recommendation settings
    DEFAULT_TOP_N = int(os.getenv('DEFAULT_TOP_N', 5))
    MAX_TOP_N = int(os.getenv('MAX_TOP_N', 10))

    # Upload directories
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    
    # Create directories if they don't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)

class SecurityConfig:
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
    CORS_ALLOW_METHODS = os.getenv('CORS_ALLOW_METHODS', '*').split(',')

class LoggingConfig:
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')
    LOG_ENCODING = 'utf-8'  # Set UTF-8 encoding for log files

# Combined configuration class for easy import
class Config(DatabaseConfig, AppConfig, SecurityConfig, LoggingConfig):
    pass