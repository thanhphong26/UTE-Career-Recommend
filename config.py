import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig:
    SQLALCHEMY_DATABASE_URL = os.getenv(
        'DATABASE_URL', 
        'mysql+pymysql://username:password@localhost/job_recommendation_db'
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

class SecurityConfig:
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
    CORS_ALLOW_METHODS = os.getenv('CORS_ALLOW_METHODS', '*').split(',')