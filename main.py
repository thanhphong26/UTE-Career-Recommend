import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import AppConfig
from models.database import init_db
from api import recommendations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=AppConfig.APP_NAME,
    version=AppConfig.APP_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(recommendations.router, prefix="/api")

# Database initialization on startup
@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")

# Health check endpoint with more details
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": AppConfig.APP_VERSION,
        "database": "connected"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "status": "error",
        "message": "An unexpected error occurred"
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host=AppConfig.HOST, 
        port=AppConfig.PORT, 
        reload=True
    )