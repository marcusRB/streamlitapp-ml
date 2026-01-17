"""
FastAPI Main Application
Entry point for the CKD Detection API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.config import settings
from backend.utils.logger import get_logger
from backend.api.routes import health, models, predictions, mlflow_routes

# Setup logger
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.api.TITLE,
    description=settings.api.DESCRIPTION,
    version=settings.api.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(predictions.router, prefix="/predict", tags=["Predictions"])
app.include_router(mlflow_routes.router, prefix="/mlflow", tags=["MLflow"])


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting CKD Detection API...")
    logger.info(f"API Version: {settings.api.VERSION}")
    logger.info(f"Models Directory: {settings.paths.MODELS_DIR}")
    logger.info(f"MLflow Enabled: {settings.mlflow.ENABLED}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down CKD Detection API...")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "CKD Detection API",
        "version": settings.api.VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api.HOST,
        port=settings.api.PORT,
        reload=settings.api.RELOAD,
        log_level=settings.api.LOG_LEVEL.lower()
    )