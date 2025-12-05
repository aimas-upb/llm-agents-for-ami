"""Main FastAPI application for RD4 Signifier System.

This module creates and configures the FastAPI application
with all routes and middleware.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import matching, signifiers, validation, simple_signifiers
from src.config import get_settings, setup_logging

settings = get_settings()
setup_logging(settings)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Args:
        app: FastAPI application instance

    Yields:
        None
    """
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info(f"Storage directory: {settings.storage_dir}")
    logger.info(f"Enabled modules: {settings.enabled_modules}")
    logger.info(f"Authoring validation: {settings.enable_authoring_validation}")
    yield
    logger.info(f"Shutting down {settings.app_name}")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="RD4 Signifier System - Phase 3: Intent Matcher",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simple_signifiers.router)
# app.include_router(signifiers.router)
# app.include_router(validation.router)
# app.include_router(matching.router)


@app.get("/")
async def root():
    """Root endpoint.

    Returns:
        API information
    """
    return {
        "name": settings.app_name,
        "version": settings.version,
        "phase": "Phase 3 - Intent Matcher",
        "status": "operational",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "version": settings.version,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
