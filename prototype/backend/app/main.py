"""
FastAPI application entry point for the CDSS Phase 09 Prototype.
Loads model artifacts at startup and mounts all routers.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.services.model_registry import registry
from app.api import catalogue, inference, frontend as frontend_module

# ─────────────────────────── Logging ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────── Lifespan ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting CDSS Prototype API …")
    try:
        registry.load(settings.ARTIFACTS_DIR)
        logger.info("All artifacts loaded. API is ready.")
    except Exception as exc:
        logger.error("Failed to load model artifacts: %s", exc)
        logger.warning("API will start in DEGRADED mode (no inference available).")
    yield
    logger.info("Shutting down CDSS Prototype API.")


# ─────────────────────────── App ───────────────────────────────────────────
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "**Clinical Decision Support System (CDSS) Research Prototype — Phase 09**\n\n"
        "Accepts a binary symptom profile and returns a ranked differential diagnosis "
        "from 41 disease classes, calibrated probabilities (Phase 07), and TreeSHAP "
        "local explanations (Phase 08).\n\n"
        "> ⚠️ **Research use only.** Not validated for clinical or emergency use."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────── CORS ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# ─────────────────────────── Routers ───────────────────────────────────────
app.include_router(catalogue.router)
app.include_router(inference.router)

# Mount frontend (serves index.html at /)
frontend_module.mount_frontend(app)
