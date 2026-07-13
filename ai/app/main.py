import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import predict, explain
from app.models.schemas import HealthResponse
from app.services.gnn_service import gnn_service

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events."""
    logger.info("🚀 MolXAI AI Service starting up...")
    logger.info(f"   Model version: {settings.AI_MODEL_VERSION}")
    logger.info(f"   Model loaded:  {gnn_service.model_loaded}")
    yield
    logger.info("🛑 MolXAI AI Service shutting down...")


app = FastAPI(
    title="MolXAI AI Service",
    description=(
        "FastAPI microservice for the EQ-KA-GCN molecular toxicity prediction model. "
        "Provides SMILES-based toxicity prediction and GNN attention explainability. "
        "Currently running in PLACEHOLDER mode — model integration pending."
    ),
    version=settings.AI_MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────
app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(explain.router, prefix="/api", tags=["Explainability"])


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="online",
        version=settings.AI_MODEL_VERSION,
        model_loaded=gnn_service.model_loaded,
        message=(
            "AI service is running. Model integration pending."
            if not gnn_service.model_loaded
            else "AI service is fully operational."
        ),
    )
