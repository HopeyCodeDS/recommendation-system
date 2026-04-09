"""
FastAPI backend for the Recommender System portfolio app.
"""

import os
import json
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .model_loader import store
from .routers import books, recommendations, metrics
from .schemas import HealthResponse


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy scalars that Pydantic may miss."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and data on startup."""
    store.load()
    yield


app = FastAPI(
    title="Recommender System API",
    description="Book recommender system with CF, Content-Based, and Hybrid algorithms.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = (
    [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    if allowed_origins_env
    else ["*"]  # open during development
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(books.router, tags=["Books"])
app.include_router(recommendations.router, tags=["Recommendations"])
app.include_router(metrics.router, tags=["Metrics"])


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        models_loaded=store.models_loaded_status(),
        dataset_stats=store.dataset_stats(),
    )
