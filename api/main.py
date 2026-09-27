"""
PHASE 22: FastAPI Main Application
Restaurant Intelligence System
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import health, rating, recommend, cuisine, location
from api.schemas import ModelInfoResponse
from src.config.config import MODELS_DIR

app = FastAPI(
    title="Restaurant Intelligence System API",
    description="Rating prediction, recommendation, cuisine classification, and location intelligence.",
    version="1.0.0"
)

# CORS - open for demo/dev purposes only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(rating.router)
app.include_router(recommend.router)
app.include_router(cuisine.router)
app.include_router(location.router)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    with open(MODELS_DIR / "model_registry.json", "r") as f:
        registry = json.load(f)
    return ModelInfoResponse(**registry)


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Restaurant Intelligence System API",
        "docs": "/docs",
        "endpoints": [
            "/health", "/model-info", "/predict/rating",
            "/recommend", "/predict/cuisine", "/location/analysis"
        ]
    }