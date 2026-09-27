"""PHASE 22: Health check endpoint"""

from fastapi import APIRouter
from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(status="ok", message="Restaurant Intelligence System API is running")