"""PHASE 22: Location intelligence endpoint"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from api.schemas import LocationAnalysisResponse
from src.inference.predict import analyze_location, ValidationError

router = APIRouter()


@router.get("/location/analysis", response_model=LocationAnalysisResponse, tags=["Location Intelligence"])
def location_analysis_endpoint(
    city: str = Query(..., description="City name, e.g. 'New Delhi'"),
    locality: Optional[str] = Query(None, description="Optional locality within the city")
):
    try:
        result = analyze_location(city=city, locality=locality)
        return LocationAnalysisResponse(**result)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")