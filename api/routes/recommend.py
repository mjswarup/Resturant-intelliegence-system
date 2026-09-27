"""PHASE 22: Recommendation endpoint"""

from fastapi import APIRouter, HTTPException
from api.schemas import RecommendationRequest, RecommendationResponse
from src.inference.predict import recommend_restaurants, ValidationError

router = APIRouter()


@router.post("/recommend", response_model=RecommendationResponse, tags=["Recommendation"])
def recommend_endpoint(request: RecommendationRequest):
    try:
        result = recommend_restaurants(
            city=request.city,
            cuisine=request.cuisine,
            min_rating=request.min_rating,
            max_cost=request.max_cost,
            price_range=request.price_range,
            online_delivery=request.online_delivery,
            table_booking=request.table_booking,
            top_n=request.top_n
        )
        return RecommendationResponse(**result)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")