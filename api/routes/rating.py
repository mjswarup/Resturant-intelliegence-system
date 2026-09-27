"""PHASE 22: Rating prediction endpoint"""

from fastapi import APIRouter, HTTPException
from api.schemas import RatingPredictionRequest, RatingPredictionResponse
from src.inference.predict import predict_rating, ValidationError

router = APIRouter()


@router.post("/predict/rating", response_model=RatingPredictionResponse, tags=["Rating Prediction"])
def predict_rating_endpoint(request: RatingPredictionRequest):
    try:
        input_dict = request.dict(by_alias=True, exclude_none=False)
        result = predict_rating(input_dict)
        return RatingPredictionResponse(**result)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")