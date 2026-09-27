"""PHASE 22: Cuisine classification endpoint"""

from fastapi import APIRouter, HTTPException
from api.schemas import CuisinePredictionRequest, CuisinePredictionResponse
from src.inference.predict import predict_cuisine, ValidationError

router = APIRouter()


@router.post("/predict/cuisine", response_model=CuisinePredictionResponse, tags=["Cuisine Classification"])
def predict_cuisine_endpoint(request: CuisinePredictionRequest):
    try:
        input_dict = request.dict(by_alias=True, exclude_none=False)
        result = predict_cuisine(input_dict)
        return CuisinePredictionResponse(**result)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")