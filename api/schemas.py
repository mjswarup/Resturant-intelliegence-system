"""
PHASE 22: Pydantic Schemas
Restaurant Intelligence System - API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ============================================
# RATING PREDICTION
# ============================================

class RatingPredictionRequest(BaseModel):
    City: Optional[str] = None
    Locality: Optional[str] = None
    Currency: Optional[str] = None
    Primary_Cuisine: Optional[str] = Field(None, alias="Primary_Cuisine")
    Country_Code: Optional[int] = Field(1, alias="Country Code")
    Longitude: Optional[float] = None
    Latitude: Optional[float] = None
    Average_Cost_for_two: Optional[float] = Field(None, alias="Average Cost for two")
    Price_range: Optional[int] = Field(None, alias="Price range", ge=1, le=4)
    Has_Table_booking: Optional[int] = Field(0, alias="Has Table booking", ge=0, le=1)
    Has_Online_delivery: Optional[int] = Field(0, alias="Has Online delivery", ge=0, le=1)
    Is_delivering_now: Optional[int] = Field(0, alias="Is delivering now", ge=0, le=1)
    Cuisine_Count: Optional[int] = None
    Name_Length: Optional[int] = None
    Address_Length: Optional[int] = None
    City_Frequency: Optional[float] = None
    Locality_Frequency: Optional[float] = None

    class Config:
        populate_by_name = True
        extra = "allow"  # allow any additional feature fields raw dataset had


class RatingPredictionResponse(BaseModel):
    predicted_rating: float
    model_version: str
    model_notes: str
    warnings: List[str]


# ============================================
# RECOMMENDATION
# ============================================

class RecommendationRequest(BaseModel):
    city: Optional[str] = None
    cuisine: Optional[str] = None
    min_rating: float = Field(0.0, ge=0.0, le=5.0)
    max_cost: Optional[float] = Field(None, ge=0)
    price_range: Optional[int] = Field(None, ge=1, le=4)
    online_delivery: Optional[bool] = None
    table_booking: Optional[bool] = None
    top_n: int = Field(10, ge=1, le=100)


class RestaurantResult(BaseModel):
    rank: int
    restaurant: str
    cuisine: str
    rating: float
    votes: int
    price_range: int
    average_cost_for_two: float
    currency: str
    city: str
    locality: str
    online_delivery: bool
    table_booking: bool
    reason: str


class RecommendationResponse(BaseModel):
    results: List[RestaurantResult]
    count: int
    filters_applied: Dict[str, Any]


# ============================================
# CUISINE CLASSIFICATION
# ============================================

class CuisinePredictionRequest(BaseModel):
    City: Optional[str] = None
    Country_Code: Optional[int] = Field(1, alias="Country Code")
    Average_Cost_for_two: Optional[float] = Field(None, alias="Average Cost for two")
    Price_range: Optional[int] = Field(None, alias="Price range", ge=1, le=4)
    Has_Table_booking: Optional[int] = Field(0, alias="Has Table booking", ge=0, le=1)
    Has_Online_delivery: Optional[int] = Field(0, alias="Has Online delivery", ge=0, le=1)
    Aggregate_rating: Optional[float] = Field(None, alias="Aggregate rating", ge=0, le=5)
    Votes: Optional[int] = Field(0, ge=0)
    Cuisine_Count: Optional[int] = None
    Name_Length: Optional[int] = None
    City_Frequency: Optional[float] = None
    Locality_Frequency: Optional[float] = None

    class Config:
        populate_by_name = True
        extra = "allow"


class CuisinePrediction(BaseModel):
    cuisine: str
    probability: float


class CuisinePredictionResponse(BaseModel):
    predicted_cuisine: str
    confidence: float
    top_3_predictions: List[CuisinePrediction]
    model_version: str
    num_classes: int
    warnings: List[str]


# ============================================
# LOCATION INTELLIGENCE
# ============================================

class LocationAnalysisResponse(BaseModel):
    city: str
    found: bool
    restaurant_count: Optional[int] = None
    avg_rating: Optional[float] = None
    avg_cost_for_two: Optional[float] = None
    avg_votes: Optional[float] = None
    online_delivery_pct: Optional[float] = None
    table_booking_pct: Optional[float] = None
    top_cuisines: Optional[Dict[str, int]] = None
    locality: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


# ============================================
# HEALTH & MODEL INFO
# ============================================

class HealthResponse(BaseModel):
    status: str
    message: str


class ModelInfoResponse(BaseModel):
    system_name: str
    last_updated: str
    models: Dict[str, Any]
    future_extension_points: List[str]


class ErrorResponse(BaseModel):
    error: str
    detail: str