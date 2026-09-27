"""
PHASE 21: INFERENCE ENGINE
Restaurant Intelligence System

Unified, validated, structured inference functions:
    - predict_rating()
    - recommend_restaurants()
    - predict_cuisine()
    - analyze_location()
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
import json
from typing import Optional, List, Dict, Any

from src.config.config import MODELS_DIR, PROCESSED_DATA_DIR


# ============================================
# LAZY-LOADED MODEL CACHE
# ============================================
# Models load once, on first use, and stay in memory.
# This avoids the 486MB cosine_similarity matrix being loaded
# unless a similarity-based lookup is actually requested.

_cache = {
    "rating_bundle": None,
    "cuisine_bundle": None,
    "recommendation_data": None,   # lightweight: dataframe + tfidf only
    "location_data": None,
    "registry": None
}


def _load_registry() -> dict:
    if _cache["registry"] is None:
        with open(MODELS_DIR / "model_registry.json", "r") as f:
            _cache["registry"] = json.load(f)
    return _cache["registry"]


def _load_rating_bundle() -> dict:
    if _cache["rating_bundle"] is None:
        registry = _load_registry()
        file = registry["models"]["rating_prediction"]["file"]
        _cache["rating_bundle"] = joblib.load(MODELS_DIR / file)
    return _cache["rating_bundle"]


def _load_cuisine_bundle() -> dict:
    if _cache["cuisine_bundle"] is None:
        registry = _load_registry()
        file = registry["models"]["cuisine_classification"]["file"]
        _cache["cuisine_bundle"] = joblib.load(MODELS_DIR / file)
    return _cache["cuisine_bundle"]


def _load_recommendation_data() -> dict:
    """
    Loads ONLY the restaurant dataframe + TF-IDF vectorizer.
    Deliberately does NOT load the 486MB precomputed cosine_similarity
    matrix from recommendation_bundle_v1.0.joblib - the filter-based
    recommend_restaurants() function below never needs it.
    """
    if _cache["recommendation_data"] is None:
        rec_data = joblib.load(MODELS_DIR / "recommendation_data.joblib")
        tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        _cache["recommendation_data"] = {
            "data": rec_data,
            "tfidf": tfidf
        }
    return _cache["recommendation_data"]


def _load_location_data() -> pd.DataFrame:
    if _cache["location_data"] is None:
        df = pd.read_csv(PROCESSED_DATA_DIR / "featured_restaurants.csv")
        _cache["location_data"] = df[df["Has_Valid_Coordinates"] == 1].copy()
    return _cache["location_data"]


# ============================================
# INPUT VALIDATION HELPERS
# ============================================

class ValidationError(Exception):
    """Raised when inference input fails validation."""
    pass


def _require_fields(input_dict: dict, required: List[str]):
    missing = [f for f in required if f not in input_dict or input_dict[f] is None]
    if missing:
        raise ValidationError(f"Missing required fields: {missing}")


def _validate_range(value, name, low, high):
    if value is not None and not (low <= value <= high):
        raise ValidationError(f"'{name}' must be between {low} and {high}, got {value}")


# ============================================
# 1. predict_rating()
# ============================================

def predict_rating(restaurant_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict Aggregate Rating for a restaurant given raw feature input.

    Parameters
    ----------
    restaurant_input : dict
        Raw restaurant features. Required keys match the model bundle's
        'required_input_columns'. Missing optional numeric fields default
        to 0; missing categorical fields default to 'Unknown'.

    Returns
    -------
    dict with keys: predicted_rating, model_version, warnings
    """
    bundle = _load_rating_bundle()
    model = bundle["model"]
    required_cols = bundle["required_input_columns"]
    numerical = bundle["numerical_features"]
    categorical = bundle["categorical_features"]

    warnings = []
    row = {}

    for col in numerical:
        if col not in restaurant_input or restaurant_input[col] is None:
            row[col] = 0
            warnings.append(f"'{col}' missing - defaulted to 0")
        else:
            row[col] = restaurant_input[col]

    for col in categorical:
        if col not in restaurant_input or restaurant_input[col] is None:
            row[col] = "Unknown"
            warnings.append(f"'{col}' missing - defaulted to 'Unknown'")
        else:
            row[col] = restaurant_input[col]

    X = pd.DataFrame([row])[required_cols]

    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise ValidationError(f"Prediction failed: {e}")

    pred = float(np.clip(pred, 0.0, 5.0))

    return {
        "predicted_rating": round(pred, 2),
        "model_version": bundle["version"],
        "model_notes": bundle["notes"],
        "warnings": warnings
    }


# ============================================
# 2. recommend_restaurants()
# ============================================

def recommend_restaurants(
    city: Optional[str] = None,
    cuisine: Optional[str] = None,
    min_rating: float = 0.0,
    max_cost: Optional[float] = None,
    price_range: Optional[int] = None,
    online_delivery: Optional[bool] = None,
    table_booking: Optional[bool] = None,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Content/filter-based restaurant recommendation.

    Returns
    -------
    dict with keys: results (list of dicts), count, filters_applied
    """
    if top_n < 1 or top_n > 100:
        raise ValidationError("'top_n' must be between 1 and 100")
    _validate_range(min_rating, "min_rating", 0.0, 5.0)
    if price_range is not None and price_range not in (1, 2, 3, 4):
        raise ValidationError("'price_range' must be one of 1, 2, 3, 4")

    rec = _load_recommendation_data()
    results = rec["data"].copy()

    filters_applied = {}

    if city:
        results = results[results['City'].str.contains(city, case=False, na=False)]
        filters_applied["city"] = city

    if cuisine:
        results = results[results['Cuisines'].str.contains(cuisine, case=False, na=False)]
        filters_applied["cuisine"] = cuisine

    if min_rating > 0:
        results = results[results['Aggregate rating'] >= min_rating]
        filters_applied["min_rating"] = min_rating

    if max_cost is not None:
        results = results[results['Average Cost for two'] <= max_cost]
        filters_applied["max_cost"] = max_cost

    if price_range is not None:
        results = results[results['Price range'] == price_range]
        filters_applied["price_range"] = price_range

    if online_delivery is not None:
        results = results[results['Has Online delivery'] == int(online_delivery)]
        filters_applied["online_delivery"] = online_delivery

    if table_booking is not None:
        results = results[results['Has Table booking'] == int(table_booking)]
        filters_applied["table_booking"] = table_booking

    results = results.sort_values(
        by=['Aggregate rating', 'Votes'], ascending=[False, False]
    ).head(top_n)

    output = []
    for rank, (_, row) in enumerate(results.iterrows(), 1):
        output.append({
            "rank": rank,
            "restaurant": row['Restaurant Name'],
            "cuisine": row['Cuisines'],
            "rating": float(row['Aggregate rating']),
            "votes": int(row['Votes']),
            "price_range": int(row['Price range']),
            "average_cost_for_two": float(row['Average Cost for two']),
            "currency": row['Currency'],
            "city": row['City'],
            "locality": row['Locality'],
            "online_delivery": bool(row['Has Online delivery']),
            "table_booking": bool(row['Has Table booking']),
            "reason": f"Rating {row['Aggregate rating']} matches your preferences"
        })

    return {
        "results": output,
        "count": len(output),
        "filters_applied": filters_applied
    }


# ============================================
# 3. predict_cuisine()
# ============================================

def predict_cuisine(restaurant_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict the Primary Cuisine of a restaurant given raw feature input.

    Returns
    -------
    dict with keys: predicted_cuisine, confidence, top_3_predictions, warnings
    """
    bundle = _load_cuisine_bundle()
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    required_cols = bundle["required_input_columns"]

    warnings = []
    row = {}
    for col in required_cols:
        if col not in restaurant_input or restaurant_input[col] is None:
            row[col] = 0 if col != "City" else "Unknown"
            warnings.append(f"'{col}' missing - defaulted")
        else:
            row[col] = restaurant_input[col]

    X = pd.DataFrame([row])[required_cols]

    try:
        probs = model.predict_proba(X)[0]
    except Exception as e:
        raise ValidationError(f"Prediction failed: {e}")

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"cuisine": label_encoder.classes_[i], "probability": round(float(probs[i]), 4)}
        for i in top3_idx
    ]

    return {
        "predicted_cuisine": top3[0]["cuisine"],
        "confidence": top3[0]["probability"],
        "top_3_predictions": top3,
        "model_version": bundle["version"],
        "num_classes": bundle["num_classes"],
        "warnings": warnings
    }


# ============================================
# 4. analyze_location()
# ============================================

def analyze_location(city: str, locality: Optional[str] = None) -> Dict[str, Any]:
    """
    Descriptive geospatial analytics for a given city (and optionally locality).

    Returns
    -------
    dict with city-level and (optional) locality-level statistics
    """
    if not city or not city.strip():
        raise ValidationError("'city' is required and cannot be empty")

    df = _load_location_data()
    city_df = df[df['City'].str.lower() == city.strip().lower()]

    if city_df.empty:
        return {
            "city": city,
            "found": False,
            "message": f"No restaurants found for city '{city}'. Check spelling or try a nearby major city."
        }

    result = {
        "city": city,
        "found": True,
        "restaurant_count": int(len(city_df)),
        "avg_rating": round(float(city_df['Aggregate rating'].mean()), 3),
        "avg_cost_for_two": round(float(city_df['Average Cost for two'].mean()), 2),
        "avg_votes": round(float(city_df['Votes'].mean()), 1),
        "online_delivery_pct": round(float(city_df['Has Online delivery'].mean() * 100), 1),
        "table_booking_pct": round(float(city_df['Has Table booking'].mean() * 100), 1),
        "top_cuisines": city_df['Primary_Cuisine'].value_counts().head(5).to_dict()
    }

    if locality:
        loc_df = city_df[city_df['Locality'].str.lower() == locality.strip().lower()]
        if loc_df.empty:
            result["locality"] = {
                "name": locality,
                "found": False,
                "message": f"No restaurants found for locality '{locality}' in {city}."
            }
        else:
            result["locality"] = {
                "name": locality,
                "found": True,
                "restaurant_count": int(len(loc_df)),
                "avg_rating": round(float(loc_df['Aggregate rating'].mean()), 3),
                "avg_cost_for_two": round(float(loc_df['Average Cost for two'].mean()), 2)
            }

    return result


# ============================================
# SELF-TEST (runs only if executed directly)
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 21: INFERENCE ENGINE - SELF TEST")
    print("=" * 70)

    print("\n1. Testing predict_rating()...")
    result = predict_rating({
        "City": "New Delhi",
        "Locality": "Connaught Place",
        "Currency": "Indian Rupees(Rs.)",
        "Primary_Cuisine": "North Indian",
        "Country Code": 1,
        "Longitude": 77.22,
        "Latitude": 28.63,
        "Average Cost for two": 800,
        "Price range": 2,
        "Has Table booking": 1,
        "Has Online delivery": 1,
        "Is delivering now": 0,
        "Has_Valid_Coordinates": 1,
        "Is_Zero_Cost": 0,
        "Is_Extreme_Cost": 0,
        "Cuisine_Count": 2,
        "Name_Length": 15,
        "Address_Length": 50,
        "City_Frequency": 0.573,
        "Locality_Frequency": 0.013,
        "Log_Average_Cost": np.log1p(800),
        "Cost_to_PriceRange": 400,
        "Has_Booking_and_Delivery": 1
    })
    print(f"   Predicted rating: {result['predicted_rating']}")
    print(f"   Warnings: {result['warnings']}")

    print("\n2. Testing recommend_restaurants()...")
    result = recommend_restaurants(city="New Delhi", cuisine="North Indian", min_rating=4.0, top_n=3)
    print(f"   Found {result['count']} results")
    for r in result['results']:
        print(f"   #{r['rank']} {r['restaurant']} - {r['rating']}")

    print("\n3. Testing predict_cuisine()...")
    result = predict_cuisine({
        "Country Code": 1,
        "City": "New Delhi",
        "Average Cost for two": 800,
        "Price range": 2,
        "Has Table booking": 1,
        "Has Online delivery": 1,
        "Aggregate rating": 4.2,
        "Votes": 150,
        "Cuisine_Count": 2,
        "Name_Length": 15,
        "City_Frequency": 0.573,
        "Locality_Frequency": 0.013,
        "Log_Average_Cost": np.log1p(800)
    })
    print(f"   Predicted cuisine: {result['predicted_cuisine']} (confidence: {result['confidence']})")
    print(f"   Top 3: {result['top_3_predictions']}")

    print("\n4. Testing analyze_location()...")
    result = analyze_location(city="New Delhi", locality="Connaught Place")
    print(f"   City stats: {result['restaurant_count']} restaurants, avg rating {result['avg_rating']}")
    print(f"   Locality stats: {result.get('locality')}")

    print("\n5. Testing error handling...")
    try:
        recommend_restaurants(min_rating=10.0)
    except ValidationError as e:
        print(f"   ✅ Correctly caught invalid input: {e}")

    try:
        analyze_location(city="")
    except ValidationError as e:
        print(f"   ✅ Correctly caught empty city: {e}")

    print("\n" + "=" * 70)
    print("PHASE 21 COMPLETED - Inference Engine Ready")
    print("=" * 70)