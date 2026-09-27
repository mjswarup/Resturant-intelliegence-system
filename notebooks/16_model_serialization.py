"""
PHASE 20: MODEL SERIALIZATION
Package models + preprocessing + metadata into versioned bundles
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
import json

from src.config.config import (
    PROCESSED_DATA_DIR, MODELS_DIR,
    RATING_MODEL_VERSION, CUISINE_MODEL_VERSION, RANDOM_STATE
)

print("=" * 70)
print("PHASE 20: MODEL SERIALIZATION")
print("=" * 70)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ============================================
# 1. BUNDLE: RATING PREDICTION MODEL (Realistic - Deployed)
# ============================================
print("\n" + "─" * 70)
print("1. BUNDLING RATING PREDICTION MODEL")
print("─" * 70)

rating_model = joblib.load(MODELS_DIR / "rating_model_realistic.joblib")

X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = rating_model.predict(X_test)

rating_metrics = {
    "test_mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
    "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
    "test_r2": round(float(r2_score(y_test, y_pred)), 4)
}

numerical_features_no_votes = [
    'Country Code', 'Longitude', 'Latitude', 'Average Cost for two',
    'Price range', 'Has Table booking', 'Has Online delivery',
    'Is delivering now', 'Has_Valid_Coordinates', 'Is_Zero_Cost',
    'Is_Extreme_Cost', 'Cuisine_Count', 'Name_Length', 'Address_Length',
    'City_Frequency', 'Locality_Frequency', 'Log_Average_Cost',
    'Cost_to_PriceRange', 'Has_Booking_and_Delivery'
]
categorical_features = ['City', 'Locality', 'Currency', 'Primary_Cuisine']

rating_bundle = {
    "model": rating_model,
    "task": "rating_prediction",
    "version": RATING_MODEL_VERSION,
    "algorithm": "RandomForestRegressor",
    "target": "Aggregate rating",
    "numerical_features": numerical_features_no_votes,
    "categorical_features": categorical_features,
    "required_input_columns": numerical_features_no_votes + categorical_features,
    "metrics": rating_metrics,
    "training_metadata": {
        "trained_on_rows": 7640,
        "test_rows": 1911,
        "random_state": RANDOM_STATE,
        "excludes_votes": True,
        "exclusion_reason": "Votes causes target leakage - see Phase 12 ablation study",
        "created_at": timestamp
    },
    "notes": "Deployment model. Excludes 'Votes' due to leakage risk identified in Phase 12."
}

joblib.dump(rating_bundle, MODELS_DIR / f"rating_model_bundle_{RATING_MODEL_VERSION}.joblib")
print(f"✅ Saved: models/rating_model_bundle_{RATING_MODEL_VERSION}.joblib")
print(f"   Metrics: {rating_metrics}")

# ============================================
# 2. BUNDLE: CUISINE CLASSIFICATION MODEL
# ============================================
print("\n" + "─" * 70)
print("2. BUNDLING CUISINE CLASSIFICATION MODEL")
print("─" * 70)

cuisine_model = joblib.load(MODELS_DIR / "cuisine_classifier.joblib")
label_encoder = joblib.load(MODELS_DIR / "cuisine_label_encoder.joblib")

cuisine_feature_cols = [
    'Country Code', 'City', 'Average Cost for two', 'Price range',
    'Has Table booking', 'Has Online delivery', 'Aggregate rating',
    'Votes', 'Cuisine_Count', 'Name_Length', 'City_Frequency',
    'Locality_Frequency', 'Log_Average_Cost'
]

cuisine_bundle = {
    "model": cuisine_model,
    "label_encoder": label_encoder,
    "task": "cuisine_classification",
    "version": CUISINE_MODEL_VERSION,
    "algorithm": "XGBClassifier",
    "target": "Primary_Cuisine",
    "classes": list(label_encoder.classes_),
    "num_classes": len(label_encoder.classes_),
    "required_input_columns": cuisine_feature_cols,
    "metrics": {
        "accuracy": 0.4287,
        "weighted_f1": 0.3856
    },
    "training_metadata": {
        "min_samples_per_class": 50,
        "random_state": RANDOM_STATE,
        "created_at": timestamp,
        "class_imbalance_note": "North Indian dominates (~34% of filtered data)"
    },
    "notes": "Single-label classification on Primary Cuisine (first listed cuisine). Moderate accuracy due to class imbalance and limited discriminative features - see Phase 19."
}

joblib.dump(cuisine_bundle, MODELS_DIR / f"cuisine_model_bundle_{CUISINE_MODEL_VERSION}.joblib")
print(f"✅ Saved: models/cuisine_model_bundle_{CUISINE_MODEL_VERSION}.joblib")
print(f"   Classes: {len(label_encoder.classes_)} | Accuracy: 0.4287")

# ============================================
# 3. BUNDLE: RECOMMENDATION SYSTEM
# ============================================
print("\n" + "─" * 70)
print("3. BUNDLING RECOMMENDATION SYSTEM")
print("─" * 70)

rec_data = joblib.load(MODELS_DIR / "recommendation_data.joblib")
tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
cosine_sim = joblib.load(MODELS_DIR / "cosine_similarity.joblib")

recommendation_bundle = {
    "data": rec_data,
    "tfidf_vectorizer": tfidf,
    "cosine_similarity_matrix": cosine_sim,
    "task": "recommendation",
    "version": "v1.0",
    "approach": "content_based_tfidf",
    "content_fields": ["Cuisines", "City", "Locality", "Primary_Cuisine"],
    "filterable_fields": [
        "city", "cuisine", "min_rating", "max_cost",
        "price_range", "online_delivery", "table_booking"
    ],
    "training_metadata": {
        "unique_restaurants": len(rec_data),
        "tfidf_max_features": 5000,
        "created_at": timestamp
    },
    "notes": "No supervised target - evaluated qualitatively via test queries in Phase 14."
}

joblib.dump(recommendation_bundle, MODELS_DIR / "recommendation_bundle_v1.0.joblib")
print(f"✅ Saved: models/recommendation_bundle_v1.0.joblib")
print(f"   Restaurants indexed: {len(rec_data)}")

# ============================================
# 4. MASTER MODEL REGISTRY
# ============================================
print("\n" + "─" * 70)
print("4. CREATING MASTER MODEL REGISTRY")
print("─" * 70)

registry = {
    "system_name": "Restaurant Intelligence System",
    "last_updated": timestamp,
    "models": {
        "rating_prediction": {
            "file": f"rating_model_bundle_{RATING_MODEL_VERSION}.joblib",
            "version": RATING_MODEL_VERSION,
            "type": "regression",
            "status": "production",
            "metrics": rating_metrics
        },
        "cuisine_classification": {
            "file": f"cuisine_model_bundle_{CUISINE_MODEL_VERSION}.joblib",
            "version": CUISINE_MODEL_VERSION,
            "type": "classification",
            "status": "production",
            "metrics": {"accuracy": 0.4287, "weighted_f1": 0.3856}
        },
        "recommendation": {
            "file": "recommendation_bundle_v1.0.joblib",
            "version": "v1.0",
            "type": "content_based_filtering",
            "status": "production",
            "metrics": None
        },
        "location_intelligence": {
            "file": None,
            "version": "v1.0",
            "type": "descriptive_analytics",
            "status": "production",
            "metrics": None,
            "note": "No saved model - runs aggregation on processed data directly"
        }
    },
    "future_extension_points": [
        "demand_prediction", "revenue_prediction", "sales_forecasting",
        "customer_segmentation", "customer_churn_prediction",
        "sentiment_analysis", "restaurant_success_prediction",
        "price_optimization"
    ]
}

with open(MODELS_DIR / "model_registry.json", "w") as f:
    json.dump(registry, f, indent=2)

print("✅ Saved: models/model_registry.json")

# ============================================
# 5. VERIFY BUNDLE INTEGRITY
# ============================================
print("\n" + "─" * 70)
print("5. VERIFYING BUNDLES LOAD CORRECTLY")
print("─" * 70)

test_rating = joblib.load(MODELS_DIR / f"rating_model_bundle_{RATING_MODEL_VERSION}.joblib")
test_cuisine = joblib.load(MODELS_DIR / f"cuisine_model_bundle_{CUISINE_MODEL_VERSION}.joblib")
test_rec = joblib.load(MODELS_DIR / "recommendation_bundle_v1.0.joblib")

print(f"✅ Rating bundle loads. Keys: {list(test_rating.keys())}")
print(f"✅ Cuisine bundle loads. Keys: {list(test_cuisine.keys())}")
print(f"✅ Recommendation bundle loads. Keys: {list(test_rec.keys())}")

# ============================================
# 6. LIST ALL MODEL ARTIFACTS
# ============================================
print("\n" + "─" * 70)
print("6. ALL FILES IN models/ DIRECTORY")
print("─" * 70)

for f in sorted(MODELS_DIR.glob("*")):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<50} {size_kb:>10.1f} KB")

print("\n" + "=" * 70)
print("PHASE 20 COMPLETED - Model Serialization Finished")
print("=" * 70)