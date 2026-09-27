"""
PHASE 12: Final Rating Prediction Models
- Realistic version (No Votes)
- Strong version (With Votes) for comparison
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE

print("=" * 70)
print("PHASE 12: FINAL RATING PREDICTION MODELS")
print("=" * 70)

# Load data
X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
X_test  = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").values.ravel()
y_test  = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# COMMON PREPROCESSOR COMPONENTS
# =====================================================
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

categorical_features = ['City', 'Locality', 'Currency', 'Primary_Cuisine']

# =====================================================
# VERSION 1: REALISTIC MODEL (NO VOTES)
# =====================================================
print("\n" + "─" * 70)
print("VERSION 1: REALISTIC MODEL (Without Votes)")
print("─" * 70)

numerical_no_votes = [
    'Country Code', 'Longitude', 'Latitude', 'Average Cost for two',
    'Price range', 'Has Table booking', 'Has Online delivery',
    'Is delivering now', 'Has_Valid_Coordinates', 'Is_Zero_Cost',
    'Is_Extreme_Cost', 'Cuisine_Count', 'Name_Length', 'Address_Length',
    'City_Frequency', 'Locality_Frequency', 'Log_Average_Cost',
    'Cost_to_PriceRange', 'Has_Booking_and_Delivery'
]

preprocessor_no_votes = ColumnTransformer([
    ('num', numeric_transformer, numerical_no_votes),
    ('cat', categorical_transformer, categorical_features)
])

# Light tuning for Random Forest (No Votes)
rf_no_votes = Pipeline([
    ('preprocessor', preprocessor_no_votes),
    ('model', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
])

param_dist_rf = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 15, 20, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

print("→ Tuning Random Forest (No Votes)...")
search_rf = RandomizedSearchCV(
    rf_no_votes, param_distributions=param_dist_rf,
    n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)
search_rf.fit(X_train, y_train)

best_rf_no_votes = search_rf.best_estimator_
y_pred = best_rf_no_votes.predict(X_test)

print(f"\nBest Params: {search_rf.best_params_}")
print(f"Test MAE : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Test R²  : {r2_score(y_test, y_pred):.4f}")

# Save realistic model
joblib.dump(best_rf_no_votes, MODELS_DIR / "rating_model_realistic.joblib")
print("✅ Saved: models/rating_model_realistic.joblib")

# =====================================================
# VERSION 2: STRONG MODEL (WITH VOTES) - For comparison
# =====================================================
print("\n" + "─" * 70)
print("VERSION 2: STRONG MODEL (With Votes) - For Comparison")
print("─" * 70)

numerical_with_votes = numerical_no_votes + ['Votes']

preprocessor_with_votes = ColumnTransformer([
    ('num', numeric_transformer, numerical_with_votes),
    ('cat', categorical_transformer, categorical_features)
])

rf_with_votes = Pipeline([
    ('preprocessor', preprocessor_with_votes),
    ('model', RandomForestRegressor(
        n_estimators=200, max_depth=20, 
        random_state=RANDOM_STATE, n_jobs=-1
    ))
])

print("→ Training Random Forest (With Votes)...")
rf_with_votes.fit(X_train, y_train)
y_pred2 = rf_with_votes.predict(X_test)

print(f"Test MAE : {mean_absolute_error(y_test, y_pred2):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred2)):.4f}")
print(f"Test R²  : {r2_score(y_test, y_pred2):.4f}")

joblib.dump(rf_with_votes, MODELS_DIR / "rating_model_with_votes.joblib")
print("✅ Saved: models/rating_model_with_votes.joblib")

# =====================================================
# SUMMARY
# =====================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Realistic Model (No Votes)     → models/rating_model_realistic.joblib
Strong Model (With Votes)      → models/rating_model_with_votes.joblib

Recommendation:
- Use the REALISTIC model for actual deployment & business decisions.
- Keep the WITH VOTES model only for comparison and research.
""")
print("=" * 70)
print("PHASE 12 COMPLETED")
print("=" * 70)