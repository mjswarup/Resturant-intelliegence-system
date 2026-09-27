"""
PHASE 11: TASK 1 - RESTAURANT RATING PREDICTION
Model Development & Comparison
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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from src.config.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE

print("=" * 70)
print("PHASE 11: TASK 1 - RATING PREDICTION (MODEL DEVELOPMENT)")
print("=" * 70)

# ============================================
# 1. LOAD DATA
# ============================================
X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
X_test  = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").values.ravel()
y_test  = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}  | y_test : {y_test.shape}")

# ============================================
# 2. DEFINE COLUMN TYPES
# ============================================
numerical_features = [
    'Country Code', 'Longitude', 'Latitude', 'Average Cost for two',
    'Price range', 'Votes', 'Has Table booking', 'Has Online delivery',
    'Is delivering now', 'Has_Valid_Coordinates', 'Is_Zero_Cost',
    'Is_Extreme_Cost', 'Cuisine_Count', 'Name_Length', 'Address_Length',
    'City_Frequency', 'Locality_Frequency', 'Log_Average_Cost',
    'Cost_to_PriceRange', 'Has_Booking_and_Delivery'
]

categorical_features = [
    'City', 'Locality', 'Currency', 'Primary_Cuisine'
]

# ============================================
# 3. BUILD PREPROCESSING PIPELINE
# ============================================
print("\n" + "─" * 70)
print("3. BUILDING PREPROCESSING PIPELINE")
print("─" * 70)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print("✅ Preprocessor created successfully")

# ============================================
# 4. DEFINE MODELS
# ============================================
print("\n" + "─" * 70)
print("4. DEFINING MODELS")
print("─" * 70)

models = {
    "Dummy Regressor": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "XGBoost": XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
}

# ============================================
# 5. TRAIN & EVALUATE
# ============================================
print("\n" + "─" * 70)
print("5. TRAINING & EVALUATION")
print("─" * 70)

results = []

for name, model in models.items():
    print(f"\n→ Training: {name}")
    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit
    pipe.fit(X_train, y_train)
    
    # Predict
    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)
    
    # Metrics
    train_mae  = mean_absolute_error(y_train, y_pred_train)
    test_mae   = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2   = r2_score(y_train, y_pred_train)
    test_r2    = r2_score(y_test, y_pred_test)
    
    # Cross-validation (on training set)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std  = cv_scores.std()
    
    results.append({
        "Model": name,
        "Train MAE": round(train_mae, 4),
        "Test MAE": round(test_mae, 4),
        "Train RMSE": round(train_rmse, 4),
        "Test RMSE": round(test_rmse, 4),
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "CV R² Mean": round(cv_mean, 4),
        "CV R² Std": round(cv_std, 4)
    })
    
    print(f"   Test MAE : {test_mae:.4f} | Test R² : {test_r2:.4f} | CV R² : {cv_mean:.4f} ± {cv_std:.4f}")

# ============================================
# 6. RESULTS TABLE
# ============================================
print("\n" + "─" * 70)
print("6. MODEL COMPARISON RESULTS")
print("─" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Test R²", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(PROCESSED_DATA_DIR / "model_comparison_rating.csv", index=False)
print("\n✅ Results saved to data/processed/model_comparison_rating.csv")

# ============================================
# 7. SELECT BEST MODEL (by Test R²)
# ============================================
print("\n" + "─" * 70)
print("7. BEST MODEL SELECTION")
print("─" * 70)

best_model_name = results_df.iloc[0]["Model"]
print(f"Best model based on Test R²: {best_model_name}")

print("\n" + "=" * 70)
print("PHASE 11 COMPLETED - Model Comparison Finished")
print("=" * 70)