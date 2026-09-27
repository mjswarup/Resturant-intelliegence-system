"""
Quick Check: Impact of 'Votes' feature on Rating Prediction
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.config.config import PROCESSED_DATA_DIR, RANDOM_STATE

print("=" * 70)
print("IMPACT OF 'VOTES' FEATURE - QUICK CHECK")
print("=" * 70)

# Load data
X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
X_test  = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").values.ravel()
y_test  = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()

# Define features WITHOUT Votes
numerical_features_no_votes = [
    'Country Code', 'Longitude', 'Latitude', 'Average Cost for two',
    'Price range', 'Has Table booking', 'Has Online delivery',
    'Is delivering now', 'Has_Valid_Coordinates', 'Is_Zero_Cost',
    'Is_Extreme_Cost', 'Cuisine_Count', 'Name_Length', 'Address_Length',
    'City_Frequency', 'Locality_Frequency', 'Log_Average_Cost',
    'Cost_to_PriceRange', 'Has_Booking_and_Delivery'
]

categorical_features = ['City', 'Locality', 'Currency', 'Primary_Cuisine']

# Preprocessor without Votes
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_features_no_votes),
    ('cat', categorical_transformer, categorical_features)
])

# Models to test
models = {
    "Random Forest (No Votes)": RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost (No Votes)": XGBRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    )
}

print("\nTraining models WITHOUT the Votes feature...\n")

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    
    print(f"{name}")
    print(f"   Test MAE : {mae:.4f}")
    print(f"   Test R²  : {r2:.4f}")
    print()

print("=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print("""
With Votes (from previous run):
  Random Forest → Test R² = 0.9632
  XGBoost       → Test R² = 0.9618

Without Votes (this run):
  See results above
""")
print("=" * 70)