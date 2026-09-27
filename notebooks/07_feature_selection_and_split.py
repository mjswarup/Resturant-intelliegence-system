"""
PHASE 8, 9, 10: FEATURE SELECTION + DATA SPLITTING + PREPROCESSING PREP
Restaurant Intelligence System - Task 1 (Rating Prediction)
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config.config import PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE

print("=" * 70)
print("PHASE 8-10: FEATURE SELECTION + SPLITTING + PREP")
print("=" * 70)

df = pd.read_csv(PROCESSED_DATA_DIR / "featured_restaurants.csv")
print(f"Loaded featured data: {df.shape}")

# ============================================
# 1. DEFINE FINAL FEATURE SET FOR RATING PREDICTION
# ============================================
print("\n" + "─" * 70)
print("1. FINAL FEATURE SET FOR TASK 1 (Rating Prediction)")
print("─" * 70)

# Columns to DROP completely for rating prediction
drop_cols = [
    'Restaurant ID',          # Identifier
    'Restaurant Name',        # High cardinality + not useful for pure rating model
    'Address',                # Extremely high cardinality
    'Cuisines',               # We already extracted Cuisine_Count + Primary_Cuisine
    'Is_Rated',               # Derived from target → leakage
]

# Target
target = 'Aggregate rating'

# Final feature list
feature_cols = [col for col in df.columns if col not in drop_cols + [target]]

print(f"\nTotal features selected: {len(feature_cols)}")
print("\nSelected Features:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

# ============================================
# 2. PREPARE X AND y
# ============================================
print("\n" + "─" * 70)
print("2. PREPARE X AND y")
print("─" * 70)

X = df[feature_cols].copy()
y = df[target].copy()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ============================================
# 3. IDENTIFY COLUMN TYPES
# ============================================
print("\n" + "─" * 70)
print("3. COLUMN TYPE IDENTIFICATION")
print("─" * 70)

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

# Verify
print(f"\nNumerical features ({len(numerical_features)}):")
print(numerical_features)

print(f"\nCategorical features ({len(categorical_features)}):")
print(categorical_features)

# Quick check
missing_in_X = set(feature_cols) - set(numerical_features + categorical_features)
if missing_in_X:
    print(f"\n⚠ Warning - columns not assigned: {missing_in_X}")
else:
    print("\n✅ All features correctly assigned to numerical or categorical.")

# ============================================
# 4. TRAIN - TEST SPLIT
# ============================================
print("\n" + "─" * 70)
print("4. TRAIN - TEST SPLIT (80/20)")
print("─" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape}")
print(f"y_test  : {y_test.shape}")

# ============================================
# 5. SAVE SPLITS
# ============================================
print("\n" + "─" * 70)
print("5. SAVE TRAIN & TEST SETS")
print("─" * 70)

X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

print("✅ Train and Test sets saved successfully.")

# ============================================
# 6. SUMMARY
# ============================================
print("\n" + "─" * 70)
print("6. SUMMARY")
print("─" * 70)

print(f"""
Task                  : Restaurant Rating Prediction (Regression)
Target                : Aggregate rating
Total features        : {len(feature_cols)}
Numerical features    : {len(numerical_features)}
Categorical features  : {len(categorical_features)}
Train size            : {X_train.shape[0]:,} rows
Test size             : {X_test.shape[0]:,} rows
Random state          : {RANDOM_STATE}
""")

print("=" * 70)
print("PHASE 8-10 COMPLETED")
print("=" * 70)