"""
PHASE 1: DATASET DISCOVERY
Restaurant Intelligence System
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.data.loader import load_raw_data

# ============================================
# 1. LOAD DATA
# ============================================
print("=" * 70)
print("PHASE 1: DATASET DISCOVERY")
print("=" * 70)

df = load_raw_data()

# ============================================
# 2. BASIC INFORMATION
# ============================================
print("\n" + "─" * 70)
print("1. BASIC DATASET INFORMATION")
print("─" * 70)

print(f"\nShape              : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory Usage       : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Duplicate Rows     : {df.duplicated().sum():,}")

# ============================================
# 3. COLUMN OVERVIEW
# ============================================
print("\n" + "─" * 70)
print("2. COLUMN OVERVIEW")
print("─" * 70)

info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.values,
    "Non-Null Count": df.count().values,
    "Null Count": df.isnull().sum().values,
    "Null %": (df.isnull().sum() / len(df) * 100).round(2).values,
    "Unique Values": [df[col].nunique() for col in df.columns]
})

print(info_df.to_string(index=False))

# ============================================
# 4. DATA TYPES SUMMARY
# ============================================
print("\n" + "─" * 70)
print("3. DATA TYPES SUMMARY")
print("─" * 70)
print(df.dtypes.value_counts())

# ============================================
# 5. NUMERICAL COLUMNS STATISTICS
# ============================================
print("\n" + "─" * 70)
print("4. NUMERICAL COLUMNS - DESCRIPTIVE STATISTICS")
print("─" * 70)
print(df.describe().T)

# ============================================
# 6. CATEGORICAL / OBJECT COLUMNS SAMPLE
# ============================================
print("\n" + "─" * 70)
print("5. CATEGORICAL COLUMNS - SAMPLE VALUES")
print("─" * 70)

object_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in object_cols:
    print(f"\n▶ {col}")
    print(f"   Unique values : {df[col].nunique()}")
    print(f"   Sample values : {df[col].dropna().unique()[:5].tolist()}")

# ============================================
# 7. POTENTIAL TARGET COLUMNS
# ============================================
print("\n" + "─" * 70)
print("6. POTENTIAL TARGET COLUMNS IDENTIFICATION")
print("─" * 70)

print("""
Task 1 - Rating Prediction     → Target: Aggregate rating
Task 2 - Recommendation        → No single target (content-based)
Task 3 - Cuisine Classification→ Target: Cuisines
Task 4 - Location Intelligence → Geospatial (Latitude, Longitude, City, Locality)
""")

# ============================================
# 8. KEY COLUMNS QUICK VIEW
# ============================================
print("\n" + "─" * 70)
print("7. KEY COLUMNS QUICK VIEW")
print("─" * 70)

print("\nAggregate rating distribution:")
print(df['Aggregate rating'].value_counts().sort_index().head(10))

print("\nPrice range distribution:")
print(df['Price range'].value_counts().sort_index())

print("\nTop 10 Cities:")
print(df['City'].value_counts().head(10))

print("\nHas Online delivery:")
print(df['Has Online delivery'].value_counts())

print("\nHas Table booking:")
print(df['Has Table booking'].value_counts())

print("\n" + "=" * 70)
print("PHASE 1 COMPLETED - Dataset Discovery Finished")
print("=" * 70)