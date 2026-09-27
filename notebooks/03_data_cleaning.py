"""
PHASE 3: DATA CLEANING
Restaurant Intelligence System
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.data.loader import load_raw_data
from src.config.config import PROCESSED_DATA_DIR

print("=" * 70)
print("PHASE 3: DATA CLEANING")
print("=" * 70)

df = load_raw_data()
original_shape = df.shape
print(f"\nOriginal shape: {original_shape}")

# Create a cleaning log
cleaning_log = []

# ============================================
# 1. DROP CONSTANT COLUMN
# ============================================
print("\n" + "─" * 70)
print("1. DROP CONSTANT COLUMN: Switch to order menu")
print("─" * 70)

print(f"Unique values before: {df['Switch to order menu'].unique()}")
df = df.drop(columns=['Switch to order menu'])
cleaning_log.append("Dropped constant column: Switch to order menu")
print("→ Column dropped successfully")

# ============================================
# 2. DROP TARGET LEAKAGE COLUMNS
# ============================================
print("\n" + "─" * 70)
print("2. DROP TARGET LEAKAGE COLUMNS: Rating color & Rating text")
print("─" * 70)

df = df.drop(columns=['Rating color', 'Rating text'])
cleaning_log.append("Dropped leakage columns: Rating color, Rating text")
print("→ Columns dropped successfully")

# ============================================
# 3. HANDLE MISSING CUISINES
# ============================================
print("\n" + "─" * 70)
print("3. HANDLE MISSING VALUES IN CUISINES")
print("─" * 70)

missing_before = df['Cuisines'].isnull().sum()
print(f"Missing values before: {missing_before}")

df['Cuisines'] = df['Cuisines'].fillna('Unknown')
missing_after = df['Cuisines'].isnull().sum()
print(f"Missing values after : {missing_after}")
cleaning_log.append(f"Filled {missing_before} missing Cuisines with 'Unknown'")

# ============================================
# 4. DROP REDUNDANT COLUMN
# ============================================
print("\n" + "─" * 70)
print("4. DROP REDUNDANT COLUMN: Locality Verbose")
print("─" * 70)

df = df.drop(columns=['Locality Verbose'])
cleaning_log.append("Dropped redundant column: Locality Verbose")
print("→ Column dropped successfully")

# ============================================
# 5. CREATE BINARY FLAGS (from Yes/No)
# ============================================
print("\n" + "─" * 70)
print("5. CONVERT YES/NO COLUMNS TO BINARY (0/1)")
print("─" * 70)

binary_cols = ['Has Table booking', 'Has Online delivery', 'Is delivering now']

for col in binary_cols:
    print(f"\n{col} before:")
    print(df[col].value_counts())
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    print(f"{col} after:")
    print(df[col].value_counts())

cleaning_log.append("Converted Yes/No columns to binary (0/1)")

# ============================================
# 6. FLAG ZERO COORDINATES
# ============================================
print("\n" + "─" * 70)
print("6. FLAG ZERO COORDINATES")
print("─" * 70)

df['Has_Valid_Coordinates'] = ((df['Latitude'] != 0) | (df['Longitude'] != 0)).astype(int)
print(df['Has_Valid_Coordinates'].value_counts())
cleaning_log.append("Created flag: Has_Valid_Coordinates")

# ============================================
# 7. HANDLE ZERO / EXTREME COSTS (Flag only for now)
# ============================================
print("\n" + "─" * 70)
print("7. FLAG ZERO AND EXTREME COSTS")
print("─" * 70)

df['Is_Zero_Cost'] = (df['Average Cost for two'] == 0).astype(int)
df['Is_Extreme_Cost'] = (df['Average Cost for two'] > 100000).astype(int)

print(f"Zero cost restaurants     : {df['Is_Zero_Cost'].sum()}")
print(f"Extreme cost restaurants  : {df['Is_Extreme_Cost'].sum()}")
cleaning_log.append("Created flags: Is_Zero_Cost, Is_Extreme_Cost")

# ============================================
# 8. FINAL CHECK
# ============================================
print("\n" + "─" * 70)
print("8. FINAL CLEANED DATASET CHECK")
print("─" * 70)

print(f"\nOriginal shape : {original_shape}")
print(f"Cleaned shape  : {df.shape}")
print(f"\nRemaining columns ({len(df.columns)}):")
print(list(df.columns))

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

print("\nData types:")
print(df.dtypes)

# ============================================
# 9. SAVE CLEANED DATA
# ============================================
print("\n" + "─" * 70)
print("9. SAVE CLEANED DATASET")
print("─" * 70)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
output_path = PROCESSED_DATA_DIR / "cleaned_restaurants.csv"
df.to_csv(output_path, index=False)
print(f"✅ Cleaned data saved to: {output_path}")

# ============================================
# 10. CLEANING LOG
# ============================================
print("\n" + "─" * 70)
print("10. CLEANING LOG")
print("─" * 70)

for i, log in enumerate(cleaning_log, 1):
    print(f"{i}. {log}")

print("\n" + "=" * 70)
print("PHASE 3 COMPLETED - Data Cleaning Finished")
print("=" * 70)