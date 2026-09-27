"""
PHASE 2: DATA QUALITY ANALYSIS
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

print("=" * 70)
print("PHASE 2: DATA QUALITY ANALYSIS")
print("=" * 70)

df = load_raw_data()

# ============================================
# 1. MISSING VALUES DETAIL
# ============================================
print("\n" + "─" * 70)
print("1. MISSING VALUES ANALYSIS")
print("─" * 70)

missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)
    print("\nRows with missing Cuisines:")
    print(df[df['Cuisines'].isnull()][['Restaurant Name', 'City', 'Aggregate rating']].head(10))

# ============================================
# 2. CONSTANT & NEAR-CONSTANT COLUMNS
# ============================================
print("\n" + "─" * 70)
print("2. CONSTANT / NEAR-CONSTANT COLUMNS")
print("─" * 70)

for col in df.columns:
    nunique = df[col].nunique()
    if nunique == 1:
        print(f"⚠ CONSTANT → {col} (only value: {df[col].unique()[0]})")
    elif nunique <= 3:
        print(f"• Low cardinality → {col} ({nunique} unique values): {df[col].unique().tolist()}")

# ============================================
# 3. INVALID GEOGRAPHIC COORDINATES
# ============================================
print("\n" + "─" * 70)
print("3. GEOGRAPHIC COORDINATES VALIDATION")
print("─" * 70)

invalid_lat = df[(df['Latitude'] < -90) | (df['Latitude'] > 90)]
invalid_lon = df[(df['Longitude'] < -180) | (df['Longitude'] > 180)]
zero_coords = df[(df['Latitude'] == 0) & (df['Longitude'] == 0)]

print(f"Invalid Latitude (< -90 or > 90) : {len(invalid_lat)}")
print(f"Invalid Longitude (< -180 or > 180): {len(invalid_lon)}")
print(f"Zero coordinates (0, 0)           : {len(zero_coords)}")

if len(zero_coords) > 0:
    print("\nSample of zero-coordinate restaurants:")
    print(zero_coords[['Restaurant Name', 'City', 'Latitude', 'Longitude']].head())

# ============================================
# 4. INVALID / SUSPICIOUS VALUES
# ============================================
print("\n" + "─" * 70)
print("4. INVALID / SUSPICIOUS VALUES")
print("─" * 70)

print(f"Negative Average Cost          : {(df['Average Cost for two'] < 0).sum()}")
print(f"Zero Average Cost              : {(df['Average Cost for two'] == 0).sum()}")
print(f"Extremely high cost (> 100000) : {(df['Average Cost for two'] > 100000).sum()}")

print(f"\nNegative Votes                 : {(df['Votes'] < 0).sum()}")
print(f"Rating < 0                     : {(df['Aggregate rating'] < 0).sum()}")
print(f"Rating > 5                     : {(df['Aggregate rating'] > 5).sum()}")

print(f"\nPrice range outside 1-4        : {(~df['Price range'].isin([1,2,3,4])).sum()}")

# ============================================
# 5. RATING vs RATING TEXT / COLOR CONSISTENCY
# ============================================
print("\n" + "─" * 70)
print("5. RATING LEAKAGE CHECK (Rating Text & Color)")
print("─" * 70)

print("\nRating Text distribution:")
print(df['Rating text'].value_counts())

print("\nRating Color distribution:")
print(df['Rating color'].value_counts())

print("\nCross-tab: Aggregate rating vs Rating text (sample)")
print(pd.crosstab(df['Aggregate rating'].round(1), df['Rating text']).head(15))

# ============================================
# 6. HIGH CARDINALITY COLUMNS
# ============================================
print("\n" + "─" * 70)
print("6. HIGH CARDINALITY COLUMNS")
print("─" * 70)

high_card = []
for col in df.columns:
    nunique = df[col].nunique()
    if nunique > 100:
        high_card.append((col, nunique))
        
for col, n in sorted(high_card, key=lambda x: -x[1]):
    print(f"{col:<25} → {n:>5} unique values")

# ============================================
# 7. SUMMARY TABLE
# ============================================
print("\n" + "─" * 70)
print("7. DATA QUALITY SUMMARY TABLE")
print("─" * 70)

summary = []
for col in df.columns:
    dtype = str(df[col].dtype)
    null_pct = round(df[col].isnull().mean() * 100, 2)
    nunique = df[col].nunique()
    
    issues = []
    if null_pct > 0:
        issues.append(f"Missing {null_pct}%")
    if nunique == 1:
        issues.append("Constant")
    if nunique > 1000:
        issues.append("Very High Cardinality")
    if col in ['Rating color', 'Rating text']:
        issues.append("Potential Target Leakage")
    if col == 'Restaurant ID':
        issues.append("Identifier")
        
    summary.append({
        "Column": col,
        "Data Type": dtype,
        "Missing %": null_pct,
        "Unique": nunique,
        "Potential Issues": ", ".join(issues) if issues else "None",
        "Recommended Action": ""
    })

summary_df = pd.DataFrame(summary)

# Add recommended actions
actions = {
    "Restaurant ID": "Drop (Identifier)",
    "Restaurant Name": "Keep for Recommendation / Drop for Rating model",
    "Country Code": "Keep or encode",
    "City": "Keep (encode carefully)",
    "Address": "Drop or extract features (length, etc.)",
    "Locality": "Keep (high cardinality → frequency encode)",
    "Locality Verbose": "Drop (redundant with Locality + City)",
    "Longitude": "Keep (validate zeros)",
    "Latitude": "Keep (validate zeros)",
    "Cuisines": "Keep (multi-label or primary cuisine)",
    "Average Cost for two": "Keep (handle outliers + currency)",
    "Currency": "Keep (important for cost interpretation)",
    "Has Table booking": "Keep (binary)",
    "Has Online delivery": "Keep (binary)",
    "Is delivering now": "Keep (binary)",
    "Switch to order menu": "Drop (constant)",
    "Price range": "Keep",
    "Aggregate rating": "TARGET for Task 1",
    "Rating color": "DROP (Leakage)",
    "Rating text": "DROP (Leakage)",
    "Votes": "Investigate carefully (possible leakage)"
}

summary_df["Recommended Action"] = summary_df["Column"].map(actions)

print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("PHASE 2 COMPLETED - Data Quality Analysis Finished")
print("=" * 70)