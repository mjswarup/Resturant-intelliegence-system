"""
PHASE 4: TARGET AND DATA-LEAKAGE IDENTIFICATION
Restaurant Intelligence System
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
from src.config.config import PROCESSED_DATA_DIR

print("=" * 70)
print("PHASE 4: TARGET AND DATA-LEAKAGE IDENTIFICATION")
print("=" * 70)

# Load cleaned data
df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_restaurants.csv")
print(f"Loaded cleaned data: {df.shape}")

# ============================================
# 1. DEFINE TARGETS FOR EACH TASK
# ============================================
print("\n" + "─" * 70)
print("1. TARGET DEFINITION FOR EACH TASK")
print("─" * 70)

print("""
TASK 1: Restaurant Rating Prediction
------------------------------------
Target Variable     : Aggregate rating
Problem Type        : Regression
Target Range        : 0.0 to 4.9
Special Note        : 2148 restaurants have rating = 0.0 (Not rated)

TASK 2: Restaurant Recommendation
---------------------------------
No conventional supervised target
Approach            : Content-based filtering using features
                      (Cuisines, City, Price range, Cost, Rating, etc.)

TASK 3: Cuisine Classification
------------------------------
Target Variable     : Cuisines
Problem Type        : Multi-label Classification (most restaurants have multiple cuisines)
Alternative         : Can also extract Primary Cuisine for single-label

TASK 4: Location-Based Intelligence
-----------------------------------
No ML target initially
Approach            : Geospatial analysis using Latitude, Longitude, City, Locality
""")

# ============================================
# 2. DETAILED LEAKAGE ANALYSIS
# ============================================
print("\n" + "─" * 70)
print("2. FEATURE LEAKAGE RISK ANALYSIS")
print("─" * 70)

leakage_analysis = [
    {
        "Feature": "Restaurant ID",
        "Leakage Risk": "High",
        "Decision": "DROP for all models",
        "Reason": "Pure identifier, unique for every restaurant"
    },
    {
        "Feature": "Restaurant Name",
        "Leakage Risk": "Medium",
        "Decision": "DROP for Rating Prediction | KEEP for Recommendation",
        "Reason": "Can cause overfitting; useful for content-based recommendation"
    },
    {
        "Feature": "Address",
        "Leakage Risk": "Medium",
        "Decision": "DROP or extract features only",
        "Reason": "Extremely high cardinality, mostly unique"
    },
    {
        "Feature": "Votes",
        "Leakage Risk": "High (Potential)",
        "Decision": "INVESTIGATE carefully | Possibly DROP or use with caution",
        "Reason": "Votes are usually collected after ratings. Strong correlation with rating expected."
    },
    {
        "Feature": "Rating color / Rating text",
        "Leakage Risk": "Critical",
        "Decision": "ALREADY DROPPED",
        "Reason": "Directly derived from Aggregate rating"
    },
    {
        "Feature": "Aggregate rating",
        "Leakage Risk": "N/A",
        "Decision": "TARGET for Task 1",
        "Reason": "This is the prediction target"
    },
    {
        "Feature": "Cuisines",
        "Leakage Risk": "None",
        "Decision": "KEEP (Target for Task 3 / Feature for others)",
        "Reason": "Independent feature"
    },
    {
        "Feature": "Average Cost for two + Currency",
        "Leakage Risk": "None",
        "Decision": "KEEP",
        "Reason": "Important business feature"
    },
    {
        "Feature": "Has Table booking / Online delivery",
        "Leakage Risk": "None",
        "Decision": "KEEP",
        "Reason": "Useful predictive features"
    },
    {
        "Feature": "Latitude / Longitude / City / Locality",
        "Leakage Risk": "None",
        "Decision": "KEEP",
        "Reason": "Core for Location Intelligence + useful for other tasks"
    }
]

leakage_df = pd.DataFrame(leakage_analysis)
print(leakage_df.to_string(index=False))

# ============================================
# 3. CORRELATION CHECK: Votes vs Rating
# ============================================
print("\n" + "─" * 70)
print("3. VOTES vs AGGREGATE RATING CORRELATION")
print("─" * 70)

correlation = df['Votes'].corr(df['Aggregate rating'])
print(f"Pearson Correlation (Votes vs Aggregate rating): {correlation:.4f}")

print("\nInterpretation:")
if abs(correlation) > 0.5:
    print("→ Strong correlation. Votes may contain leakage or post-rating information.")
    print("→ Recommendation: Be cautious. Consider models with and without Votes.")
else:
    print("→ Moderate/Weak correlation. Votes can be kept with monitoring.")

# ============================================
# 4. FINAL FEATURE SETS (PRELIMINARY)
# ============================================
print("\n" + "─" * 70)
print("4. PRELIMINARY FEATURE SETS")
print("─" * 70)

print("""
FOR TASK 1 (Rating Prediction - Regression):
--------------------------------------------
DROP     : Restaurant ID, Restaurant Name, Address, Rating color, Rating text
CAUTION  : Votes (test both with and without)
KEEP     : Country Code, City, Locality, Longitude, Latitude,
           Cuisines (engineered), Average Cost for two, Currency,
           Has Table booking, Has Online delivery, Is delivering now,
           Price range, Has_Valid_Coordinates

FOR TASK 3 (Cuisine Classification):
------------------------------------
Target   : Cuisines (multi-label)
Features : Similar to above + Aggregate rating + Votes possible

FOR TASK 2 (Recommendation):
----------------------------
Use almost all available features including Restaurant Name
""")

print("\n" + "=" * 70)
print("PHASE 4 COMPLETED - Target & Leakage Analysis Finished")
print("=" * 70)