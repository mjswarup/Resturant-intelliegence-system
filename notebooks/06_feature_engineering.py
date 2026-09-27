"""
PHASE 6 & 7: FEATURE EXTRACTION + FEATURE ENGINEERING
Restaurant Intelligence System
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.config.config import PROCESSED_DATA_DIR

print("=" * 70)
print("PHASE 6 & 7: FEATURE EXTRACTION + FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_restaurants.csv")
print(f"Loaded cleaned data: {df.shape}")

# ============================================
# 1. CUISINE FEATURES
# ============================================
print("\n" + "─" * 70)
print("1. CUISINE FEATURES")
print("─" * 70)

# Number of cuisines
df['Cuisine_Count'] = df['Cuisines'].apply(lambda x: len(str(x).split(',')))

# Primary Cuisine (first one listed)
df['Primary_Cuisine'] = df['Cuisines'].apply(lambda x: str(x).split(',')[0].strip())

print(f"Cuisine_Count created. Mean: {df['Cuisine_Count'].mean():.2f}")
print(f"Primary_Cuisine created. Unique values: {df['Primary_Cuisine'].nunique()}")

# ============================================
# 2. TEXT LENGTH FEATURES
# ============================================
print("\n" + "─" * 70)
print("2. TEXT LENGTH FEATURES")
print("─" * 70)

df['Name_Length'] = df['Restaurant Name'].astype(str).apply(len)
df['Address_Length'] = df['Address'].astype(str).apply(len)

print(f"Name_Length   → mean: {df['Name_Length'].mean():.1f}")
print(f"Address_Length→ mean: {df['Address_Length'].mean():.1f}")

# ============================================
# 3. CITY & LOCALITY FREQUENCY ENCODING
# ============================================
print("\n" + "─" * 70)
print("3. FREQUENCY ENCODING (City & Locality)")
print("─" * 70)

city_freq = df['City'].value_counts(normalize=True)
df['City_Frequency'] = df['City'].map(city_freq)

locality_freq = df['Locality'].value_counts(normalize=True)
df['Locality_Frequency'] = df['Locality'].map(locality_freq)

print("City_Frequency and Locality_Frequency created.")

# ============================================
# 4. COST FEATURES
# ============================================
print("\n" + "─" * 70)
print("4. COST RELATED FEATURES")
print("─" * 70)

# Log transform of cost (handles skewness)
df['Log_Average_Cost'] = np.log1p(df['Average Cost for two'])

# Cost per price range (simple interaction idea)
df['Cost_to_PriceRange'] = df['Average Cost for two'] / df['Price range']

print("Log_Average_Cost and Cost_to_PriceRange created.")

# ============================================
# 5. BINARY & INTERACTION FEATURES
# ============================================
print("\n" + "─" * 70)
print("5. INTERACTION FEATURES")
print("─" * 70)

df['Has_Booking_and_Delivery'] = ((df['Has Table booking'] == 1) & 
                                   (df['Has Online delivery'] == 1)).astype(int)

print(f"Has_Booking_and_Delivery created. Count: {df['Has_Booking_and_Delivery'].sum()}")

# ============================================
# 6. RATING RELATED HELPER
# ============================================
print("\n" + "─" * 70)
print("6. RATING HELPER FEATURE")
print("─" * 70)

df['Is_Rated'] = (df['Aggregate rating'] > 0).astype(int)
print(f"Is_Rated created. Rated restaurants: {df['Is_Rated'].sum()}")

# ============================================
# 7. FINAL FEATURE LIST
# ============================================
print("\n" + "─" * 70)
print("7. FINAL FEATURE SUMMARY")
print("─" * 70)

print(f"\nTotal columns after feature engineering: {df.shape[1]}")
print("\nNewly created features:")
new_features = [
    'Cuisine_Count', 'Primary_Cuisine', 'Name_Length', 'Address_Length',
    'City_Frequency', 'Locality_Frequency', 'Log_Average_Cost',
    'Cost_to_PriceRange', 'Has_Booking_and_Delivery', 'Is_Rated'
]
for f in new_features:
    print(f"  • {f}")

# ============================================
# 8. SAVE FEATURE-ENGINEERED DATA
# ============================================
print("\n" + "─" * 70)
print("8. SAVE FEATURE-ENGINEERED DATASET")
print("─" * 70)

output_path = PROCESSED_DATA_DIR / "featured_restaurants.csv"
df.to_csv(output_path, index=False)
print(f"✅ Feature-engineered data saved to: {output_path}")

print("\n" + "=" * 70)
print("PHASE 6 & 7 COMPLETED - Feature Engineering Finished")
print("=" * 70)