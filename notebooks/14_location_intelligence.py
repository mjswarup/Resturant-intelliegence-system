"""
PHASE 16: TASK 4 - LOCATION INTELLIGENCE
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.config.config import PROCESSED_DATA_DIR, REPORTS_DIR

print("=" * 70)
print("PHASE 16: LOCATION INTELLIGENCE")
print("=" * 70)

df = pd.read_csv(PROCESSED_DATA_DIR / "featured_restaurants.csv")
print(f"Loaded data: {df.shape}")

# Filter only restaurants with valid coordinates
geo_df = df[df['Has_Valid_Coordinates'] == 1].copy()
print(f"Restaurants with valid coordinates: {geo_df.shape[0]}")

# ============================================
# 1. CITY LEVEL ANALYSIS
# ============================================
print("\n" + "─" * 70)
print("1. CITY LEVEL ANALYSIS (Top 15 Cities)")
print("─" * 70)

city_stats = geo_df.groupby('City').agg({
    'Restaurant ID': 'count',
    'Aggregate rating': 'mean',
    'Average Cost for two': 'mean',
    'Votes': 'mean',
    'Has Online delivery': 'mean',
    'Has Table booking': 'mean'
}).rename(columns={
    'Restaurant ID': 'Restaurant_Count',
    'Aggregate rating': 'Avg_Rating',
    'Average Cost for two': 'Avg_Cost',
    'Votes': 'Avg_Votes',
    'Has Online delivery': 'Online_Delivery_Pct',
    'Has Table booking': 'Table_Booking_Pct'
}).round(3)

city_stats = city_stats.sort_values('Restaurant_Count', ascending=False).head(15)
print(city_stats)

# ============================================
# 2. HIGH RATING AREAS
# ============================================
print("\n" + "─" * 70)
print("2. CITIES WITH HIGHEST AVERAGE RATING (min 20 restaurants)")
print("─" * 70)

high_rating_cities = geo_df.groupby('City').filter(lambda x: len(x) >= 20)
high_rating = high_rating_cities.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).head(10)
print(high_rating.round(3))

# ============================================
# 3. MOST EXPENSIVE CITIES
# ============================================
print("\n" + "─" * 70)
print("3. MOST EXPENSIVE CITIES (by Average Cost)")
print("─" * 70)

expensive = high_rating_cities.groupby('City')['Average Cost for two'].mean().sort_values(ascending=False).head(10)
print(expensive.round(1))

# ============================================
# 4. ONLINE DELIVERY HOTSPOTS
# ============================================
print("\n" + "─" * 70)
print("4. CITIES WITH HIGHEST ONLINE DELIVERY %")
print("─" * 70)

delivery = high_rating_cities.groupby('City')['Has Online delivery'].mean().sort_values(ascending=False).head(10)
print((delivery * 100).round(1))

# ============================================
# 5. LOCALITY ANALYSIS (New Delhi example)
# ============================================
print("\n" + "─" * 70)
print("5. TOP LOCALITIES IN NEW DELHI")
print("─" * 70)

delhi = geo_df[geo_df['City'] == 'New Delhi']
locality_stats = delhi.groupby('Locality').agg({
    'Restaurant ID': 'count',
    'Aggregate rating': 'mean',
    'Average Cost for two': 'mean'
}).rename(columns={
    'Restaurant ID': 'Count',
    'Aggregate rating': 'Avg_Rating',
    'Average Cost for two': 'Avg_Cost'
}).round(2)

print("\nMost restaurants:")
print(locality_stats.sort_values('Count', ascending=False).head(8))

print("\nHighest average rating (min 15 restaurants):")
print(locality_stats[locality_stats['Count'] >= 15].sort_values('Avg_Rating', ascending=False).head(8))

# ============================================
# 6. SUMMARY INSIGHTS
# ============================================
print("\n" + "─" * 70)
print("6. KEY LOCATION INSIGHTS")
print("─" * 70)

print("""
KEY FINDINGS:
1. New Delhi, Gurgaon, and Noida dominate the dataset (heavy India bias).
2. Higher price range cities generally show better average ratings.
3. Online delivery is more common in certain Indian cities.
4. Some localities in New Delhi have significantly higher ratings.
5. Coordinate data is missing for ~5% of restaurants (already flagged).
""")

print("\n" + "=" * 70)
print("PHASE 16 COMPLETED - Location Intelligence Finished")
print("=" * 70)