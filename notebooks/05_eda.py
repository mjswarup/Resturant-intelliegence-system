"""
PHASE 5: EXPLORATORY DATA ANALYSIS
Restaurant Intelligence System
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config.config import PROCESSED_DATA_DIR, REPORTS_DIR

# Create reports folder
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PHASE 5: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_restaurants.csv")
print(f"Loaded cleaned data: {df.shape}")

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

# ============================================
# 1. TARGET DISTRIBUTION (Aggregate rating)
# ============================================
print("\n" + "─" * 70)
print("1. TARGET DISTRIBUTION: Aggregate rating")
print("─" * 70)

print(df['Aggregate rating'].describe())
print(f"\nRestaurants with rating = 0 (Not rated): {(df['Aggregate rating'] == 0).sum()}")
print(f"Percentage of Not rated               : {(df['Aggregate rating'] == 0).mean()*100:.1f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Aggregate rating'], bins=30, kde=True, color="steelblue")
plt.title("Distribution of Aggregate Rating")
plt.xlabel("Aggregate Rating")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Aggregate rating'], color="lightcoral")
plt.title("Boxplot of Aggregate Rating")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "01_rating_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Saved: reports/01_rating_distribution.png")

# ============================================
# 2. PRICE RANGE & COST ANALYSIS
# ============================================
print("\n" + "─" * 70)
print("2. PRICE RANGE & AVERAGE COST")
print("─" * 70)

print("\nPrice Range Distribution:")
print(df['Price range'].value_counts().sort_index())

print("\nAverage Cost by Price Range:")
print(df.groupby('Price range')['Average Cost for two'].agg(['mean', 'median', 'min', 'max']))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='Price range', data=df, palette="viridis")
plt.title("Count of Restaurants by Price Range")

plt.subplot(1, 2, 2)
sns.boxplot(x='Price range', y='Average Cost for two', data=df[df['Average Cost for two'] < 5000])
plt.title("Average Cost for two by Price Range (Cost < 5000)")
plt.yscale('log')

plt.tight_layout()
plt.savefig(REPORTS_DIR / "02_price_cost_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Saved: reports/02_price_cost_analysis.png")

# ============================================
# 3. CITY DISTRIBUTION
# ============================================
print("\n" + "─" * 70)
print("3. CITY DISTRIBUTION (Top 15)")
print("─" * 70)

top_cities = df['City'].value_counts().head(15)
print(top_cities)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_cities.values, y=top_cities.index, palette="rocket")
plt.title("Top 15 Cities by Number of Restaurants")
plt.xlabel("Number of Restaurants")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "03_top_cities.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Saved: reports/03_top_cities.png")

# ============================================
# 4. ONLINE DELIVERY & TABLE BOOKING
# ============================================
print("\n" + "─" * 70)
print("4. ONLINE DELIVERY & TABLE BOOKING")
print("─" * 70)

print("\nHas Online delivery:")
print(df['Has Online delivery'].value_counts(normalize=True).round(3)*100)

print("\nHas Table booking:")
print(df['Has Table booking'].value_counts(normalize=True).round(3)*100)

# Rating by delivery & booking
print("\nAverage Rating by Online Delivery:")
print(df.groupby('Has Online delivery')['Aggregate rating'].mean().round(3))

print("\nAverage Rating by Table Booking:")
print(df.groupby('Has Table booking')['Aggregate rating'].mean().round(3))

# ============================================
# 5. VOTES ANALYSIS
# ============================================
print("\n" + "─" * 70)
print("5. VOTES ANALYSIS")
print("─" * 70)

print(df['Votes'].describe())
print(f"\nRestaurants with 0 votes: {(df['Votes'] == 0).sum()}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df[df['Votes'] < 500]['Votes'], bins=40, color="teal")
plt.title("Votes Distribution (Votes < 500)")

plt.subplot(1, 2, 2)
sns.scatterplot(x='Votes', y='Aggregate rating', data=df[df['Votes'] < 2000], alpha=0.4)
plt.title("Votes vs Aggregate Rating")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "04_votes_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Saved: reports/04_votes_analysis.png")

# ============================================
# 6. CUISINES OVERVIEW
# ============================================
print("\n" + "─" * 70)
print("6. CUISINES OVERVIEW")
print("─" * 70)

# Count how many cuisines per restaurant
df['Cuisine_Count'] = df['Cuisines'].apply(lambda x: len(str(x).split(',')))
print("Number of cuisines per restaurant:")
print(df['Cuisine_Count'].value_counts().sort_index().head(10))

print(f"\nAverage cuisines per restaurant: {df['Cuisine_Count'].mean():.2f}")
print(f"Max cuisines in one restaurant : {df['Cuisine_Count'].max()}")

# ============================================
# 7. CORRELATION HEATMAP (Numerical)
# ============================================
print("\n" + "─" * 70)
print("7. NUMERICAL CORRELATION")
print("─" * 70)

num_cols = ['Average Cost for two', 'Price range', 'Aggregate rating', 
            'Votes', 'Has Table booking', 'Has Online delivery', 
            'Is delivering now', 'Has_Valid_Coordinates']

corr = df[num_cols].corr()
print(corr['Aggregate rating'].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Heatmap - Numerical Features")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Saved: reports/05_correlation_heatmap.png")

# ============================================
# 8. RATING BY PRICE RANGE
# ============================================
print("\n" + "─" * 70)
print("8. RATING BY PRICE RANGE")
print("─" * 70)

print(df.groupby('Price range')['Aggregate rating'].agg(['mean', 'median', 'count']).round(3))

plt.figure(figsize=(8, 5))
sns.boxplot(x='Price range', y='Aggregate rating', data=df)
plt.title("Aggregate Rating by Price Range")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "06_rating_by_price.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Saved: reports/06_rating_by_price.png")

print("\n" + "=" * 70)
print("PHASE 5 COMPLETED - Exploratory Data Analysis Finished")
print("=" * 70)
print(f"\nAll plots saved in: {REPORTS_DIR}")