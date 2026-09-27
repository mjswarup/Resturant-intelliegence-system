"""
PHASE 14: TASK 2 - RESTAURANT RECOMMENDATION SYSTEM
Content-Based Filtering
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config.config import PROCESSED_DATA_DIR, MODELS_DIR
import joblib

print("=" * 70)
print("PHASE 14: RESTAURANT RECOMMENDATION SYSTEM")
print("=" * 70)

# ============================================
# 1. LOAD DATA
# ============================================
df = pd.read_csv(PROCESSED_DATA_DIR / "featured_restaurants.csv")
print(f"Loaded data: {df.shape}")

# Keep only useful columns for recommendation
rec_df = df[[
    'Restaurant ID', 'Restaurant Name', 'City', 'Locality',
    'Cuisines', 'Primary_Cuisine', 'Average Cost for two',
    'Currency', 'Price range', 'Aggregate rating', 'Votes',
    'Has Table booking', 'Has Online delivery'
]].copy()

# Remove duplicates by Restaurant Name + City (keep highest rated)
rec_df = rec_df.sort_values('Aggregate rating', ascending=False)
rec_df = rec_df.drop_duplicates(subset=['Restaurant Name', 'City'], keep='first')
rec_df = rec_df.reset_index(drop=True)

print(f"After removing duplicates: {rec_df.shape}")

# ============================================
# 2. CREATE CONTENT FEATURES
# ============================================
print("\n" + "─" * 70)
print("2. CREATING CONTENT FEATURES")
print("─" * 70)

# Combine text features for TF-IDF
rec_df['content'] = (
    rec_df['Cuisines'].fillna('') + ' ' +
    rec_df['City'].fillna('') + ' ' +
    rec_df['Locality'].fillna('') + ' ' +
    rec_df['Primary_Cuisine'].fillna('')
)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(rec_df['content'])

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Cosine similarity matrix shape: {cosine_sim.shape}")

# ============================================
# 3. RECOMMENDATION FUNCTION
# ============================================
print("\n" + "─" * 70)
print("3. BUILDING RECOMMENDATION FUNCTION")
print("─" * 70)

def recommend_restaurants(
    city: str = None,
    cuisine: str = None,
    min_rating: float = 0.0,
    max_cost: float = None,
    price_range: int = None,
    online_delivery: bool = None,
    table_booking: bool = None,
    top_n: int = 10
):
    """
    Content-based restaurant recommendation.
    
    Returns top_n restaurants matching the user preferences.
    """
    
    results = rec_df.copy()
    
    # Apply filters
    if city:
        results = results[results['City'].str.contains(city, case=False, na=False)]
    
    if cuisine:
        results = results[results['Cuisines'].str.contains(cuisine, case=False, na=False)]
    
    if min_rating > 0:
        results = results[results['Aggregate rating'] >= min_rating]
    
    if max_cost is not None:
        results = results[results['Average Cost for two'] <= max_cost]
    
    if price_range is not None:
        results = results[results['Price range'] == price_range]
    
    if online_delivery is not None:
        results = results[results['Has Online delivery'] == int(online_delivery)]
    
    if table_booking is not None:
        results = results[results['Has Table booking'] == int(table_booking)]
    
    # Sort by rating and votes
    results = results.sort_values(
        by=['Aggregate rating', 'Votes'],
        ascending=[False, False]
    )
    
    # Select top N
    results = results.head(top_n)
    
    # Format output
    output = []
    for rank, (_, row) in enumerate(results.iterrows(), 1):
        output.append({
            "Rank": rank,
            "Restaurant": row['Restaurant Name'],
            "Cuisine": row['Cuisines'],
            "Rating": row['Aggregate rating'],
            "Votes": int(row['Votes']),
            "Price Range": int(row['Price range']),
            "Average Cost for two": row['Average Cost for two'],
            "Currency": row['Currency'],
            "City": row['City'],
            "Locality": row['Locality'],
            "Online Delivery": "Yes" if row['Has Online delivery'] == 1 else "No",
            "Table Booking": "Yes" if row['Has Table booking'] == 1 else "No",
            "Reason": f"High rating ({row['Aggregate rating']}) + matching preferences"
        })
    
    return pd.DataFrame(output)


# ============================================
# 4. TEST RECOMMENDATIONS
# ============================================
print("\n" + "─" * 70)
print("4. TESTING RECOMMENDATION ENGINE")
print("─" * 70)

print("\n▶ Test 1: Best North Indian restaurants in New Delhi (Rating >= 4.0)")
recs1 = recommend_restaurants(
    city="New Delhi",
    cuisine="North Indian",
    min_rating=4.0,
    top_n=5
)
print(recs1[['Rank', 'Restaurant', 'Cuisine', 'Rating', 'Average Cost for two']].to_string(index=False))

print("\n▶ Test 2: Cheap restaurants with Online Delivery in Gurgaon")
recs2 = recommend_restaurants(
    city="Gurgaon",
    max_cost=500,
    online_delivery=True,
    min_rating=3.5,
    top_n=5
)
print(recs2[['Rank', 'Restaurant', 'Cuisine', 'Rating', 'Average Cost for two']].to_string(index=False))

print("\n▶ Test 3: High-end restaurants with Table Booking")
recs3 = recommend_restaurants(
    price_range=4,
    table_booking=True,
    min_rating=4.0,
    top_n=5
)
print(recs3[['Rank', 'Restaurant', 'City', 'Rating', 'Average Cost for two']].to_string(index=False))

# ============================================
# 5. SAVE RECOMMENDATION ARTIFACTS
# ============================================
print("\n" + "─" * 70)
print("5. SAVING RECOMMENDATION ARTIFACTS")
print("─" * 70)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Save the recommendation dataframe and vectorizer
joblib.dump(rec_df, MODELS_DIR / "recommendation_data.joblib")
joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
joblib.dump(cosine_sim, MODELS_DIR / "cosine_similarity.joblib")

print("✅ Saved: models/recommendation_data.joblib")
print("✅ Saved: models/tfidf_vectorizer.joblib")
print("✅ Saved: models/cosine_similarity.joblib")

print("\n" + "=" * 70)
print("PHASE 14 COMPLETED - Recommendation System Ready")
print("=" * 70)