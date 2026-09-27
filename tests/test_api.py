"""
PHASE 23: API TESTING
Restaurant Intelligence System

Covers: valid requests, missing values, invalid types, unknown categories,
invalid numerical ranges, empty requests, malformed requests.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ============================================
# HELPERS
# ============================================

def print_result(name, response, expected_status):
    status_ok = "✅" if response.status_code == expected_status else "❌"
    print(f"{status_ok} {name}")
    print(f"   Expected: {expected_status} | Got: {response.status_code}")
    if response.status_code != expected_status:
        print(f"   Body: {response.text[:300]}")
    print()


# ============================================
# 1. SYSTEM ENDPOINTS
# ============================================

def test_system_endpoints():
    print("=" * 70)
    print("1. SYSTEM ENDPOINTS")
    print("=" * 70)

    r = client.get("/health")
    print_result("GET /health", r, 200)
    assert r.json()["status"] == "ok"

    r = client.get("/model-info")
    print_result("GET /model-info", r, 200)
    assert "rating_prediction" in r.json()["models"]

    r = client.get("/")
    print_result("GET / (root)", r, 200)


# ============================================
# 2. RATING PREDICTION - VALID
# ============================================

def test_rating_valid():
    print("=" * 70)
    print("2. RATING PREDICTION - VALID REQUEST")
    print("=" * 70)

    payload = {
        "City": "New Delhi",
        "Locality": "Connaught Place",
        "Currency": "Indian Rupees(Rs.)",
        "Primary_Cuisine": "North Indian",
        "Country Code": 1,
        "Longitude": 77.22,
        "Latitude": 28.63,
        "Average Cost for two": 800,
        "Price range": 2,
        "Has Table booking": 1,
        "Has Online delivery": 1,
        "Is delivering now": 0,
        "Cuisine_Count": 2,
        "Name_Length": 15,
        "Address_Length": 50,
        "City_Frequency": 0.573,
        "Locality_Frequency": 0.013
    }
    r = client.post("/predict/rating", json=payload)
    print_result("POST /predict/rating (valid)", r, 200)
    if r.status_code == 200:
        print(f"   Predicted rating: {r.json()['predicted_rating']}")
        print()
    assert r.status_code == 200
    assert 0.0 <= r.json()["predicted_rating"] <= 5.0


# ============================================
# 3. RATING PREDICTION - MISSING / EMPTY / MALFORMED
# ============================================

def test_rating_edge_cases():
    print("=" * 70)
    print("3. RATING PREDICTION - EDGE CASES")
    print("=" * 70)

    # Empty request body - all fields optional, should still return a prediction
    # (defaults kick in inside predict_rating())
    r = client.post("/predict/rating", json={})
    print_result("POST /predict/rating (empty body)", r, 200)
    if r.status_code == 200:
        print(f"   Predicted rating (all defaults): {r.json()['predicted_rating']}")
        print(f"   Warnings count: {len(r.json()['warnings'])}")
        print()

    # Invalid data type - Price range as a string that can't convert
    r = client.post("/predict/rating", json={"Price range": "very expensive"})
    print_result("POST /predict/rating (invalid type for Price range)", r, 422)

    # Invalid range - Price range out of bounds (schema enforces 1-4)
    r = client.post("/predict/rating", json={"Price range": 9})
    print_result("POST /predict/rating (Price range out of bounds)", r, 422)

    # Malformed JSON body
    r = client.post(
        "/predict/rating",
        data="{not valid json,,,",
        headers={"Content-Type": "application/json"}
    )
    print_result("POST /predict/rating (malformed JSON)", r, 422)

    # Unknown/unexpected extra field - schema allows extras, should still succeed
    r = client.post("/predict/rating", json={"City": "New Delhi", "Some_Random_Field": "xyz"})
    print_result("POST /predict/rating (unknown extra field, should be tolerated)", r, 200)


# ============================================
# 4. RECOMMENDATION - VALID
# ============================================

def test_recommend_valid():
    print("=" * 70)
    print("4. RECOMMENDATION - VALID REQUEST")
    print("=" * 70)

    payload = {
        "city": "New Delhi",
        "cuisine": "North Indian",
        "min_rating": 4.0,
        "top_n": 5
    }
    r = client.post("/recommend", json=payload)
    print_result("POST /recommend (valid)", r, 200)
    if r.status_code == 200:
        print(f"   Results returned: {r.json()['count']}")
        print()
    assert r.status_code == 200


# ============================================
# 5. RECOMMENDATION - EDGE CASES
# ============================================

def test_recommend_edge_cases():
    print("=" * 70)
    print("5. RECOMMENDATION - EDGE CASES")
    print("=" * 70)

    # Empty body - all fields have defaults, should return top-rated overall
    r = client.post("/recommend", json={})
    print_result("POST /recommend (empty body)", r, 200)
    if r.status_code == 200:
        print(f"   Results returned (no filters): {r.json()['count']}")
        print()

    # Invalid range - min_rating above 5.0
    r = client.post("/recommend", json={"min_rating": 15.0})
    print_result("POST /recommend (min_rating out of range)", r, 422)

    # Invalid type - top_n as a string
    r = client.post("/recommend", json={"top_n": "many"})
    print_result("POST /recommend (top_n wrong type)", r, 422)

    # Invalid price_range category (schema only allows 1-4)
    r = client.post("/recommend", json={"price_range": 7})
    print_result("POST /recommend (price_range out of bounds)", r, 422)

    # City that doesn't exist - should succeed with 0 results, not error
    r = client.post("/recommend", json={"city": "Atlantis"})
    print_result("POST /recommend (nonexistent city, should return 0 results)", r, 200)
    if r.status_code == 200:
        print(f"   Results for nonexistent city: {r.json()['count']}")
        print()

    # top_n over the allowed limit
    r = client.post("/recommend", json={"top_n": 500})
    print_result("POST /recommend (top_n exceeds max)", r, 422)


# ============================================
# 6. CUISINE CLASSIFICATION - VALID
# ============================================

def test_cuisine_valid():
    print("=" * 70)
    print("6. CUISINE CLASSIFICATION - VALID REQUEST")
    print("=" * 70)

    payload = {
        "City": "New Delhi",
        "Country Code": 1,
        "Average Cost for two": 800,
        "Price range": 2,
        "Has Table booking": 1,
        "Has Online delivery": 1,
        "Aggregate rating": 4.2,
        "Votes": 150,
        "Cuisine_Count": 2,
        "Name_Length": 15,
        "City_Frequency": 0.573,
        "Locality_Frequency": 0.013
    }
    r = client.post("/predict/cuisine", json=payload)
    print_result("POST /predict/cuisine (valid)", r, 200)
    if r.status_code == 200:
        print(f"   Predicted cuisine: {r.json()['predicted_cuisine']} ({r.json()['confidence']})")
        print()
    assert r.status_code == 200


# ============================================
# 7. CUISINE CLASSIFICATION - EDGE CASES
# ============================================

def test_cuisine_edge_cases():
    print("=" * 70)
    print("7. CUISINE CLASSIFICATION - EDGE CASES")
    print("=" * 70)

    # Empty body
    r = client.post("/predict/cuisine", json={})
    print_result("POST /predict/cuisine (empty body)", r, 200)

    # Invalid rating range
    r = client.post("/predict/cuisine", json={"Aggregate rating": 12.0})
    print_result("POST /predict/cuisine (Aggregate rating out of bounds)", r, 422)

    # Negative votes
    r = client.post("/predict/cuisine", json={"Votes": -50})
    print_result("POST /predict/cuisine (negative Votes)", r, 422)

    # Wrong type for Price range
    r = client.post("/predict/cuisine", json={"Price range": "high"})
    print_result("POST /predict/cuisine (wrong type)", r, 422)


# ============================================
# 8. LOCATION INTELLIGENCE - VALID
# ============================================

def test_location_valid():
    print("=" * 70)
    print("8. LOCATION INTELLIGENCE - VALID REQUEST")
    print("=" * 70)

    r = client.get("/location/analysis", params={"city": "New Delhi"})
    print_result("GET /location/analysis?city=New Delhi", r, 200)
    if r.status_code == 200:
        print(f"   Restaurant count: {r.json()['restaurant_count']}")
        print()
    assert r.status_code == 200

    r = client.get("/location/analysis", params={"city": "New Delhi", "locality": "Connaught Place"})
    print_result("GET /location/analysis (with locality)", r, 200)


# ============================================
# 9. LOCATION INTELLIGENCE - EDGE CASES
# ============================================

def test_location_edge_cases():
    print("=" * 70)
    print("9. LOCATION INTELLIGENCE - EDGE CASES")
    print("=" * 70)

    # Missing required query param entirely
    r = client.get("/location/analysis")
    print_result("GET /location/analysis (missing required 'city')", r, 422)

    # Empty city string
    r = client.get("/location/analysis", params={"city": ""})
    print_result("GET /location/analysis (empty city)", r, 422)

    # Nonexistent city - should be a graceful 200 with found=false, not an error
    r = client.get("/location/analysis", params={"city": "Atlantis"})
    print_result("GET /location/analysis (nonexistent city, graceful)", r, 200)
    if r.status_code == 200:
        print(f"   found: {r.json()['found']}")
        print()

    # Nonexistent locality within a valid city
    r = client.get("/location/analysis", params={"city": "New Delhi", "locality": "Nowhereville"})
    print_result("GET /location/analysis (nonexistent locality)", r, 200)


# ============================================
# RUN ALL TESTS
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 23: API TESTING - FULL SUITE")
    print("=" * 70 + "\n")

    test_system_endpoints()
    test_rating_valid()
    test_rating_edge_cases()
    test_recommend_valid()
    test_recommend_edge_cases()
    test_cuisine_valid()
    test_cuisine_edge_cases()
    test_location_valid()
    test_location_edge_cases()

    print("=" * 70)
    print("PHASE 23 COMPLETED - All test cases executed")
    print("=" * 70)