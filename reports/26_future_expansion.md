# Phase 26: Future Expansion Points

This system was architected so that new prediction capabilities can be added
by following the same pattern already established for the four current tasks,
without modifying existing code. This document describes how.

## The Established Pattern

Every current task follows the same five-layer shape:
Raw feature input
↓
src/inference/predict.py → one function per task, validates + loads a bundle
↓
models/bundle.joblib → model + preprocessing + metadata, versioned
↓
models/model_registry.json → central index of what exists and its status
↓
api/routes/*.py → one FastAPI route file per task, thin wrapper

Adding a new capability means adding one new function, one new bundle, one
registry entry, and one route file — never touching the other three tasks.

## How Each Future System Would Slot In

| Future System | Target Column (likely) | Task Type | Reuses From Existing Pipeline |
|---|---|---|---|
| **Demand Prediction** | Votes (as a time-series proxy) or a new demand signal | Regression / time-series | Same numerical/categorical preprocessing pattern as rating prediction (Phase 10); City_Frequency and Locality_Frequency features directly reusable |
| **Revenue Prediction** | Would require a new revenue column not in this dataset | Regression | Same feature engineering functions (Phase 7) apply directly to any new numeric target |
| **Sales Forecasting** | Would require order/transaction data not in this dataset | Time-series regression | Location Intelligence's city/locality aggregation logic (Phase 16) is directly reusable for demand-by-region forecasting |
| **Customer Segmentation** | No target — clustering | Unsupervised | Recommendation system's TF-IDF content vectors (Phase 14) could seed segment features; would need customer-level data not present here |
| **Customer Churn Prediction** | Would require customer return/visit data not in this dataset | Classification | Same leakage-detection discipline from Phase 4 is critical here — churn labels are especially prone to look-ahead leakage |
| **Sentiment Analysis** | Would require review text not in this dataset | NLP classification | TF-IDF infrastructure already built (Phase 14) is a direct starting point once review text is available |
| **Restaurant Success Prediction** | Composite of rating + votes + longevity | Classification/regression | Directly extends rating prediction (Phase 11-13); same feature set, same leakage discipline around Votes |
| **Price Optimization** | Average Cost for two (as an outcome, not input) | Regression | Would need to invert the current setup — Average Cost for two is currently an *input* feature for rating; here it becomes the *target*, with rating/location as inputs instead |

## What Data Would Be Needed (Honest Gap Assessment)

The current dataset supports Rating Prediction, Cuisine Classification,
Recommendation, and Location Intelligence well. Every future system listed
above requires data this project does not have:

- **Time-series data** (order history, visit timestamps) — needed for Demand/Sales Forecasting
- **Transaction/revenue data** — needed for Revenue Prediction, Price Optimization
- **Customer-level identifiers and repeat-visit history** — needed for Segmentation, Churn
- **Review text** — needed for Sentiment Analysis, and would also likely improve Cuisine Classification's current ~43% accuracy ceiling (see Phase 19)

This is stated explicitly so the "extension points" claim is honest: the
*architecture* is ready, but the *data* for these systems does not yet exist
in this project.

## Registry-Level Extension Points (Already Declared)

`models/model_registry.json` (Phase 20) already lists these eight systems
under `future_extension_points`, so any code consuming the registry can see
what's planned without needing this document:

```json
"future_extension_points": [
    "demand_prediction", "revenue_prediction", "sales_forecasting",
    "customer_segmentation", "customer_churn_prediction",
    "sentiment_analysis", "restaurant_success_prediction",
    "price_optimization"
]
```

## Concrete Steps to Add a New Task (Worked Example: Restaurant Success Prediction)

If review/longevity data became available, adding "Restaurant Success
Prediction" would follow exactly this sequence — no existing file changes:

1. Define target (e.g., `Is_Successful` = rating ≥ 4.0 AND votes ≥ 100)
2. Run through Phases 2-10 (quality check → cleaning → leakage check → EDA →
   feature engineering → split → preprocessing) exactly as Task 1 did
3. Train/compare models (Phase 11 pattern), tune (Phase 12 pattern)
4. Add `predict_success()` to `src/inference/predict.py`, following the
   same shape as `predict_rating()` — load bundle, validate input, return
   structured output
5. Add `success_prediction` entry to `model_registry.json`
6. Create `api/routes/success.py`, mirroring `api/routes/rating.py`
7. Register the new router in `api/main.py`
8. Add test cases to `tests/test_api.py` following the existing pattern

No modification to rating prediction, cuisine classification, recommendation,
or location intelligence code would be required at any step.

## Monitoring-Readiness (Not Implemented, But Structurally Supported)

The versioned bundle + registry pattern (Phase 20) already supports future
monitoring without redesign:
- Each bundle carries `training_metadata` (row counts, random state, creation
  timestamp) — a natural place to compare against live prediction distributions
- Model versions are already tracked in the registry, so A/B comparison
  between versions (e.g., `rating_model_bundle_v1.0` vs a future `v1.1`) is
  a registry lookup away, not a rebuild