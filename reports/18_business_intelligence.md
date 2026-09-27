# Phase 18: Business Intelligence
Restaurant Intelligence System — Insights derived from Phases 5, 16, and 17
Translating everything from Phases 5, 16, and 17 into structured business insights. As required by the spec, each insight is tagged by type — Correlation, Prediction, or Causation flagged as unsupported — so nothing gets overstated.

1. What Drives Higher Ratings
Insight	Type	Evidence
Restaurants with online delivery average 3.25 vs 2.47 without	Correlation	Phase 5 groupby
Restaurants with table booking average 3.44 vs 2.56 without	Correlation	Phase 5 groupby
Higher price range → higher average rating (1: 2.00, 4: 3.82)	Correlation	Phase 5 groupby
Price range is the single strongest built-in predictor; City_Frequency is the strongest by real-world impact	Prediction driver	Phase 17

Not causation. Offering delivery or booking doesn't make a restaurant better-rated — it's far more likely that better-run, more established restaurants are the ones that can afford to offer these services in the first place. Same logic for price range: expensive restaurants aren't rated higher because they're expensive; both price and rating likely trace back to a common cause (quality of operation, chef, ambience) we don't have in this data.

2. Geography Matters More Than Almost Anything Else
Insight	Type	Evidence
City_Frequency (a proxy for market saturation) is by far the top feature by permutation importance	Prediction driver	Phase 17
Highest-rated cities (London 4.54, Orlando 4.48, Tampa Bay 4.41) are not the highest-volume cities	Correlation	Phase 16
Most expensive cities (Pune, Jaipur, Kolkata) are also not the highest-volume cities	Correlation	Phase 16
Connaught Place, New Delhi: highest restaurant count (122) and a strong rating (3.69)	Correlation	Phase 16
Hauz Khas Village: small restaurant count (24) but highest average rating (3.85) among New Delhi localities	Correlation	Phase 16

Business read: rating quality, restaurant density, and cost don't move together in any simple way. A city or locality being a "high-volume restaurant hub" says nothing about whether it's a "high-quality" or "expensive" one — these are three separate axes, and a business decision (e.g., where to open a new restaurant) needs to look at all three independently rather than assuming one predicts another.

3. The Rating Prediction Model Is Honest About Its Limits
Insight	Type	Evidence
Votes-included model looked excellent (R²=0.96) but was mostly measuring vote-count leakage, not real signal	Prediction — flagged as unreliable	Phase 12 impact check
Realistic (no-Votes) model achieves R²≈0.50 — moderate, honest predictive power	Prediction	Phase 12/13
Model performs much worse on "Not Rated" restaurants (MAE 1.39) than on genuinely rated ones (MAE 0.60)	Prediction limitation	Phase 13
Model performs better on higher price ranges (R²=0.48 at price range 4) than lower ones (R²=0.39 at price range 1)	Prediction limitation	Phase 13

Business read: the model is a reasonable planning tool for restaurants that already have some track record, but it should not be trusted to predict ratings for brand-new, unrated, budget-tier restaurants — that's exactly where it's weakest.

4. Cuisine Patterns
Insight	Type	Evidence
North Indian dominates the market (2,992 of ~8,900 filtered restaurants)	Correlation (descriptive)	Phase 15
Cuisine classification only reaches ~43% accuracy across 26 classes	Prediction limitation	Phase 15
Primary_Cuisine has low importance in rating prediction — far below geography and price	Prediction driver	Phase 17

Business read: cuisine type alone is a weak signal for both "what rating will this get" and "what cuisine is this" (the latter being genuinely hard because non-cuisine features like city and cost only loosely imply cuisine). This suggests cuisine-based menu/marketing strategy should lean on richer text data (dish names, reviews) if it were available — not present in this dataset.

5. Delivery & Booking Adoption Is Geographically Skewed
Insight	Type	Evidence
Chennai (65%), Abu Dhabi/Sharjah (55%), Ahmedabad (52%) lead in online delivery adoption	Correlation	Phase 16
The dominant high-volume cities (Gurgaon 40%, Noida 37%) trail behind	Correlation	Phase 16

Business read: delivery adoption isn't simply "bigger market = more delivery" — smaller or mid-size markets can out-adopt the biggest ones. Worth investigating (outside this dataset) whether this reflects platform rollout timing rather than restaurant behavior.

Summary: Correlation vs Prediction vs Causation
Reliable correlations (repeatable, consistent across phases): price range ↔ rating, delivery/booking ↔ rating, geography ↔ rating.
Reliable predictions (validated on held-out test data): the realistic rating model (R²≈0.50), the cuisine classifier (43% on 26 classes) — both moderate, both honestly reported rather than inflated.
No causal claims made or supported anywhere in this analysis. The dataset is observational; nothing here was randomized or controlled, so statements like "adding online delivery will raise your rating" are explicitly not supported by this work and shouldn't be presented as such in the final report.