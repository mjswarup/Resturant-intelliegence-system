\# Model Card: Restaurant Intelligence System



\## Model 1: Rating Prediction



\*\*Model type:\*\* Random Forest Regressor

\*\*Version:\*\* v1.0

\*\*Intended use:\*\* Estimate a restaurant's likely aggregate rating from

observable, pre-rating features (location, price, service offerings).



\*\*Training data:\*\* 7,640 restaurants (80% split), Zomato-style dataset,

heavily India-concentrated (New Delhi/Gurgaon/Noida ≈ 68% of training rows).



\*\*Performance:\*\*

\- Test MAE: 0.77 (on a 0–5 rating scale)

\- Test R²: 0.50

\- Performs notably worse on unrated restaurants (MAE 1.39) than rated ones (MAE 0.60)

\- Performs better at higher price ranges (R²=0.48) than lower ones (R²=0.39)



\*\*Known exclusions:\*\* The `Votes` feature was deliberately excluded after

Phase 12 identified it as a leakage risk — including it inflated R² to 0.96,

but Votes is unavailable for new/unrated restaurants, making that version

unusable for the model's primary real-world purpose.



\*\*Limitations:\*\*

\- Should not be used to evaluate brand-new or budget-tier restaurants with confidence

\- Does not incorporate review text, service quality signals, or time-series trends

\- Geographic bias means performance on non-Indian cities is based on small samples



\*\*Ethical considerations:\*\* This model reflects patterns in historical rating

data, which may encode existing biases (e.g., toward certain price tiers or

cities). Predictions should inform, not replace, human judgment — particularly

for decisions with business or livelihood impact (e.g., loan/investment decisions

based on predicted restaurant success).



\---



\## Model 2: Cuisine Classification



\*\*Model type:\*\* XGBoost Classifier

\*\*Version:\*\* v1.0

\*\*Intended use:\*\* Predict a restaurant's primary (first-listed) cuisine from

non-text features.



\*\*Training data:\*\* 8,898 restaurants (after filtering to cuisines with ≥50 samples), 26 classes.



\*\*Performance:\*\*

\- Accuracy: 42.9%

\- Weighted F1: 0.386

\- Strong class imbalance: North Indian is \~34% of the filtered dataset



\*\*Limitations:\*\*

\- Moderate accuracy reflects a genuinely hard problem given available features —

&#x20; cuisine is only weakly implied by cost/location/price range

\- Rare cuisines (<50 samples) were excluded entirely and cannot be predicted

\- Would likely benefit substantially from text features (restaurant name, description) not present in this dataset



\---



\## Model 3: Recommendation System



\*\*Approach:\*\* Content-based filtering — TF-IDF over cuisine/city/locality text,

combined with explicit business-rule filters (cost, price range, delivery, booking),

ranked by rating and vote count.



\*\*No formal accuracy metric\*\* — recommendation systems of this type are

evaluated qualitatively. Validated via representative test queries (Phase 14)

producing sensible, explainable results.



\*\*Limitations:\*\*

\- No personalization or collaborative filtering (no user interaction history exists in this dataset)

\- Recommendations reflect historical rating patterns, inheriting the same geographic bias as other models



\---



\## Model 4: Location Intelligence



\*\*Approach:\*\* Descriptive statistical aggregation (no trained model).

\*\*Limitations:\*\* Correlational only — see `reports/18\_business\_intelligence.md`

for explicit correlation vs. causation framing. No causal claims are supported

by this analysis.

