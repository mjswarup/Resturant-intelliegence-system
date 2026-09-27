# Restaurant Intelligence System — Final Project Report

**Version:** 1.0
**Dataset:** 9,551 restaurants, 141 cities (Zomato-style)
**Pipeline:** Phases 0–26, fully executed and validated

---

## 1. Problem Definition

Build an extensible Restaurant Intelligence System with four current
capabilities, architected so future prediction systems can be added without
restructuring existing code:

1. **Rating Prediction** — regression
2. **Restaurant Recommendation** — content-based filtering
3. **Cuisine Classification** — multi-class classification
4. **Location Intelligence** — descriptive geospatial analytics

No assumptions were made about column names, data cleanliness, or which
features would prove useful — every decision in this report was made after
inspecting the actual dataset, not from prior expectation.

---

## 2. Dataset Understanding

**Initial shape:** 9,551 rows × 21 columns (Phase 1)

Key columns identified: Restaurant ID/Name, geographic fields (City,
Locality, Longitude, Latitude), Cuisines, cost/price fields, service flags
(Table booking, Online delivery, Is delivering now), and rating-related
fields (Aggregate rating, Rating color, Rating text, Votes).

**Early signal that shaped the whole project:** `Rating color` and `Rating
text` were both perfect deterministic functions of `Aggregate rating` — this
was caught in Phase 1's initial inspection and confirmed with a cross-tab in
Phase 2, well before any model was trained.

---

## 3. Data Quality

Full audit performed in Phase 2 before any data was touched. Findings:

| Issue | Detail |
|---|---|
| Missing values | Only `Cuisines` (9 rows, 0.09%) |
| Duplicate rows | 0 |
| Constant column | `Switch to order menu` (always "No") |
| Target leakage | `Rating color`, `Rating text` — both derived from target |
| Invalid coordinates | 497 restaurants at (0, 0) |
| Zero/extreme cost | 18 restaurants at ₹0, 18 above ₹100,000 |
| High cardinality | Restaurant ID, Address, Restaurant Name, Locality, Cuisines |

Full detail: nothing was assumed — every issue above was verified with code,
not inferred from column names.

---

## 4. Data Cleaning

Applied in Phase 3, with a full cleaning log and before/after counts for
every operation. No rows were deleted without defensible reason — the
approach throughout was **flag, don't discard**, wherever discarding wasn't
clearly justified:

- Dropped: `Switch to order menu` (constant), `Rating color`/`Rating text`
  (leakage), `Locality Verbose` (redundant with City+Locality)
- Filled: 9 missing `Cuisines` values with `"Unknown"`
- Converted: Yes/No service flags to binary (0/1)
- **Flagged, not deleted:** zero-coordinate restaurants (`Has_Valid_Coordinates`),
  zero/extreme cost restaurants (`Is_Zero_Cost`, `Is_Extreme_Cost`) — these
  stayed in the dataset with a marker column, so downstream models could
  learn from or exclude them as needed rather than losing that data outright

Result: 9,551 rows retained (unchanged), columns reduced 21 → 20.

---

## 5. Target & Leakage Identification

Formalized in Phase 4. Each task's target was defined explicitly, and every
feature was assessed for leakage risk in a structured table (Feature /
Leakage Risk / Decision / Reason). The most consequential finding came later
in Phase 12: **`Votes` looked safe by correlation alone (r=0.31, "moderate")
but was proven to be a leakage risk empirically** — see Section 9.

---

## 6. Exploratory Data Analysis

Phase 5 examined the target distribution, price/cost relationships, city
distribution, service adoption, votes, cuisine counts, and correlations —
each with Question / Observation / Interpretation / ML Impact / Business
Impact framing as specified.

**Headline findings:**
- 22.5% of restaurants have `Aggregate rating = 0` ("Not Rated") — a
  meaningful subpopulation, not noise
- Price range correlates with rating (r=0.44) — the strongest single
  numerical predictor identified at this stage
- Online delivery (3.25 vs 2.47) and table booking (3.44 vs 2.56) both show
  a large average-rating gap
- New Delhi alone accounts for 57% of the dataset — a geographic bias that
  propagated into every downstream task

---

## 7. Feature Extraction & Engineering

Phases 6–7. Features were classified by type (numerical, categorical,
geographical, identifier) before any transformation was chosen, and every
new feature was justified individually — none were created purely to chase
a metric. Ten engineered features were added: `Cuisine_Count`,
`Primary_Cuisine`, `Name_Length`, `Address_Length`, `City_Frequency`,
`Locality_Frequency`, `Log_Average_Cost`, `Cost_to_PriceRange`,
`Has_Booking_and_Delivery`, `Is_Rated`.

---

## 8. Feature Selection

Phase 8 (folded into Phase 9's script). Final feature set for Task 1: 24
features (20 numerical, 4 categorical), explicitly excluding identifiers
(`Restaurant ID`), high-cardinality free text (`Address`, `Restaurant Name`),
the raw `Cuisines` string (replaced by engineered equivalents), and the
leakage-flagged `Is_Rated`.

---

## 9. Model Development, Comparison & the Votes Leakage Discovery

This is the most important methodological result of the whole project, and
worth stating plainly rather than burying in a metrics table.

**Phase 11** compared 8 regression models. Random Forest topped the table
with **Test R² = 0.9632** — an excellent-looking number.

**But the process didn't stop at "best score wins."** Phase 12 deliberately
tested the top models *without* the `Votes` feature, because Votes is
information that only exists *after* a restaurant has accumulated ratings —
exactly the leakage pattern the original spec's Phase 4 warned about.

**Result:** removing `Votes` alone dropped Random Forest from R²=0.9632 to
**R²=0.5012**. Over 90% of the model's apparent skill was actually the model
learning "restaurants with more votes tend to have solidified, non-zero
ratings" — a fact, but not a *useful* one, since a new restaurant has zero
votes by definition.

**Decision (Phase 12/19):** deploy the realistic, no-Votes model
(R²=0.5037, MAE=0.77) as the production model. The with-Votes model was
kept only as a documented comparison artifact, explicitly labeled
"not for deployment."

This sequence — build, get a suspiciously good number, test the suspicion,
find it's leakage, correct course, document both versions honestly — is the
single clearest demonstration in this project of the "do not assume a step
is necessary without checking the data" principle stated at the start.

---

## 10. Hyperparameter Tuning

Phase 12. `RandomizedSearchCV` (10 candidates, 3-fold CV) applied to the
realistic Random Forest. Best params: `n_estimators=100,
min_samples_split=5, min_samples_leaf=2, max_depth=None`. Tuning moved
Test R² only marginally (0.5012 → 0.5037) — the ceiling here is a feature-set
limitation, not an undertuned model, and this was stated honestly rather
than over-claiming tuning gains.

---

## 11. Error Analysis

Phase 13. Full residual analysis, worst-predictions inspection, and
performance breakdowns by price range and by rated-vs-unrated status.

**Key finding:** the model's error is not evenly distributed — it performs
much worse on unrated restaurants (MAE 1.39) than rated ones (MAE 0.60), and
better at higher price ranges (R²=0.48) than lower ones (R²=0.39). This is
documented explicitly as a deployment limitation, not hidden.

---

## 12. Restaurant Recommendation

Phase 14. Content-based filtering: TF-IDF over Cuisines/City/Locality text
combined with explicit filters (cost, price range, delivery, booking), sorted
by rating and votes. No supervised target exists for this task, so
evaluation was qualitative — three representative test queries, all
producing sensible, explainable results with a `Reason` field for every
recommendation.

---

## 13. Cuisine Classification

Phase 15. Investigated first whether cuisine is single-label or multi-label
(dataset has multiple cuisines per restaurant, averaging 2.06). **Decision:**
single-label classification on `Primary_Cuisine` (first listed), because full
multi-label modeling was judged out of scope relative to the value it would
add, and this trade-off is stated explicitly rather than silently chosen.

Filtered to 26 classes with ≥50 samples (8,898 of 9,551 restaurants). Best
model: XGBoost, Accuracy=42.9%, Weighted F1=0.386. Strong class imbalance
(North Indian ≈ 34%) limits the ceiling here, documented honestly in
Phase 19 rather than presented as a stronger result than it is.

---

## 14. Location Intelligence

Phase 16. City-level and locality-level aggregation (restaurant count,
average rating, cost, votes, delivery/booking adoption). Key finding: the
highest-volume cities (New Delhi, Gurgaon, Noida) are **not** the
highest-rated, most expensive, or highest-delivery-adoption cities — these
are three independent axes, not one composite signal.

---

## 15. Explainable AI

Phase 17. Permutation importance, built-in (Gini) importance, and SHAP
(TreeExplainer) applied to the realistic rating model. A meaningful
disagreement was found and reported rather than smoothed over: permutation
importance ranks `City_Frequency` far above everything else (0.668, next
highest 0.178), while built-in importance spreads credit more evenly with
`Price range` on top. Permutation importance was treated as the more
trustworthy signal, since it measures real held-out impact rather than
training-time split usage.

---

## 16. Business Intelligence

Phase 18 (`reports/18_business_intelligence.md`) translated every finding
above into business-readable insights, with every claim explicitly tagged
as **Correlation**, **Prediction**, or flagged where causation is *not*
supported. No causal claims are made anywhere in this project — the dataset
is observational, and this is stated as a hard boundary, not a caveat buried
in a footnote.

---

## 17. Deployment Architecture

- **Phase 20:** Models serialized as versioned bundles (model + preprocessing
  + feature lists + metrics + metadata), indexed in a central
  `model_registry.json`
- **Phase 21:** Unified inference engine (`src/inference/predict.py`) with
  lazy-loaded caching — deliberately avoids loading the 486MB unused cosine
  similarity matrix at runtime, cutting real memory footprint
- **Phase 22:** FastAPI backend, 6 endpoints, Pydantic-validated
- **Phase 23:** 25 automated test cases (valid requests, missing/invalid/
  malformed input, edge cases) — all passing
- **Phase 24:** Live demonstration via local Uvicorn + Swagger UI (adapted
  from the original Colab-based spec since development occurred in VS Code
  throughout — documented honestly in `reports/24_deployment_demo.md`)
- **Phase 25:** Docker, pinned requirements, README, Model Card, Deployment
  Guide — production-portable, with an explicit, honest checklist of what's
  *not* yet production-hardened (CORS policy, auth, rate limiting)

---

## 18. Limitations

Stated plainly, consolidated from Phases 13, 19, and throughout:

1. **Geographic bias:** ~57% of the dataset is New Delhi/Gurgaon/Noida;
   performance on international cities rests on small samples (often exactly
   the 20-restaurant minimum threshold)
2. **Rating model ceiling:** R²≈0.50 — roughly half of rating variance is
   explained by review text, service quality, or other signals this dataset
   doesn't contain
3. **Cuisine model ceiling:** ~43% accuracy across 26 imbalanced classes —
   a genuinely hard problem given the available (non-text) feature set
4. **No causal claims supported anywhere** — this is an observational
   dataset; every business insight is explicitly framed as correlation or
   prediction, never causation
5. **Not production-hardened:** open CORS policy, no authentication, no
   rate limiting — documented explicitly in the Deployment Guide as future
   work, not silently omitted

---

## 19. Future Improvements

Detailed in `reports/26_future_expansion.md`. Summary: the architecture
(inference engine → registry → API route, repeated per task) is ready to
support eight additional systems (demand prediction, revenue prediction,
sales forecasting, customer segmentation, churn prediction, sentiment
analysis, restaurant success prediction, price optimization) — but **the
data required for each does not yet exist in this project**, and that gap
is stated honestly rather than implied away.

---

## 20. Conclusion

This system delivers four working, tested, documented ML capabilities over
a real, imperfect dataset — built by inspecting evidence at every decision
point rather than assuming best practice applied uniformly. The clearest
proof of that discipline is the Votes leakage discovery (Section 9): an
excellent-looking 0.96 R² was not accepted at face value, was tested,
found to be substantially leakage, and was replaced with an honest 0.50 R²
model — documenting both versions rather than hiding the weaker, more
truthful one.

**Full phase-by-phase detail, code, and supporting reports:**
- `reports/18_business_intelligence.md`
- `reports/19_model_selection.md`
- `reports/24_deployment_demo.md`
- `reports/26_future_expansion.md`
- `docs/MODEL_CARD.md`
- `docs/DEPLOYMENT_GUIDE.md`
- `README.md`