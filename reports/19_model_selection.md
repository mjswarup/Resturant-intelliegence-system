Task 1: Restaurant Rating Prediction

Two candidate models were built (Phase 12):

Version	Test MAE	Test R²	Status
Random Forest — With Votes	0.1893	0.9632	Rejected for deployment
Random Forest — Without Votes (Realistic)	0.7732	0.5037	Selected for deployment

Decision: Deploy the Realistic (No Votes) model.

Reasoning:

Generalization over vanity metrics. The With-Votes model's R²=0.96 is not a genuine skill signal — Phase 12's ablation test proved Votes alone explains most of that score (dropping it alone crashed both RF and XGBoost to R²≈0.50). Deploying that model would mean shipping something that looks excellent in a demo but is actually leaking near-target information.
Real-world usability. At prediction time for a new restaurant, Votes doesn't exist yet — a restaurant has zero votes before it has any ratings. A model trained to depend on Votes is unusable for its most obvious real use case: predicting how a new restaurant will be rated before it accumulates history.
Honesty over impressiveness. R²≈0.50 is a defensible, honest number for this problem. It says "price range, location, and service features explain about half the variance in rating" — which is plausible and useful, versus a number that would collapse the moment Votes stopped being available.
The With-Votes model is kept, not discarded — documented and saved as a comparison artifact (rating_model_with_votes.joblib) for the final report, explicitly labeled as "not for deployment; demonstrates leakage risk."

Why Random Forest over XGBoost/Gradient Boosting (both without Votes)?
From Phase 11's full comparison, all three tree-based models perform similarly once Votes is present or absent consistently. Random Forest was carried forward because:

Comparable accuracy to XGBoost/Gradient Boosting in every run
Simpler hyperparameter surface, tuned cleanly via RandomizedSearchCV in Phase 12
Slightly more stable CV std deviation across folds in Phase 11 (0.0033 vs XGBoost's 0.0034, Gradient Boosting's 0.0032 — all close, RF included in the win either way since we're not chasing marginal gains at the cost of interpretability)
Native feature_importances_ and full SHAP TreeExplainer support, which Phase 17 relied on directly
Task 2: Restaurant Recommendation

Decision: Content-based filtering with TF-IDF + business-rule filters (Phase 14).

No model comparison was needed here — recommendation isn't a supervised task with ground truth to score against. The chosen approach (TF-IDF over Cuisine+City+Locality text, combined with explicit filters for cost/price range/delivery/booking, sorted by rating+votes) was selected because:

It directly matches the spec's requirement for structured, explainable output (Rank, Restaurant, Reason)
Cosine similarity from TF-IDF gives interpretable "why this restaurant" reasoning — every recommendation can point to matching cuisine/location text
Test runs (Phase 14) produced sensible, checkable results across three different preference profiles
Task 3: Cuisine Classification
Model	Accuracy	Weighted F1
Logistic Regression	0.3691	0.2447
Random Forest	0.4202	0.3766
XGBoost	0.4287	0.3856

Decision: XGBoost. It wins on both metrics, though the margin over Random Forest is modest (~1 point accuracy, ~1 point F1). Given 26 imbalanced classes with North Indian dominating (2,992 of ~8,900 samples), this level of performance is a known limitation of the feature set, not the algorithm choice — swapping the model wouldn't meaningfully change the ceiling here (see Limitations below).

Task 4: Location Intelligence

No model to select — this is a descriptive analytics task (Phase 16), not predictive. Deliverable is the aggregation logic itself (city/locality groupby functions), not a saved model artifact.

Cross-Cutting Selection Criteria Applied
Criterion	How it was weighted
Generalization	Primary — drove the Votes decision above all else
Test performance	Secondary — used to rank within already-generalizable candidates
Stability (CV std)	Tiebreaker among close performers
Interpretability	Directly required SHAP/permutation support — favored tree models throughout
Computational cost	Not a binding constraint at this dataset size (~9.5K rows); not a deciding factor
Business usefulness	Drove rejection of the With-Votes model despite its superior raw metric
Known Limitations Carried Forward
Rating model ceiling (~R²=0.50): the dataset lacks review text, service-quality signals, or time-series data that likely explain the other 50% of rating variance.
Cuisine classifier ceiling (~43% accuracy): class imbalance (North Indian is a third of the data) and weak non-text features limit this significantly; a text-based feature (restaurant name embeddings, description) would likely help more than a different algorithm.
Geographic bias: ~57% of the dataset is New Delhi/Gurgaon/Noida — all four tasks inherit this skew, and model behavior in international cities is based on much smaller samples.