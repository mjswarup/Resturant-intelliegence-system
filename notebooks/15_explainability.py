"""
PHASE 17: EXPLAINABLE AI
Realistic Rating Prediction Model
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from src.config.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR

print("=" * 70)
print("PHASE 17: EXPLAINABLE AI")
print("=" * 70)

# ============================================
# 1. LOAD MODEL AND DATA
# ============================================
model = joblib.load(MODELS_DIR / "rating_model_realistic.joblib")
X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()

print(f"Model loaded. Test set: {X_test.shape}")

# The model pipeline drops Votes internally via ColumnTransformer selection,
# but X_test still has the Votes column - the pipeline simply ignores it.
# We keep X_test as-is since the pipeline's ColumnTransformer selects columns by name.

# ============================================
# 2. PERMUTATION IMPORTANCE (on raw features, before preprocessing)
# ============================================
print("\n" + "─" * 70)
print("1. PERMUTATION IMPORTANCE")
print("─" * 70)

print("Computing permutation importance (this may take a minute)...")

perm_result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1, scoring='r2'
)

perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance_Mean': perm_result.importances_mean,
    'Importance_Std': perm_result.importances_std
}).sort_values('Importance_Mean', ascending=False)

print("\nTop 15 Features by Permutation Importance:")
print(perm_df.head(15).to_string(index=False))

# Plot
plt.figure(figsize=(10, 8))
top15 = perm_df.head(15).sort_values('Importance_Mean')
plt.barh(top15['Feature'], top15['Importance_Mean'], xerr=top15['Importance_Std'], color='steelblue')
plt.xlabel("Permutation Importance (R² drop)")
plt.title("Top 15 Features - Permutation Importance (Realistic Model)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "09_permutation_importance.png", dpi=150)
plt.close()
print("\n→ Saved: reports/09_permutation_importance.png")

# ============================================
# 3. BUILT-IN FEATURE IMPORTANCE (Random Forest)
# ============================================
print("\n" + "─" * 70)
print("2. MODEL'S BUILT-IN FEATURE IMPORTANCE")
print("─" * 70)

# Extract feature names after preprocessing
preprocessor = model.named_steps['preprocessor']
rf_model = model.named_steps['model']

# Get feature names from ColumnTransformer
num_features = preprocessor.transformers_[0][2]
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features_raw = preprocessor.transformers_[1][2]
cat_feature_names = cat_encoder.get_feature_names_out(cat_features_raw)

all_feature_names = list(num_features) + list(cat_feature_names)

builtin_importance = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Features (Built-in, after One-Hot Encoding):")
print(builtin_importance.head(15).to_string(index=False))

# ============================================
# 4. SHAP ANALYSIS
# ============================================
print("\n" + "─" * 70)
print("3. SHAP ANALYSIS")
print("─" * 70)

try:
    import shap
    
    print("Transforming test data through preprocessor...")
    X_test_transformed = preprocessor.transform(X_test)
    
    # Use a sample for speed
    sample_size = min(300, X_test_transformed.shape[0])
    sample_idx = np.random.RandomState(42).choice(X_test_transformed.shape[0], sample_size, replace=False)
    X_sample = X_test_transformed[sample_idx]
    
    print(f"Computing SHAP values on {sample_size} samples...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=all_feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "10_shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("→ Saved: reports/10_shap_summary.png")
    
    # Bar plot (mean absolute SHAP)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=all_feature_names, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "11_shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("→ Saved: reports/11_shap_bar.png")
    
    shap_available = True
    
except ImportError:
    print("⚠ SHAP not installed. Run: pip install shap")
    shap_available = False
except Exception as e:
    print(f"⚠ SHAP computation failed: {e}")
    shap_available = False

# ============================================
# 5. EXPLAIN INDIVIDUAL PREDICTIONS
# ============================================
print("\n" + "─" * 70)
print("4. INDIVIDUAL PREDICTION EXPLANATIONS")
print("─" * 70)

def explain_prediction(index):
    """Explain a single prediction using top features."""
    row = X_test.iloc[[index]]
    actual = y_test[index]
    predicted = model.predict(row)[0]
    
    print(f"\n--- Restaurant at index {index} ---")
    print(f"City: {row['City'].values[0]} | Cuisine: {row['Primary_Cuisine'].values[0]}")
    print(f"Price Range: {row['Price range'].values[0]} | Cost: {row['Average Cost for two'].values[0]}")
    print(f"Predicted Rating : {predicted:.2f}")
    print(f"Actual Rating    : {actual:.2f}")
    print(f"Prediction Error : {abs(predicted - actual):.2f}")
    
    if shap_available:
        row_transformed = preprocessor.transform(row)
        row_shap = explainer.shap_values(row_transformed)[0]
        
        shap_row_df = pd.DataFrame({
            'Feature': all_feature_names,
            'SHAP_Value': row_shap
        }).sort_values('SHAP_Value', key=abs, ascending=False)
        
        print("\nTop Positive Factors:")
        pos = shap_row_df[shap_row_df['SHAP_Value'] > 0].head(5)
        print(pos.to_string(index=False))
        
        print("\nTop Negative Factors:")
        neg = shap_row_df[shap_row_df['SHAP_Value'] < 0].head(5)
        print(neg.to_string(index=False))

# Explain 3 sample predictions
for idx in [0, 10, 50]:
    explain_prediction(idx)

# ============================================
# 6. SAVE IMPORTANCE TABLES
# ============================================
print("\n" + "─" * 70)
print("5. SAVING IMPORTANCE TABLES")
print("─" * 70)

perm_df.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
builtin_importance.to_csv(REPORTS_DIR / "builtin_feature_importance.csv", index=False)

print("✅ Saved: reports/permutation_importance.csv")
print("✅ Saved: reports/builtin_feature_importance.csv")

print("\n" + "=" * 70)
print("PHASE 17 COMPLETED - Explainable AI Finished")
print("=" * 70)