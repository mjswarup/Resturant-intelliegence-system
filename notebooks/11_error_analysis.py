"""
PHASE 13: ERROR ANALYSIS
Realistic Rating Prediction Model (No Votes)
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR

print("=" * 70)
print("PHASE 13: ERROR ANALYSIS (Realistic Model)")
print("=" * 70)

# Load data and model
X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()

model = joblib.load(MODELS_DIR / "rating_model_realistic.joblib")
y_pred = model.predict(X_test)

# ============================================
# 1. BASIC ERROR METRICS
# ============================================
print("\n" + "─" * 70)
print("1. BASIC ERROR METRICS")
print("─" * 70)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ============================================
# 2. RESIDUALS
# ============================================
print("\n" + "─" * 70)
print("2. RESIDUAL ANALYSIS")
print("─" * 70)

residuals = y_test - y_pred

print(f"Mean Residual     : {residuals.mean():.4f}")
print(f"Std Residual      : {residuals.std():.4f}")
print(f"Min Residual      : {residuals.min():.4f}")
print(f"Max Residual      : {residuals.max():.4f}")

# ============================================
# 3. ACTUAL vs PREDICTED PLOT
# ============================================
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)
plt.plot([0, 5], [0, 5], 'r--', label="Perfect Prediction")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Rating (Realistic Model)")
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS_DIR / "07_actual_vs_predicted.png", dpi=150)
plt.close()
print("→ Saved: reports/07_actual_vs_predicted.png")

# ============================================
# 4. RESIDUAL DISTRIBUTION
# ============================================
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=40, kde=True, color="steelblue")
plt.axvline(0, color='red', linestyle='--')
plt.title("Residual Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "08_residual_distribution.png", dpi=150)
plt.close()
print("→ Saved: reports/08_residual_distribution.png")

# ============================================
# 5. WORST PREDICTIONS
# ============================================
print("\n" + "─" * 70)
print("5. WORST PREDICTIONS (Highest Absolute Error)")
print("─" * 70)

error_df = X_test.copy()
error_df['Actual'] = y_test
error_df['Predicted'] = y_pred
error_df['Absolute_Error'] = np.abs(residuals)

worst = error_df.nlargest(10, 'Absolute_Error')[
    ['City', 'Primary_Cuisine', 'Price range', 'Average Cost for two', 
     'Actual', 'Predicted', 'Absolute_Error']
]

print(worst.to_string(index=False))

# ============================================
# 6. PERFORMANCE BY PRICE RANGE
# ============================================
print("\n" + "─" * 70)
print("6. PERFORMANCE BY PRICE RANGE")
print("─" * 70)

error_df['Price range'] = X_test['Price range']
perf_by_price = error_df.groupby('Price range').apply(
    lambda g: pd.Series({
        'Count': len(g),
        'MAE': mean_absolute_error(g['Actual'], g['Predicted']),
        'R2': r2_score(g['Actual'], g['Predicted']) if len(g) > 1 else np.nan
    })
).round(4)

print(perf_by_price)

# ============================================
# 7. PERFORMANCE ON RATED vs NOT RATED
# ============================================
print("\n" + "─" * 70)
print("7. PERFORMANCE: Rated vs Not Rated (Actual = 0)")
print("─" * 70)

rated_mask = y_test > 0
print(f"Rated restaurants   (Actual > 0): {rated_mask.sum()}")
print(f"Not Rated           (Actual = 0): {(~rated_mask).sum()}")

print(f"\nMAE on Rated     : {mean_absolute_error(y_test[rated_mask], y_pred[rated_mask]):.4f}")
print(f"MAE on Not Rated : {mean_absolute_error(y_test[~rated_mask], y_pred[~rated_mask]):.4f}")

print("\n" + "=" * 70)
print("PHASE 13 COMPLETED - Error Analysis Finished")
print("=" * 70)