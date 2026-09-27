"""
PHASE 15: TASK 3 - CUISINE CLASSIFICATION
Primary Cuisine Classification (Single-label)
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.config.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE

print("=" * 70)
print("PHASE 15: CUISINE CLASSIFICATION")
print("=" * 70)

# ============================================
# 1. LOAD DATA
# ============================================
df = pd.read_csv(PROCESSED_DATA_DIR / "featured_restaurants.csv")
print(f"Loaded data: {df.shape}")

# ============================================
# 2. PREPARE TARGET (Primary Cuisine)
# ============================================
print("\n" + "─" * 70)
print("2. PRIMARY CUISINE DISTRIBUTION")
print("─" * 70)

# Keep only the most frequent cuisines (to avoid extreme imbalance)
top_cuisines = df['Primary_Cuisine'].value_counts()
print("Top 15 Primary Cuisines:")
print(top_cuisines.head(15))

# Keep cuisines that appear at least 50 times
valid_cuisines = top_cuisines[top_cuisines >= 50].index.tolist()
df_clf = df[df['Primary_Cuisine'].isin(valid_cuisines)].copy()

print(f"\nAfter filtering (min 50 samples): {df_clf.shape}")
print(f"Number of classes: {df_clf['Primary_Cuisine'].nunique()}")

# ============================================
# 3. FEATURES & TARGET
# ============================================
feature_cols = [
    'Country Code', 'City', 'Average Cost for two', 'Price range',
    'Has Table booking', 'Has Online delivery', 'Aggregate rating',
    'Votes', 'Cuisine_Count', 'Name_Length', 'City_Frequency',
    'Locality_Frequency', 'Log_Average_Cost'
]

X = df_clf[feature_cols]
y = df_clf['Primary_Cuisine']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nX shape: {X.shape}")
print(f"Number of classes: {len(le.classes_)}")

# ============================================
# 4. TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ============================================
# 5. PREPROCESSING
# ============================================
numerical_features = [
    'Country Code', 'Average Cost for two', 'Price range',
    'Has Table booking', 'Has Online delivery', 'Aggregate rating',
    'Votes', 'Cuisine_Count', 'Name_Length', 'City_Frequency',
    'Locality_Frequency', 'Log_Average_Cost'
]

categorical_features = ['City']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_features)
])

# ============================================
# 6. TRAIN MODELS
# ============================================
print("\n" + "─" * 70)
print("6. TRAINING CLASSIFIERS")
print("─" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, use_label_encoder=False)
}

results = []

for name, model in models.items():
    print(f"\n→ Training: {name}")
    
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Weighted F1": round(f1, 4)
    })
    
    print(f"   Accuracy    : {acc:.4f}")
    print(f"   Weighted F1 : {f1:.4f}")

# ============================================
# 7. RESULTS
# ============================================
print("\n" + "─" * 70)
print("7. MODEL COMPARISON")
print("─" * 70)

results_df = pd.DataFrame(results).sort_values("Weighted F1", ascending=False)
print(results_df.to_string(index=False))

# ============================================
# 8. SAVE BEST MODEL
# ============================================
print("\n" + "─" * 70)
print("8. SAVING BEST MODEL")
print("─" * 70)

best_name = results_df.iloc[0]["Model"]
print(f"Best model: {best_name}")

# Retrain best model on full train set
best_model = models[best_name]
final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])
final_pipe.fit(X_train, y_train)

joblib.dump(final_pipe, MODELS_DIR / "cuisine_classifier.joblib")
joblib.dump(le, MODELS_DIR / "cuisine_label_encoder.joblib")

print("✅ Saved: models/cuisine_classifier.joblib")
print("✅ Saved: models/cuisine_label_encoder.joblib")

print("\n" + "=" * 70)
print("PHASE 15 COMPLETED - Cuisine Classification Finished")
print("=" * 70)