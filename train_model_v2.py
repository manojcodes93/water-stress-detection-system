import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

base = r'D:\Projects\Water Stress Detection'

# ============================================================
# STEP 1: Load and prepare data
# ============================================================
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df = pd.read_csv(os.path.join(base, 'water_stress_dataset.csv'))
df['datetime_rounded'] = pd.to_datetime(df['datetime_rounded'])

print(f"  Rows: {len(df)}")

# REMOVE data-leakage features
# water_soil defines the label — can't be a feature
# moisture_rolling_mean is 0.987 correlated with water_soil
leakage_cols = ['water_soil', 'moisture_rolling_mean']
feature_cols = [c for c in df.columns if c not in ['datetime_rounded', 'stress_level'] + leakage_cols]

print(f"\n  REMOVED (data leakage): {leakage_cols}")
print(f"  Features ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols].values
feature_names = feature_cols

# Target
le = LabelEncoder()
y = le.fit_transform(df['stress_level'].values)

print(f"  Target classes: {list(le.classes_)}")
for i, cls in enumerate(le.classes_):
    count = (y == i).sum()
    print(f"    {cls}: {count} ({count/len(y)*100:.1f}%)")

# ============================================================
# STEP 2: Train/test split
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Train/test split (80/20 stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
for i, cls in enumerate(le.classes_):
    train_n = (y_train == i).sum()
    test_n = (y_test == i).sum()
    print(f"    {cls}: train={train_n} ({train_n/len(y_train)*100:.1f}%), test={test_n} ({test_n/len(y_test)*100:.1f}%)")

# ============================================================
# STEP 3: Train Random Forest
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Training Random Forest (NO data leakage)")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("  Model trained")

# ============================================================
# STEP 4: Cross-validation
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: 5-Fold Cross-Validation")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# STEP 5: Test evaluation
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Test Set Evaluation")
print("=" * 60)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n  Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

print(f"\n  Classification Report:")
report = classification_report(y_test, y_pred, target_names=le.classes_, digits=3)
print(report)

print(f"  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  {'':>25} Predicted")
print(f"  {'':>25} {'Severely':>12} {'Moderately':>12} {'Healthy':>12}")
for i, cls in enumerate(le.classes_):
    print(f"  {'Actual ' + cls:>25} {cm[i][0]:>12} {cm[i][1]:>12} {cm[i][2]:>12}")

print(f"\n  Per-class accuracy:")
for i, cls in enumerate(le.classes_):
    class_mask = y_test == i
    if class_mask.sum() > 0:
        class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
        print(f"    {cls}: {class_acc:.4f} ({class_acc*100:.1f}%)")

# ============================================================
# STEP 6: Feature importance
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Feature Importance (without leakage)")
print("=" * 60)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\n  {'Rank':<6} {'Feature':<30} {'Importance':>10}")
print(f"  {'-'*48}")
for rank, idx in enumerate(indices):
    bar = '#' * max(1, int(importances[idx] * 100))
    print(f"  {rank+1:<6} {feature_names[idx]:<30} {importances[idx]:>10.4f}  {bar}")

# ============================================================
# STEP 7: Save
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Saving model")
print("=" * 60)

model_path = os.path.join(base, 'stress_model.pkl')
encoder_path = os.path.join(base, 'label_encoder.pkl')
features_path = os.path.join(base, 'feature_names.json')

joblib.dump(model, model_path)
joblib.dump(le, encoder_path)
with open(features_path, 'w') as f:
    json.dump(feature_names, f)

model_size = os.path.getsize(model_path) / 1024
print(f"  Model: {model_path} ({model_size:.1f} KB)")
print(f"  Encoder: {encoder_path}")
print(f"  Features: {features_path}")

# ============================================================
# STEP 8: Sanity check
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Sample Predictions")
print("=" * 60)

np.random.seed(42)
sample_indices = np.random.choice(len(X_test), 8, replace=False)

for idx in sample_indices:
    true_label = le.classes_[y_test[idx]]
    pred_label = le.classes_[y_pred[idx]]
    proba = model.predict_proba(X_test[idx:idx+1])[0]
    max_proba = proba.max()
    match = "OK" if true_label == pred_label else "WRONG"
    
    # Show key feature values for this sample
    leaf_m = X_test[idx][feature_names.index('leaf_moisture')]
    leaf_t = X_test[idx][feature_names.index('leaf_temperature')]
    ph = X_test[idx][feature_names.index('ph1_soil')]
    ec = X_test[idx][feature_names.index('conduct_soil')]
    
    print(f"\n  [{match}] leaf_m={leaf_m:.1f}, leaf_t={leaf_t:.1f}, pH={ph:.2f}, EC={ec:.0f}")
    print(f"    True: {true_label}")
    print(f"    Pred: {pred_label} ({max_proba:.1%})")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
