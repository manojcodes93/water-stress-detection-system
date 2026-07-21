import pandas as pd
import numpy as np
import os

base = r'D:\Projects\Water Stress Detection'

print("=" * 60)
print("MULTI-SENSOR STRESS RELABELING")
print("=" * 60)

df = pd.read_csv(os.path.join(base, 'data', 'water_stress_dataset.csv'))
df['datetime_rounded'] = pd.to_datetime(df['datetime_rounded'])

print(f"\nLoaded: {df.shape}")
print(f"Current labels: {df['stress_level'].value_counts().to_dict()}")

# ============================================================
# STEP 1: Compute individual stress scores (0-1 each)
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Computing individual stress scores")
print("=" * 60)

# Soil moisture: low = stressed (inverted)
df['score_moisture'] = 1 - (df['water_soil'] / 30.0)
df['score_moisture'] = df['score_moisture'].clip(0, 1)

# Leaf moisture: low = stressed (inverted)
df['score_leaf'] = 1 - (df['leaf_moisture'] / 100.0)
df['score_leaf'] = df['score_leaf'].clip(0, 1)

# Soil EC: high = stressed (salinity blocks water uptake)
df['score_ec'] = df['conduct_soil'] / 630.0
df['score_ec'] = df['score_ec'].clip(0, 1)

# Soil temp: high = more evaporation = stressed
df['score_soil_temp'] = (df['soil_temp_moisture'] - 15.0) / 35.0
df['score_soil_temp'] = df['score_soil_temp'].clip(0, 1)

# pH: deviation from optimal 5.5 = stressed
df['score_ph'] = abs(df['ph1_soil'] - 5.5) / 2.0
df['score_ph'] = df['score_ph'].clip(0, 1)

# NPK: low = stressed (inverted, averaged)
df['score_n'] = 1 - (df['soilnitrogen'] / 17.0)
df['score_p'] = 1 - (df['soilphosphorous'] / 50.0)
df['score_k'] = 1 - (df['soilpottasium'] / 25.0)
df['score_npk'] = (df['score_n'] + df['score_p'] + df['score_k']) / 3.0
df['score_npk'] = df['score_npk'].clip(0, 1)

# Leaf temperature: high = stressed
df['score_leaf_temp'] = (df['leaf_temperature'] - 15.0) / 40.0
df['score_leaf_temp'] = df['score_leaf_temp'].clip(0, 1)

# Show individual score distributions
for col in ['score_moisture', 'score_leaf', 'score_ec', 'score_soil_temp',
            'score_ph', 'score_npk', 'score_leaf_temp']:
    print(f"  {col:20s}: mean={df[col].mean():.3f}, min={df[col].min():.3f}, max={df[col].max():.3f}")

# ============================================================
# STEP 2: Compute composite stress score with weights
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Computing composite stress score")
print("=" * 60)

weights = {
    'score_moisture': 0.30,
    'score_leaf': 0.20,
    'score_ec': 0.15,
    'score_soil_temp': 0.10,
    'score_ph': 0.10,
    'score_npk': 0.10,
    'score_leaf_temp': 0.05,
}

print("  Weights:")
for k, v in weights.items():
    print(f"    {k:20s}: {v:.2f}")

df['composite_score'] = sum(df[k] * v for k, v in weights.items())

print(f"\n  Composite score stats:")
print(f"    Min:    {df['composite_score'].min():.3f}")
print(f"    Mean:   {df['composite_score'].mean():.3f}")
print(f"    Median: {df['composite_score'].median():.3f}")
print(f"    Max:    {df['composite_score'].max():.3f}")
print(f"    Std:    {df['composite_score'].std():.3f}")

# ============================================================
# STEP 3: Determine thresholds from percentiles
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Determining class thresholds")
print("=" * 60)

p25 = df['composite_score'].quantile(0.33)
p50 = df['composite_score'].quantile(0.66)
print(f"  33rd percentile: {p25:.3f}")
print(f"  66th percentile: {p50:.3f}")

# Use the actual percentiles as thresholds
SEVERE_THRESHOLD = p50
MODERATE_THRESHOLD = p25

print(f"\n  Thresholds:")
print(f"    Healthy:              composite_score < {MODERATE_THRESHOLD:.3f}")
print(f"    Moderately Stressed:  {MODERATE_THRESHOLD:.3f} <= composite_score < {SEVERE_THRESHOLD:.3f}")
print(f"    Severely Stressed:    composite_score >= {SEVERE_THRESHOLD:.3f}")

# ============================================================
# STEP 4: Apply new labels
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Applying new multi-sensor labels")
print("=" * 60)

def classify_composite(score):
    if score >= SEVERE_THRESHOLD:
        return 'Severely Stressed'
    elif score >= MODERATE_THRESHOLD:
        return 'Moderately Stressed'
    else:
        return 'Healthy'

df['stress_level'] = df['composite_score'].apply(classify_composite)

new_dist = df['stress_level'].value_counts()
print("\n  New class distribution:")
for cls, count in new_dist.items():
    pct = count / len(df) * 100
    print(f"    {cls}: {count} ({pct:.1f}%)")

# Show sensor stats per class
print("\n  Key sensor stats per class (NEW labels):")
for cls in ['Severely Stressed', 'Moderately Stressed', 'Healthy']:
    subset = df[df['stress_level'] == cls]
    print(f"\n    {cls} (n={len(subset)}):")
    print(f"      water_soil:      {subset['water_soil'].mean():.1f} +/- {subset['water_soil'].std():.1f}")
    print(f"      leaf_moisture:   {subset['leaf_moisture'].mean():.1f} +/- {subset['leaf_moisture'].std():.1f}")
    print(f"      conduct_soil:    {subset['conduct_soil'].mean():.0f} +/- {subset['conduct_soil'].std():.0f}")
    print(f"      soil_temp:       {subset['soil_temp_moisture'].mean():.1f} +/- {subset['soil_temp_moisture'].std():.1f}")
    print(f"      pH:              {subset['ph1_soil'].mean():.2f} +/- {subset['ph1_soil'].std():.2f}")
    print(f"      nitrogen:        {subset['soilnitrogen'].mean():.1f}")
    print(f"      phosphorus:      {subset['soilphosphorous'].mean():.1f}")
    print(f"      potassium:       {subset['soilpottasium'].mean():.1f}")
    print(f"      composite_score: {subset['composite_score'].mean():.3f}")

# ============================================================
# STEP 5: Compare old vs new labels
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Old vs New label comparison")
print("=" * 60)

old_df = pd.read_csv(os.path.join(base, 'data', 'water_stress_dataset.csv'))
old_labels = old_df['stress_level']
new_labels = df['stress_level']

agreement = (old_labels == new_labels).sum()
total = len(old_labels)
print(f"  Labels that stayed the same: {agreement}/{total} ({agreement/total*100:.1f}%)")
print(f"  Labels that changed:         {total - agreement}/{total} ({(total-agreement)/total*100:.1f}%)")

# Show which ones changed
changed = df[old_labels != new_labels][['water_soil', 'leaf_moisture', 'conduct_soil', 'soil_temp_moisture', 'ph1_soil', 'composite_score']].copy()
changed['old_label'] = old_labels[old_labels != new_labels].values
changed['new_label'] = new_labels[old_labels != new_labels].values
print(f"\n  Sample of changed labels:")
print(changed.head(10).to_string(index=False))

# ============================================================
# STEP 6: Save new dataset
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Saving new dataset")
print("=" * 60)

# Drop temporary score columns
output_cols = [
    'datetime_rounded',
    'water_soil', 'soil_temp_moisture', 'conduct_soil',
    'leaf_moisture', 'leaf_temperature',
    'ph1_soil', 'soil_temp_ph',
    'soilnitrogen', 'soilphosphorous', 'soilpottasium',
    'hour_of_day', 'is_daylight', 'day_of_week', 'day_of_period',
    'moisture_rolling_mean', 'moisture_rolling_std',
    'moisture_rate_of_change', 'temp_rolling_max',
    'stress_level'
]
final = df[output_cols]

output_path = os.path.join(base, 'data', 'water_stress_dataset.csv')
final.to_csv(output_path, index=False)
print(f"  Saved: {output_path}")
print(f"  Shape: {final.shape}")
print(f"  Classes: {final['stress_level'].value_counts().to_dict()}")

print("\n" + "=" * 60)
print("DONE - Dataset relabeled with multi-sensor composite scores")
print("=" * 60)
