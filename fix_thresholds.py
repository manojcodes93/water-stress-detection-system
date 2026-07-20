import pandas as pd
import numpy as np
import os

base = r'D:\Projects\Water Stress Detection'

df = pd.read_csv(os.path.join(base, 'water_stress_dataset.csv'))
df['datetime_rounded'] = pd.to_datetime(df['datetime_rounded'])

print(f"Loaded: {df.shape}")
print(f"Zeros in water_soil: {(df['water_soil'] == 0).sum()}")

# Remove zero readings (sensor errors)
before = len(df)
df = df[df['water_soil'] > 0].copy()
print(f"Removed {before - len(df)} zero-readings")
print(f"Remaining: {len(df)} rows")

# Apply new thresholds based on actual data distribution
def classify_stress(moisture):
    if moisture < 13:
        return 'Severely Stressed'
    elif moisture <= 18:
        return 'Moderately Stressed'
    else:
        return 'Healthy'

df['stress_level'] = df['water_soil'].apply(classify_stress)

class_dist = df['stress_level'].value_counts()
print("\n=== New Class Distribution ===")
for cls, count in class_dist.items():
    pct = count / len(df) * 100
    subset = df[df['stress_level'] == cls]['water_soil']
    print(f"  {cls}: {count} ({pct:.1f}%) | water_soil: min={subset.min():.1f}, median={subset.median():.1f}, max={subset.max():.1f}")

# Verify no zeros remain
print(f"\nZeros after cleanup: {(df['water_soil'] == 0).sum()}")
print(f"Min water_soil: {df['water_soil'].min():.2f}")

# Save
output_path = os.path.join(base, 'water_stress_dataset.csv')
df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
print(f"Final shape: {df.shape}")
print(f"Date range: {df['datetime_rounded'].min()} to {df['datetime_rounded'].max()}")
