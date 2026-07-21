import pandas as pd
import numpy as np
import os

base = r'D:\Projects\Water Stress Detection'

print("=" * 60)
print("STEP 1: Reading IoT sensor files")
print("=" * 60)

leaf = pd.read_csv(os.path.join(base, 'Leaf+Parameters+Trend.csv'))
soil_ph = pd.read_csv(os.path.join(base, 'Soil+Parameters+Trend.csv'))
soil_npk = pd.read_csv(os.path.join(base, 'Soil+NPK (1).csv'))
soil_moisture = pd.read_csv(os.path.join(base, 'Soil+Trend+values (1).csv'))

for name, df in [('Leaf', leaf), ('Soil pH', soil_ph), ('NPK', soil_npk), ('Soil Moisture', soil_moisture)]:
    print(f"\n--- {name} ---")
    print(f"  Rows: {df.shape[0]}, Cols: {df.shape[1]}")
    print(f"  Date range: {df['Date'].iloc[-1]} to {df['Date'].iloc[0]}")

print("\n" + "=" * 60)
print("STEP 2: Parsing datetimes, rounding to 15-min windows")
print("=" * 60)

def parse_datetime(df):
    dt = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    dt_rounded = dt.dt.round('15min')
    return dt, dt_rounded

for name, df in [('Leaf', leaf), ('Soil pH', soil_ph), ('NPK', soil_npk), ('Soil Moisture', soil_moisture)]:
    dt, dt_rounded = parse_datetime(df)
    df['datetime'] = dt
    df['datetime_rounded'] = dt_rounded
    print(f"  {name}: {dt.min()} to {dt.max()} | {df['datetime_rounded'].nunique()} unique 15-min windows")

print("\n" + "=" * 60)
print("STEP 3: Extracting useful columns")
print("=" * 60)

leaf_clean = leaf[['datetime', 'datetime_rounded', 'leaf_moisture', 'leaf_temperature']].copy()
soil_ph_clean = soil_ph[['datetime', 'datetime_rounded', 'ph1_soil', 'temp_soil']].copy()
soil_ph_clean.rename(columns={'temp_soil': 'soil_temp_ph'}, inplace=True)
npk_clean = soil_npk[['datetime', 'datetime_rounded', 'soilnitrogen', 'soilphosphorous', 'soilpottasium']].copy()
soil_m_clean = soil_moisture[['datetime', 'datetime_rounded', 'water_soil', 'temp_soil', 'conduct_soil']].copy()
soil_m_clean.rename(columns={'temp_soil': 'soil_temp_moisture'}, inplace=True)

print(f"  Leaf: leaf_moisture [{leaf_clean['leaf_moisture'].min():.1f}, {leaf_clean['leaf_moisture'].max():.1f}]")
print(f"  Soil pH: pH [{soil_ph_clean['ph1_soil'].min():.2f}, {soil_ph_clean['ph1_soil'].max():.2f}]")
print(f"  NPK: N[{npk_clean['soilnitrogen'].min()}-{npk_clean['soilnitrogen'].max()}] P[{npk_clean['soilphosphorous'].min()}-{npk_clean['soilphosphorous'].max()}] K[{npk_clean['soilpottasium'].min()}-{npk_clean['soilpottasium'].max()}]")
print(f"  Soil Moisture: water_soil [{soil_m_clean['water_soil'].min():.2f}, {soil_m_clean['water_soil'].max():.2f}]")

print("\n" + "=" * 60)
print("STEP 4: Merging on rounded datetime (15-min windows)")
print("=" * 60)

all_times = pd.concat([
    leaf_clean['datetime_rounded'],
    soil_ph_clean['datetime_rounded'],
    npk_clean['datetime_rounded'],
    soil_m_clean['datetime_rounded']
]).drop_duplicates().sort_values().reset_index(drop=True)

master = pd.DataFrame({'datetime_rounded': all_times})
print(f"  Master timeline: {len(master)} unique 15-min windows")

for df in [leaf_clean, soil_ph_clean, npk_clean, soil_m_clean]:
    df.sort_values('datetime_rounded', inplace=True)

master.sort_values('datetime_rounded', inplace=True)
tolerance = pd.Timedelta('30min')

master = pd.merge_asof(master, soil_m_clean.drop(columns=['datetime']),
                       on='datetime_rounded', tolerance=tolerance, direction='nearest')
master = pd.merge_asof(master, leaf_clean.drop(columns=['datetime']),
                       on='datetime_rounded', tolerance=tolerance, direction='nearest')
master = pd.merge_asof(master, soil_ph_clean.drop(columns=['datetime']),
                       on='datetime_rounded', tolerance=tolerance, direction='nearest')
master = pd.merge_asof(master, npk_clean.drop(columns=['datetime']),
                       on='datetime_rounded', tolerance=tolerance, direction='nearest')

print(f"  Merged shape: {master.shape}")

missing = master.isnull().sum()
print("\n  Missing values after merge:")
for col in master.columns:
    if missing[col] > 0:
        print(f"    {col}: {missing[col]} ({missing[col]/len(master)*100:.1f}%)")

print("\n" + "=" * 60)
print("STEP 5: Handling missing values")
print("=" * 60)

sensor_cols = ['water_soil', 'soil_temp_moisture', 'conduct_soil',
               'leaf_moisture', 'leaf_temperature',
               'ph1_soil', 'soil_temp_ph',
               'soilnitrogen', 'soilphosphorous', 'soilpottasium']

master[sensor_cols] = master[sensor_cols].ffill(limit=2)

before = len(master)
master = master.dropna(subset=['water_soil'])
after = len(master)
print(f"  Dropped {before - after} rows with no soil moisture reading")
print(f"  Remaining: {after} rows")

for col in sensor_cols:
    if master[col].isnull().sum() > 0:
        n = master[col].isnull().sum()
        median_val = master[col].median()
        master[col] = master[col].fillna(median_val)
        print(f"  Filled {n} NaN in {col} with median ({median_val:.2f})")

print(f"  Total missing after cleanup: {master.isnull().sum().sum()}")

print("\n" + "=" * 60)
print("STEP 6: Engineering time features")
print("=" * 60)

master['hour_of_day'] = master['datetime_rounded'].dt.hour
master['is_daylight'] = ((master['hour_of_day'] >= 6) & (master['hour_of_day'] <= 18)).astype(int)
master['day_of_week'] = master['datetime_rounded'].dt.dayofweek
master['day_of_period'] = (master['datetime_rounded'] - master['datetime_rounded'].min()).dt.days

print(f"  hour_of_day: {master['hour_of_day'].min()}-{master['hour_of_day'].max()}")
print(f"  is_daylight: {master['is_daylight'].sum()} daylight / {(1-master['is_daylight']).sum()} night")
print(f"  day_of_period: {master['day_of_period'].min()}-{master['day_of_period'].max()} days")

print("\n" + "=" * 60)
print("STEP 7: Rolling statistics (3-hour window)")
print("=" * 60)

window = 12
master = master.sort_values('datetime_rounded').reset_index(drop=True)

master['moisture_rolling_mean'] = master['water_soil'].rolling(window=window, min_periods=1).mean().round(2)
master['moisture_rolling_std'] = master['water_soil'].rolling(window=window, min_periods=1).std().round(2).fillna(0)
master['moisture_rate_of_change'] = master['water_soil'].diff(periods=4).round(2).fillna(0)
master['temp_rolling_max'] = master['soil_temp_moisture'].rolling(window=window, min_periods=1).max().round(2)

print(f"  moisture_rolling_mean: [{master['moisture_rolling_mean'].min():.2f}, {master['moisture_rolling_mean'].max():.2f}]")
print(f"  moisture_rate_of_change: [{master['moisture_rate_of_change'].min():.2f}, {master['moisture_rate_of_change'].max():.2f}]")
print(f"  temp_rolling_max: [{master['temp_rolling_max'].min():.2f}, {master['temp_rolling_max'].max():.2f}]")

print("\n" + "=" * 60)
print("STEP 8: Creating stress labels (3-class)")
print("=" * 60)

def classify_stress(moisture):
    if moisture < 8:
        return 'Severely Stressed'
    elif moisture <= 15:
        return 'Moderately Stressed'
    else:
        return 'Healthy'

master['stress_level'] = master['water_soil'].apply(classify_stress)

class_dist = master['stress_level'].value_counts()
print("\n  Class distribution:")
for cls, count in class_dist.items():
    pct = count / len(master) * 100
    print(f"    {cls}: {count} ({pct:.1f}%)")

print("\n  water_soil stats per class:")
for cls in ['Severely Stressed', 'Moderately Stressed', 'Healthy']:
    subset = master[master['stress_level'] == cls]['water_soil']
    print(f"    {cls}: min={subset.min():.1f}, median={subset.median():.1f}, max={subset.max():.1f}")

print("\n" + "=" * 60)
print("STEP 9: Final dataset")
print("=" * 60)

final = master.drop(columns=['datetime'], errors='ignore')

feature_cols = [
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
final = final[feature_cols]

print(f"\n  Final shape: {final.shape}")
print(f"  Features: {final.shape[1] - 2} (excluding datetime and label)")
print(f"  Target: stress_level (3 classes)")

print("\n  Summary statistics:")
numeric_cols = final.select_dtypes(include=[np.number]).columns
summary = final[numeric_cols].describe().round(2)
print(summary.to_string())

print("\n  Correlation with water_soil:")
corr = final[numeric_cols].corr()['water_soil'].drop('water_soil').sort_values(ascending=False)
for col, val in corr.items():
    print(f"    {col}: {val:+.3f}")

output_path = os.path.join(base, 'data', 'water_stress_dataset.csv')
final.to_csv(output_path, index=False)
print(f"\n  Saved to: {output_path}")
print(f"  Total rows: {len(final)}")
print(f"  Date range: {final['datetime_rounded'].min()} to {final['datetime_rounded'].max()}")
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
