import sys
import os
import json
import numpy as np
import pandas as pd
import joblib

# --- Configuration ---
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'water_stress_dataset.csv')

FEATURE_NAMES = [
    "water_soil", "soil_temp_moisture", "conduct_soil", "leaf_moisture",
    "leaf_temperature", "ph1_soil", "soil_temp_ph", "soilnitrogen",
    "soilphosphorous", "soilpottasium", "hour_of_day", "is_daylight",
    "day_of_week", "day_of_period", "moisture_rolling_std",
    "moisture_rate_of_change", "temp_rolling_max"
]

# --- Define Test Scenarios ---
scenarios = {
    "Scenario A (Expected: Severely Stressed)": {
        "water_soil": 5.0, "soil_temp_moisture": 36.0, "conduct_soil": 112.0,
        "leaf_moisture": 4.9, "leaf_temperature": 45.0, "ph1_soil": 5.62,
        "soil_temp_ph": 36.0, "soilnitrogen": 2, "soilphosphorous": 7,
        "soilpottasium": 3, "hour_of_day": 14, "is_daylight": 1,
        "day_of_week": 3, "day_of_period": 45, "moisture_rolling_std": 2.0,
        "moisture_rate_of_change": -1.0, "temp_rolling_max": 36.0,
    },
    "Scenario B (Expected: Moderately Stressed)": {
        "water_soil": 15.8, "soil_temp_moisture": 24.3, "conduct_soil": 85.0,
        "leaf_moisture": 13.6, "leaf_temperature": 28.0, "ph1_soil": 5.69,
        "soil_temp_ph": 24.3, "soilnitrogen": 2, "soilphosphorous": 7,
        "soilpottasium": 3, "hour_of_day": 14, "is_daylight": 1,
        "day_of_week": 3, "day_of_period": 30, "moisture_rolling_std": 1.5,
        "moisture_rate_of_change": -0.5, "temp_rolling_max": 26.0,
    },
    "Scenario C (Expected: Severely Stressed)": {
        "water_soil": 11.0, "soil_temp_moisture": 38.0, "conduct_soil": 45.0,
        "leaf_moisture": 3.0, "leaf_temperature": 42.0, "ph1_soil": 5.8,
        "soil_temp_ph": 35.0, "soilnitrogen": 1, "soilphosphorous": 2,
        "soilpottasium": 1, "hour_of_day": 14, "is_daylight": 1,
        "day_of_week": 3, "day_of_period": 45, "moisture_rolling_std": 2.5,
        "moisture_rate_of_change": -3.0, "temp_rolling_max": 40.0,
    },
    "Scenario D (Expected: Healthy)": {
        "water_soil": 22.0, "soil_temp_moisture": 24.0, "conduct_soil": 90.0,
        "leaf_moisture": 30.0, "leaf_temperature": 22.0, "ph1_soil": 5.5,
        "soil_temp_ph": 22.0, "soilnitrogen": 5, "soilphosphorous": 10,
        "soilpottasium": 6, "hour_of_day": 8, "is_daylight": 1,
        "day_of_week": 1, "day_of_period": 20, "moisture_rolling_std": 0.5,
        "moisture_rate_of_change": 0.1, "temp_rolling_max": 25.0,
    },
}

EXPECTED = {
    "Scenario A (Expected: Severely Stressed)": "Severely Stressed",
    "Scenario B (Expected: Moderately Stressed)": "Moderately Stressed",
    "Scenario C (Expected: Severely Stressed)": "Severely Stressed",
    "Scenario D (Expected: Healthy)": "Healthy",
}


def main():
    # =========================================================
    # PART 1: Load model, encoder, feature names and run predictions
    # =========================================================
    print("=" * 80)
    print("PART 1: MODEL PREDICTION VERIFICATION")
    print("=" * 80)

    model = joblib.load(os.path.join(MODELS_DIR, 'stress_model.pkl'))
    le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    with open(os.path.join(MODELS_DIR, 'feature_names.json')) as f:
        feature_names = json.load(f)

    print()
    print("Model type        :", type(model).__name__)
    print("Feature count     :", len(feature_names))
    print("Feature names     :", feature_names)
    print("Encoder classes   :", list(le.classes_))
    print()

    results = []
    for name, inputs in scenarios.items():
        X = np.array([[inputs.get(f, 0.0) for f in feature_names]])
        pred_encoded = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        label = le.classes_[pred_encoded]
        confidence = float(proba[pred_encoded])

        class_probs = {}
        for i in range(len(le.classes_)):
            class_probs[le.classes_[i]] = round(float(proba[i]), 4)

        expected = EXPECTED[name]
        match = "PASS" if label == expected else "FAIL"

        print("-" * 80)
        print("  " + name)
        print("-" * 80)
        print("  Predicted class  :", label)
        print("  Expected class   :", expected)
        print("  Match            :", match)
        print("  Confidence       : {:.4f}".format(confidence))
        print("  Class Probabilities:")
        for cls, p in sorted(class_probs.items()):
            bar = "#" * int(p * 50)
            print("    {:<25s}: {:.4f}  {}".format(cls, p, bar))
        print()

        results.append({
            "scenario": name,
            "predicted": label,
            "expected": expected,
            "match": match,
            "confidence": confidence,
            "class_probabilities": class_probs,
        })

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["match"] == "PASS")
    print("=" * 80)
    print("SUMMARY: {}/{} scenarios passed".format(passed, total))
    print("=" * 80)

    # =========================================================
    # PART 2: Training Data Distribution Analysis
    # =========================================================
    print()
    print()
    print("=" * 80)
    print("PART 2: TRAINING DATA DISTRIBUTION ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    print()
    print("Dataset shape: {} rows x {} columns".format(df.shape[0], df.shape[1]))
    print("Columns:", list(df.columns))

    # Stress level distribution
    print()
    print("Stress Level Distribution:")
    for level, count in df['stress_level'].value_counts().items():
        print("  {:<25s}: {}".format(level, count))

    # Compute stats for the 17 feature columns
    print()
    print()
    print("-" * 80)
    print("{:<28s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Feature", "Min", "5th%", "Mean", "95th%", "Max"))
    print("-" * 80)

    feature_stats = {}
    for f in FEATURE_NAMES:
        if f in df.columns:
            col = df[f].astype(float)
            stats = {
                'min': col.min(),
                'p5': col.quantile(0.05),
                'mean': col.mean(),
                'p95': col.quantile(0.95),
                'max': col.max(),
            }
            feature_stats[f] = stats
            print("{:<28s} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                f, stats['min'], stats['p5'], stats['mean'], stats['p95'], stats['max']))
        else:
            print("{:<28s}  ** NOT FOUND IN CSV **".format(f))

    print("-" * 80)

    # =========================================================
    # PART 3: Check whether test inputs fall within training ranges
    # =========================================================
    print()
    print()
    print("=" * 80)
    print("PART 3: SCENARIO INPUTS vs. TRAINING DATA RANGES")
    print("=" * 80)

    for name, inputs in scenarios.items():
        print()
        print("  " + name)
        print("  {:<28s} {:>10s} {:>25s}  {}".format(
            "Feature", "Value", "[5th%-95th%] Range", "In-Range?"))
        print("  " + "-" * 78)
        all_in_range = True
        for f in FEATURE_NAMES:
            if f in feature_stats:
                val = inputs[f]
                lo = feature_stats[f]['p5']
                hi = feature_stats[f]['p95']
                in_range = lo <= val <= hi
                flag = "YES" if in_range else "NO"
                if not in_range:
                    all_in_range = False
                print("  {:<28s} {:>10.2f}   [{:>8.2f} - {:>8.2f}]   {}".format(
                    f, val, lo, hi, flag))
            else:
                print("  {:<28s}  ** stats unavailable **".format(f))
        if all_in_range:
            print()
            print("  Overall: ALL features within training 5th-95th percentile range")
        else:
            print()
            print("  Overall: SOME features OUTSIDE training 5th-95th percentile range")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
