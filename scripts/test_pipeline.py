import sys
import os
sys.path.insert(0, r'D:\Projects\Water Stress Detection')

from src.predictor import StressPredictor
from src.recommendation_engine import RecommendationEngine
from src.database import SensorDatabase
from src.llm_layer import LLMLayer

BASE = r'D:\Projects\Water Stress Detection'

print("=" * 60)
print("END-TO-END PIPELINE TEST")
print("=" * 60)

# Init all components
predictor = StressPredictor(os.path.join(BASE, 'models'))
engine = RecommendationEngine()
db = SensorDatabase(os.path.join(BASE, "data", "sensor_readings.db"))
llm = LLMLayer()

print("\n[1] Components loaded")

# Test scenarios
scenarios = [
    {
        "name": "Severely Stressed Crop",
        "data": {
            "water_soil": 11.0, "soil_temp_moisture": 38.0, "conduct_soil": 45.0,
            "leaf_moisture": 3.0, "leaf_temperature": 42.0, "ph1_soil": 5.8,
            "soil_temp_ph": 35.0, "soilnitrogen": 1, "soilphosphorous": 2,
            "soilpottasium": 1, "hour_of_day": 14, "is_daylight": 1,
            "day_of_week": 3, "day_of_period": 45, "moisture_rolling_std": 2.5,
            "moisture_rate_of_change": -3.0, "temp_rolling_max": 40.0
        }
    },
    {
        "name": "Healthy Crop",
        "data": {
            "water_soil": 22.0, "soil_temp_moisture": 24.0, "conduct_soil": 90.0,
            "leaf_moisture": 30.0, "leaf_temperature": 22.0, "ph1_soil": 5.5,
            "soil_temp_ph": 22.0, "soilnitrogen": 5, "soilphosphorous": 10,
            "soilpottasium": 6, "hour_of_day": 8, "is_daylight": 1,
            "day_of_week": 1, "day_of_period": 20, "moisture_rolling_std": 0.5,
            "moisture_rate_of_change": 0.1, "temp_rolling_max": 25.0
        }
    },
    {
        "name": "Moderately Stressed Crop",
        "data": {
            "water_soil": 14.5, "soil_temp_moisture": 30.0, "conduct_soil": 60.0,
            "leaf_moisture": 8.0, "leaf_temperature": 30.0, "ph1_soil": 5.9,
            "soil_temp_ph": 28.0, "soilnitrogen": 2, "soilphosphorous": 4,
            "soilpottasium": 2, "hour_of_day": 16, "is_daylight": 1,
            "day_of_week": 5, "day_of_period": 35, "moisture_rolling_std": 1.0,
            "moisture_rate_of_change": -1.5, "temp_rolling_max": 32.0
        }
    }
]

for scenario in scenarios:
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"{'=' * 60}")

    # Predict
    prediction = predictor.predict(scenario["data"])
    print(f"\n  [ML] Stress: {prediction['stress_level']} ({prediction['confidence']:.1%})")
    print(f"  [ML] Probabilities: {prediction['class_probabilities']}")

    # Recommend
    recommendation = engine.generate(
        prediction,
        water_soil=scenario["data"]["water_soil"],
        air_temp=scenario["data"]["soil_temp_moisture"]
    )
    print(f"\n  [Rules] Urgency: {recommendation['urgency']}")
    print(f"  [Rules] Reasons:")
    for r in recommendation["reasons"]:
        print(f"    - {r}")
    print(f"  [Rules] Actions:")
    for a in recommendation["actions"]:
        print(f"    - {a}")

    # LLM explanation
    explanation = llm.explain(recommendation)
    print(f"\n  [LLM] {explanation}")

    # Store
    db.store_reading(scenario["data"])
    db.store_prediction(
        {"timestamp": "2025-01-01T12:00:00", **prediction},
        recommendation
    )
    print(f"\n  [DB] Stored reading + prediction")

print(f"\n{'=' * 60}")
print("VOICE QUERY TEST")
print(f"{'=' * 60}")

queries = [
    "How is my field?",
    "Should I water today?",
    "Why is my crop stressed?",
    "What is the temperature?",
]

for q in queries:
    recent_preds = db.get_recent_predictions(1)
    recent_readings = db.get_recent_readings(1)
    context = {
        "latest_prediction": recent_preds[0] if recent_preds else {},
        "latest_recommendation": {
            "reasons": recent_preds[0].get("reasons", []) if recent_preds else [],
            "actions": recent_preds[0].get("actions", []) if recent_preds else [],
        } if recent_preds else {},
        "latest_sensors": recent_readings[0] if recent_readings else {},
    }
    response = llm.answer_query(q, context)
    db.store_voice_query(q, response)
    print(f"\n  Q: {q}")
    print(f"  A: {response}")

print(f"\n{'=' * 60}")
print("DATABASE STATS")
print(f"{'=' * 60}")
stats = db.get_stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

print(f"\n{'=' * 60}")
print("ALL TESTS PASSED")
print(f"{'=' * 60}")
