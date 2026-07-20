import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from src.predictor import StressPredictor
from src.recommendation_engine import RecommendationEngine
from src.database import SensorDatabase
from src.llm_layer import LLMLayer

BASE = os.path.dirname(__file__)
MODEL_DIR = BASE
DB_PATH = os.path.join(BASE, "sensor_readings.db")
DATASET_PATH = os.path.join(BASE, "water_stress_dataset.csv")

st.set_page_config(page_title="Crop Stress Monitor", layout="wide")

@st.cache_resource
def load_predictor():
    return StressPredictor(MODEL_DIR)

@st.cache_resource
def load_db():
    return SensorDatabase(DB_PATH)

@st.cache_resource
def load_llm():
    return LLMLayer()

predictor = load_predictor()
db = load_db()
engine = RecommendationEngine()
llm = load_llm()

st.title("Crop Water Stress Monitor")

tab_live, tab_history, tab_voice, tab_data = st.tabs([
    "Live Monitor", "History", "Voice Assistant", "Raw Data"
])

with tab_live:
    st.subheader("Current Sensor Readings")

    col1, col2, col3 = st.columns(3)
    with col1:
        water_soil = st.number_input("Soil Moisture", 0.0, 30.0, 15.0, 0.1)
        soil_temp = st.number_input("Soil Temperature (C)", 10.0, 50.0, 24.0, 0.1)
        conduct = st.number_input("Soil EC", 0.0, 700.0, 80.0, 1.0)
    with col2:
        leaf_m = st.number_input("Leaf Moisture", 0.0, 100.0, 15.0, 0.1)
        leaf_t = st.number_input("Leaf Temperature (C)", 5.0, 60.0, 25.0, 0.1)
        ph = st.number_input("Soil pH", 2.0, 7.0, 5.6, 0.01)
    with col3:
        soil_temp_ph = st.number_input("Soil Temp (pH sensor, C)", 10.0, 50.0, 24.0, 0.1)
        n = st.number_input("Nitrogen", 0, 20, 2)
        p = st.number_input("Phosphorus", 0, 55, 6)
        k = st.number_input("Potassium", 0, 30, 3)

    if st.button("Analyze", type="primary"):
        now = datetime.now()
        sensor_data = {
            "water_soil": water_soil,
            "soil_temp_moisture": soil_temp,
            "conduct_soil": conduct,
            "leaf_moisture": leaf_m,
            "leaf_temperature": leaf_t,
            "ph1_soil": ph,
            "soil_temp_ph": soil_temp_ph,
            "soilnitrogen": n,
            "soilphosphorous": p,
            "soilpottasium": k,
            "hour_of_day": now.hour,
            "is_daylight": 1 if 6 <= now.hour <= 18 else 0,
            "day_of_week": now.weekday(),
            "day_of_period": 0,
            "moisture_rolling_std": 0.0,
            "moisture_rate_of_change": 0.0,
            "temp_rolling_max": soil_temp,
        }

        prediction = predictor.predict(sensor_data)
        recommendation = engine.generate(prediction, water_soil, soil_temp, conduct)
        explanation = llm.explain(recommendation)

        db.store_reading(sensor_data)
        db.store_prediction({"timestamp": now.isoformat(), **prediction}, recommendation)

        urgency = recommendation["urgency"]
        if urgency == "high":
            st.error(f"**{prediction['stress_level']}** — Confidence: {prediction['confidence']:.1%}")
        elif urgency == "medium":
            st.warning(f"**{prediction['stress_level']}** — Confidence: {prediction['confidence']:.1%}")
        else:
            st.success(f"**{prediction['stress_level']}** — Confidence: {prediction['confidence']:.1%}")

        st.markdown(f"### Recommendation")
        st.markdown(explanation)

        with st.expander("Detailed Analysis"):
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Reasons:**")
                for r in recommendation["reasons"]:
                    st.markdown(f"- {r}")
            with rc2:
                st.markdown("**Actions:**")
                for a in recommendation["actions"]:
                    st.markdown(f"- {a}")

        with st.expander("Class Probabilities"):
            probs = prediction["class_probabilities"]
            prob_df = pd.DataFrame([
                {"Class": k, "Probability": v} for k, v in probs.items()
            ])
            st.bar_chart(prob_df.set_index("Class"))

    st.divider()
    st.subheader("Quick Analysis from Historical Data")

    if os.path.exists(DATASET_PATH):
        if st.button("Analyze Latest Real Sensor Reading"):
            df = pd.read_csv(DATASET_PATH)
            df['datetime_rounded'] = pd.to_datetime(df['datetime_rounded'])
            latest = df.iloc[-1]

            sensor_data = {
                col: float(latest[col]) for col in predictor.feature_names
            }

            prediction = predictor.predict(sensor_data)
            recommendation = engine.generate(
                prediction,
                water_soil=latest.get("water_soil", 0),
                air_temp=latest.get("soil_temp_moisture", 0)
            )
            explanation = llm.explain(recommendation)

            urgency = recommendation["urgency"]
            if urgency == "high":
                st.error(f"**{prediction['stress_level']}** — {prediction['confidence']:.1%}")
            elif urgency == "medium":
                st.warning(f"**{prediction['stress_level']}** — {prediction['confidence']:.1%}")
            else:
                st.success(f"**{prediction['stress_level']}** — {prediction['confidence']:.1%}")

            st.info(explanation)

            st.json({
                "sensor_readings": {k: round(v, 2) if isinstance(v, float) else v
                                    for k, v in sensor_data.items()},
                "recommendation": recommendation
            })

with tab_history:
    st.subheader("Prediction History")

    preds = db.get_recent_predictions(50)
    if preds:
        for p in reversed(preds[-10:]):
            ts = p['timestamp'][:19]
            level = p['stress_level']
            conf = p['confidence']
            urgency = p['urgency']

            if urgency == "high":
                st.error(f"**{ts}** — {level} ({conf:.1%})")
            elif urgency == "medium":
                st.warning(f"**{ts}** — {level} ({conf:.1%})")
            else:
                st.success(f"**{ts}** — {level} ({conf:.1%})")

            for reason in p.get("reasons", [])[:2]:
                st.caption(f"  {reason}")
    else:
        st.info("No predictions yet. Go to Live Monitor and analyze some readings.")

    st.divider()
    st.subheader("Sensor Trends (from dataset)")

    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df['datetime_rounded'] = pd.to_datetime(df['datetime_rounded'])

        days = st.slider("Show last N days", 7, 60, 30)
        cutoff = df['datetime_rounded'].max() - timedelta(days=days)
        df_plot = df[df['datetime_rounded'] >= cutoff]

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Soil Moisture Over Time**")
            moisture_df = df_plot[['datetime_rounded', 'water_soil']].copy()
            moisture_df = moisture_df.set_index('datetime_rounded')
            st.line_chart(moisture_df)

        with chart_col2:
            st.markdown("**Leaf Moisture Over Time**")
            leaf_df = df_plot[['datetime_rounded', 'leaf_moisture']].copy()
            leaf_df = leaf_df.set_index('datetime_rounded')
            st.line_chart(leaf_df)

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            st.markdown("**Soil EC Over Time**")
            ec_df = df_plot[['datetime_rounded', 'conduct_soil']].copy()
            ec_df = ec_df.set_index('datetime_rounded')
            st.line_chart(ec_df)

        with chart_col4:
            st.markdown("**Stress Distribution**")
            stress_counts = df_plot['stress_level'].value_counts()
            st.bar_chart(stress_counts)

with tab_voice:
    st.subheader("Voice Assistant (Text Mode)")
    st.caption("Type questions as if you were speaking to the assistant.")

    query = st.text_input("Ask about your crop:", placeholder="How is my field?")
    if query and st.button("Ask"):
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

        response = llm.answer_query(query, context)
        db.store_voice_query(query, response)

        st.markdown(f"**You:** {query}")
        st.markdown(f"**Assistant:** {response}")

    st.divider()
    st.subheader("Recent Queries")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM voice_queries ORDER BY timestamp DESC LIMIT 10"
        ).fetchall()
        for r in rows:
            d = dict(r)
            st.caption(f"**You:** {d['query']}")
            st.caption(f"**Bot:** {d['response']}")

with tab_data:
    st.subheader("Live Sensor Data")
    readings = db.get_recent_readings(500)
    if readings:
        df_live = pd.DataFrame(readings)
        st.dataframe(df_live, use_container_width=True, height=400)
        st.caption(f"**{len(readings)} readings** | Latest: {readings[0]['timestamp'][:19]}")
    else:
        st.info("No sensor data yet. Use the Live Monitor tab to add readings.")

    st.divider()
    st.subheader("Prediction History")
    preds = db.get_recent_predictions(50)
    if preds:
        df_preds = pd.DataFrame(preds)
        st.dataframe(
            df_preds[['timestamp', 'stress_level', 'confidence', 'urgency']],
            use_container_width=True, height=300
        )
    else:
        st.info("No predictions yet.")

    st.divider()
    st.subheader("Training Dataset (Reference)")
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        st.bar_chart(df['stress_level'].value_counts())
        st.caption(f"{len(df)} rows | {df.shape[1]} columns")
    else:
        st.warning("Dataset not found. Run build_dataset.py first.")

    st.divider()
    st.subheader("Database Stats")
    stats = db.get_stats()
    st.json(stats)
