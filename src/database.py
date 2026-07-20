import sqlite3
import json
import os
from datetime import datetime


class SensorDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    water_soil REAL,
                    soil_temp_moisture REAL,
                    conduct_soil REAL,
                    leaf_moisture REAL,
                    leaf_temperature REAL,
                    ph1_soil REAL,
                    soil_temp_ph REAL,
                    soilnitrogen INTEGER,
                    soilphosphorous INTEGER,
                    soilpottasium INTEGER,
                    hour_of_day INTEGER,
                    is_daylight INTEGER,
                    day_of_week INTEGER,
                    day_of_period INTEGER,
                    moisture_rolling_mean REAL,
                    moisture_rolling_std REAL,
                    moisture_rate_of_change REAL,
                    temp_rolling_max REAL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    stress_level TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    class_probabilities TEXT,
                    urgency TEXT,
                    reasons TEXT,
                    actions TEXT
                );

                CREATE TABLE IF NOT EXISTS voice_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sensor_ts ON sensor_readings(timestamp);
                CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp);
            """)

    def store_reading(self, reading: dict):
        with sqlite3.connect(self.db_path) as conn:
            cols = [k for k in reading if k != 'datetime_rounded']
            vals = [reading.get(k) for k in cols]
            placeholders = ', '.join(['?'] * len(cols))
            col_names = ', '.join(cols)
            conn.execute(
                f"INSERT INTO sensor_readings (timestamp, {col_names}) VALUES (?, {placeholders})",
                [reading.get('datetime_rounded', datetime.now().isoformat())] + vals
            )

    def store_prediction(self, prediction: dict, recommendation: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO predictions
                   (timestamp, stress_level, confidence, class_probabilities, urgency, reasons, actions)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction.get("timestamp", datetime.now().isoformat()),
                    prediction["stress_level"],
                    prediction["confidence"],
                    json.dumps(prediction.get("class_probabilities", {})),
                    recommendation.get("urgency", "low"),
                    json.dumps(recommendation.get("reasons", [])),
                    json.dumps(recommendation.get("actions", []))
                )
            )

    def store_voice_query(self, query: str, response: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO voice_queries (timestamp, query, response) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), query, response)
            )

    def get_recent_readings(self, limit=50):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_predictions(self, limit=20):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d['class_probabilities'] = json.loads(d['class_probabilities']) if d['class_probabilities'] else {}
                d['reasons'] = json.loads(d['reasons']) if d['reasons'] else []
                d['actions'] = json.loads(d['actions']) if d['actions'] else []
                results.append(d)
            return results

    def get_readings_since(self, hours=24):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM sensor_readings
                   WHERE timestamp >= datetime('now', ?)
                   ORDER BY timestamp ASC""",
                (f'-{hours} hours',)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self):
        with sqlite3.connect(self.db_path) as conn:
            readings = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
            predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            voice = conn.execute("SELECT COUNT(*) FROM voice_queries").fetchone()[0]
            latest = conn.execute(
                "SELECT timestamp FROM sensor_readings ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return {
                "total_readings": readings,
                "total_predictions": predictions,
                "total_voice_queries": voice,
                "latest_reading": latest[0] if latest else None
            }
