import numpy as np
import joblib
import json
import os


class StressPredictor:
    def __init__(self, model_dir):
        self.model = joblib.load(os.path.join(model_dir, 'stress_model.pkl'))
        self.le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        with open(os.path.join(model_dir, 'feature_names.json')) as f:
            self.feature_names = json.load(f)

    def predict(self, sensor_data: dict) -> dict:
        X = np.array([[sensor_data.get(f, 0.0) for f in self.feature_names]])
        pred_encoded = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        label = self.le.classes_[pred_encoded]
        confidence = float(proba[pred_encoded])

        class_probs = {
            self.le.classes_[i]: round(float(proba[i]), 4)
            for i in range(len(self.le.classes_))
        }

        importances = dict(zip(
            self.feature_names,
            [round(float(v), 4) for v in self.model.feature_importances_]
        ))

        return {
            "stress_level": label,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probs,
            "feature_importances": importances,
            "input_features": sensor_data,
        }
