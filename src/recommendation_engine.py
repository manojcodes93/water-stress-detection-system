from datetime import datetime


class RecommendationEngine:
    MOISTURE_CRITICAL = 13.0
    MOISTURE_LOW = 15.0
    MOISTURE_ADEQUATE = 18.0
    TEMP_HIGH = 35.0
    TEMP_VERY_HIGH = 40.0
    HUMIDITY_LOW = 30.0
    EC_HIGH = 150.0
    PH_LOW = 5.0
    PH_HIGH = 6.2
    NPK_LOW_N = 2
    NPK_LOW_P = 3
    NPK_LOW_K = 2

    def generate(self, prediction: dict, water_soil: float = None,
                 air_temp: float = None, humidity: float = None) -> dict:
        features = prediction.get("input_features", {})
        stress = prediction["stress_level"]
        confidence = prediction["confidence"]

        if water_soil is None:
            water_soil = features.get("water_soil", 0.0)
        if air_temp is None:
            air_temp = features.get("soil_temp_moisture", 0.0)
        if humidity is None:
            humidity = features.get("conduct_soil", 0.0)

        reasons = []
        actions = []
        urgency = "low"

        if stress == "Severely Stressed":
            urgency = "high"
            reasons.append("Crop is under severe water stress")
            actions.append("Irrigate immediately")
        elif stress == "Moderately Stressed":
            urgency = "medium"
            reasons.append("Crop shows moderate water stress")
            actions.append("Schedule irrigation within the next few hours")
        else:
            reasons.append("Crop is healthy with adequate moisture")

        if water_soil > 0 and water_soil < self.MOISTURE_CRITICAL:
            reasons.append(f"Soil moisture is critically low at {water_soil:.1f}")
            actions.append("Apply at least 25mm of water")
        elif water_soil > 0 and water_soil < self.MOISTURE_LOW:
            reasons.append(f"Soil moisture is below optimal at {water_soil:.1f}")
            actions.append("Apply 15-20mm of water")

        if air_temp > self.TEMP_VERY_HIGH:
            reasons.append(f"Very high temperature ({air_temp:.1f}C) increasing evaporation rapidly")
            actions.append("Irrigate during early morning or late evening to reduce evaporation loss")
            urgency = "high"
        elif air_temp > self.TEMP_HIGH:
            reasons.append(f"High temperature ({air_temp:.1f}C) accelerating moisture loss")

        leaf_m = features.get("leaf_moisture", 0)
        if leaf_m > 0 and leaf_m < 5.0:
            reasons.append(f"Leaf moisture is very low ({leaf_m:.1f}), indicating plant-level water deficit")
        elif leaf_m > 0 and leaf_m < 10.0:
            reasons.append(f"Leaf moisture is declining ({leaf_m:.1f})")

        ec = features.get("conduct_soil", 0)
        if ec > self.EC_HIGH:
            reasons.append(f"High soil salinity (EC={ec:.0f}) may be affecting water uptake")
            actions.append("Consider leaching irrigation to flush salts")

        ph = features.get("ph1_soil", 0)
        if ph > 0 and ph < self.PH_LOW:
            reasons.append(f"Soil is too acidic (pH={ph:.2f}), which can reduce nutrient absorption")
        elif ph > 0 and ph > self.PH_HIGH:
            reasons.append(f"Soil is too alkaline (pH={ph:.2f}), which can lock out nutrients")

        n = features.get("soilnitrogen", 0)
        p = features.get("soilphosphorous", 0)
        k = features.get("soilpottasium", 0)
        if n < self.NPK_LOW_N:
            reasons.append("Soil nitrogen is low")
            actions.append("Consider nitrogen-rich fertilizer after irrigation")
        if p < self.NPK_LOW_P:
            reasons.append("Soil phosphorus is low")
        if k < self.NPK_LOW_K:
            reasons.append("Soil potassium is low")

        roc = features.get("moisture_rate_of_change", 0)
        if roc < -2.0:
            reasons.append(f"Soil moisture is dropping fast (rate: {roc:.1f}/hour)")
            if urgency != "high":
                urgency = "medium"
                actions.append("Monitor closely and be ready to irrigate soon")

        if not actions:
            actions.append("No immediate action required. Continue monitoring.")

        return {
            "timestamp": datetime.now().isoformat(),
            "stress_level": stress,
            "ml_confidence": confidence,
            "urgency": urgency,
            "reasons": reasons,
            "actions": actions,
            "sensor_summary": {
                "soil_moisture": round(water_soil, 2) if water_soil else None,
                "temperature": round(air_temp, 1) if air_temp else None,
                "leaf_moisture": round(features.get("leaf_moisture", 0), 1),
                "soil_pH": round(features.get("ph1_soil", 0), 2),
                "soil_EC": round(features.get("conduct_soil", 0), 0),
                "nitrogen": features.get("soilnitrogen", 0),
                "phosphorus": features.get("soilphosphorous", 0),
                "potassium": features.get("soilpottasium", 0),
            }
        }
