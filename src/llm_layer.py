import json
import os
import urllib.request


class LLMLayer:
    def __init__(self, api_key=None, provider="groq"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.provider = provider
        self.base_url = "https://api.groq.com/openai/v1/chat/complet"
        self.model = "llama-3.3-70b-versatile"

    def explain(self, recommendation: dict) -> str:
        if self.api_key:
            return self._call_llm(recommendation)
        return self._template_response(recommendation)

    def answer_query(self, query: str, recent_data: dict) -> str:
        if self.api_key:
            return self._call_llm_query(query, recent_data)
        return self._template_query(query, recent_data)

    def _call_llm(self, recommendation: dict) -> str:
        prompt = self._build_prompt(recommendation)
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": (
                    "You are a farming assistant. Explain crop water stress to a farmer "
                    "in simple, direct language. Use short sentences. No technical jargon. "
                    "Be specific about what to do and why."
                )},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }).encode()

        req = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._template_response(recommendation)

    def _call_llm_query(self, query: str, recent_data: dict) -> str:
        context = json.dumps(recent_data, indent=2, default=str)
        prompt = f"""Farmer's question: "{query}"

Recent sensor data and recommendations:
{context}

Answer in simple language. Be direct. Explain why if the farmer asks."""

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": (
                    "You are an intelligent irrigation assistant for farmers. "
                    "Answer questions about crop health in simple language. "
                    "Reference actual sensor readings when explaining. "
                    "If the farmer asks why, explain the reasoning with data."
                )},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }).encode()

        req = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._template_query(query, recent_data)

    def _build_prompt(self, rec: dict) -> str:
        reasons = "\n".join(f"- {r}" for r in rec.get("reasons", []))
        actions = "\n".join(f"- {a}" for a in rec.get("actions", []))
        sensors = rec.get("sensor_summary", {})

        return f"""Crop Status: {rec['stress_level']}
Urgency: {rec['urgency']}

Current readings:
- Soil moisture: {sensors.get('soil_moisture', 'N/A')}
- Temperature: {sensors.get('temperature', 'N/A')} C
- Leaf moisture: {sensors.get('leaf_moisture', 'N/A')}
- Soil pH: {sensors.get('soil_pH', 'N/A')}
- Soil EC: {sensors.get('soil_EC', 'N/A')}
- Nitrogen: {sensors.get('nitrogen', 'N/A')}
- Phosphorus: {sensors.get('phosphorus', 'N/A')}
- Potassium: {sensors.get('potassium', 'N/A')}

Reasons:
{reasons}

Recommended actions:
{actions}

Explain this to a farmer in 2-3 sentences. Be direct. Say what to do and why."""

    def _template_response(self, rec: dict) -> str:
        stress = rec["stress_level"]
        sensors = rec.get("sensor_summary", {})
        reasons = rec.get("reasons", [])
        actions = rec.get("actions", [])

        parts = []

        if stress == "Severely Stressed":
            parts.append("Your crop is under severe water stress and needs immediate attention.")
        elif stress == "Moderately Stressed":
            parts.append("Your crop is showing moderate signs of water stress.")
        else:
            parts.append("Your crop is healthy right now.")

        moisture = sensors.get("soil_moisture")
        temp = sensors.get("temperature")
        if moisture is not None and moisture > 0:
            parts.append(f"Soil moisture is at {moisture:.1f}.")
        if temp is not None and temp > 0:
            parts.append(f"Temperature is {temp:.1f}C.")

        if actions:
            parts.append(f"Recommended: {actions[0].lower()}.")

        if reasons and len(reasons) > 1:
            key_reason = reasons[1] if len(reasons) > 1 else reasons[0]
            parts.append(f"Reason: {key_reason.lower()}.")

        return " ".join(parts)

    def _template_query(self, query: str, recent_data: dict) -> str:
        q = query.lower()

        if any(w in q for w in ["how is", "status", "condition", "health"]):
            pred = recent_data.get("latest_prediction", {})
            sensors = recent_data.get("latest_sensors", {})
            level = pred.get("stress_level", "unknown")
            moisture = sensors.get("water_soil", "unknown")
            if level == "Healthy":
                return f"Your crop is healthy. Soil moisture is at {moisture}. No irrigation needed right now."
            elif level == "Moderately Stressed":
                return f"Your crop has moderate water stress. Soil moisture is at {moisture}. Consider irrigating soon."
            else:
                return f"Your crop is severely stressed! Soil moisture is at {moisture}. Irrigate immediately."

        if any(w in q for w in ["water", "irrigat", "should i"]):
            pred = recent_data.get("latest_prediction", {})
            rec = recent_data.get("latest_recommendation", {})
            level = pred.get("stress_level", "unknown")
            actions = rec.get("actions", [])
            if actions:
                return actions[0] + ". " + (actions[1] if len(actions) > 1 else "")
            return "No irrigation action recommended at this time."

        if any(w in q for w in ["why", "reason", "explain"]):
            rec = recent_data.get("latest_recommendation", {})
            reasons = rec.get("reasons", [])
            if reasons:
                return "Because: " + "; ".join(reasons[:3]) + "."
            return "I don't have enough data to explain the current conditions."

        if any(w in q for w in ["temp", "hot", "cold", "temperature"]):
            sensors = recent_data.get("latest_sensors", {})
            temp = sensors.get("soil_temp_moisture", "unknown")
            return f"Current temperature is {temp}C."

        return "I can help you check crop status, irrigation needs, or explain why your crop is stressed. What would you like to know?"
