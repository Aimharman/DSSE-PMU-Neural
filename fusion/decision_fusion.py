from __future__ import annotations

from typing import Any


class DecisionFusion:
    """Fuse neural and classical decisions with the required confidence tiers."""

    def __init__(self):
        self.confidence_threshold = 0.7

    @staticmethod
    def normalize_neural(neural_result):
        if not isinstance(neural_result, dict):
            neural_result = {"fault_type": "NORMAL", "faulty_pmu": 0, "fault_confidence": 0.0}
        return {
            "fault_type": str(neural_result.get("fault_type", "NORMAL")).upper(),
            "faulty_pmu": int(neural_result.get("faulty_pmu", 0) or 0),
            "fault_confidence": float(neural_result.get("fault_confidence", neural_result.get("confidence", 0.0))),
        }

    @staticmethod
    def normalize_classical(classical_result):
        if not isinstance(classical_result, dict):
            classical_result = {"anomaly": False, "chi_square": 0.0, "p_value": 1.0, "dominant_pmu": 0, "residual_share": 0.0}
        p_value = float(classical_result.get("p_value", 1.0))
        return {
            "anomaly": bool(classical_result.get("anomaly", False) or classical_result.get("bad_data_detected", False) or (p_value < 0.05)),
            "chi_square": float(classical_result.get("chi_square", 0.0)),
            "p_value": p_value,
            "dominant_pmu": int(classical_result.get("dominant_pmu", 0) or 0),
            "residual_share": float(classical_result.get("residual_share", 0.0)),
            "confidence": p_value if not bool(classical_result.get("anomaly", False) or classical_result.get("bad_data_detected", False)) else max(0.0, 1.0 - p_value),
        }

    def fuse(self, neural_result, classical_result):
        neural = self.normalize_neural(neural_result)
        classical = self.normalize_classical(classical_result)

        neural_fault = neural["fault_type"] not in {"NORMAL", "NONE", "UNKNOWN"}
        classical_anomaly = classical["anomaly"]
        same_pmu = neural["faulty_pmu"] and classical["dominant_pmu"] and (neural["faulty_pmu"] == classical["dominant_pmu"])

        if neural_fault and classical_anomaly and same_pmu:
            level = "HIGH"
            decision = "HIGH"
            action = "PMU isolation and WLS re-estimation"
        elif neural_fault and classical_anomaly:
            level = "MEDIUM"
            decision = "MEDIUM"
            action = "Validate PMU localization and continue with conservative mitigation"
        elif neural_fault:
            level = "LOW"
            decision = "LOW"
            action = "Monitor and request classical confirmation"
        elif classical_anomaly:
            level = "STATISTICAL"
            decision = "STATISTICAL"
            action = "Classical anomaly confirmed; localize and isolate dominant PMU"
        else:
            level = "NORMAL"
            decision = "NORMAL"
            action = "Normal operation; continue without intervention"

        confidence = 0.0
        if level == "HIGH":
            confidence = min(0.99, 0.5 * neural["fault_confidence"] + 0.5 * classical["confidence"] + 0.1)
        elif level == "MEDIUM":
            confidence = min(0.85, 0.5 * neural["fault_confidence"] + 0.5 * classical["confidence"])
        elif level == "LOW":
            confidence = min(0.7, 0.7 * neural["fault_confidence"])
        elif level == "STATISTICAL":
            confidence = min(0.9, classical["confidence"] + 0.2)
        else:
            confidence = classical["p_value"] if classical["p_value"] > 0 else 0.9

        return {
            "decision": decision,
            "decision_name": decision,
            "level": level,
            "action": action,
            "confidence": float(confidence),
            "neural": neural,
            "classical": classical,
            "agreement": same_pmu or (neural_fault and classical_anomaly),
            "same_pmu": same_pmu,
        }
