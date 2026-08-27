from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "controller.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


class NeuralController:
    """Final neural controller with separate fault-type and faulty-PMU models."""

    def __init__(self, model_path: str | Path | None = None, metadata_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else METADATA_PATH
        self.type_model = None
        self.pmu_model = None
        self.metadata = {
            "feature_count": 7,
            "classes": ["NORMAL", "BAD_DATA", "SYNC", "CLOCK_DRIFT"],
            "faulty_pmu_classes": [1, 2, 3],
        }
        self._load()

    def _load(self):
        if self.model_path.exists():
            payload = joblib.load(self.model_path)
            if isinstance(payload, dict):
                self.type_model = payload.get("fault_type_model")
                self.pmu_model = payload.get("pmu_model")
                self.metadata.update({k: v for k, v in payload.items() if k not in {"fault_type_model", "pmu_model"}})
            else:
                self.type_model = payload
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as fp:
                self.metadata.update(json.load(fp))

    def predict(self, feature_vector):
        features = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        if self.type_model is not None:
            pred_type = str(self.type_model.predict(features)[0]).upper()
            try:
                probs = self.type_model.predict_proba(features)[0]
                fault_conf = float(np.max(probs))
            except Exception:
                fault_conf = 0.5
            if pred_type == "NORMAL":
                return {"fault_type": "NORMAL", "faulty_pmu": 0, "fault_confidence": fault_conf, "pmu_confidence": 1.0, "confidence": fault_conf, "pmu": 0}
            if self.pmu_model is not None:
                pred_pmu = int(self.pmu_model.predict(features)[0])
                try:
                    pmu_conf = float(np.max(self.pmu_model.predict_proba(features)[0]))
                except Exception:
                    pmu_conf = 0.5
                return {
                    "fault_type": pred_type,
                    "faulty_pmu": pred_pmu,
                    "fault_confidence": fault_conf,
                    "pmu_confidence": pmu_conf,
                    "confidence": fault_conf,
                    "pmu": pred_pmu,
                }
            return {"fault_type": pred_type, "faulty_pmu": 0, "fault_confidence": fault_conf, "pmu_confidence": 0.0, "confidence": fault_conf, "pmu": 0}

        # Fallback rule-based logic.
        if features.size == 0:
            return {"fault_type": "NORMAL", "faulty_pmu": 0, "fault_confidence": 0.0, "pmu_confidence": 0.0, "confidence": 0.0, "pmu": 0}
        vals = features[0]
        offset = vals[0]
        short_slope = abs(vals[1])
        long_slope = abs(vals[2])
        variance = vals[3]
        step_change = vals[4]
        persistence = vals[5]
        if offset > 8 and short_slope < 2.0:
            fault_type = "SYNC"
            faulty_pmu = 1 if short_slope < 1 else 2
        elif long_slope > 2.0 and persistence > 0.5:
            fault_type = "CLOCK_DRIFT"
            faulty_pmu = 2
        elif step_change > 10 or variance > 30:
            fault_type = "BAD_DATA"
            faulty_pmu = 1
        else:
            fault_type = "NORMAL"
            faulty_pmu = 0
        confidence = 0.8 if fault_type != "NORMAL" else 0.9
        return {"fault_type": fault_type, "faulty_pmu": faulty_pmu, "fault_confidence": confidence, "pmu_confidence": confidence, "confidence": confidence, "pmu": faulty_pmu}
