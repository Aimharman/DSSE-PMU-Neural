"""Inference and active measurement-management layer.

This converts the neural prediction into an explicit management action.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from feature_extractor import WINDOW, extract_window_features


def load_controller(model_path: str):
    return joblib.load(model_path)


def management_action(label: str, confidence: float) -> dict:
    """Map neural classification to an active PDC measurement action."""
    if confidence < 0.70:
        return {"action": "HOLD / REQUEST MORE DATA", "weights": [1.0, 1.0, 1.0]}
    if label == "NORMAL":
        return {"action": "ACCEPT ALL PMUs", "weights": [1.0, 1.0, 1.0]}

    weights = [1.0, 1.0, 1.0]
    pmu = None
    for n in (1, 2, 3):
        if label.startswith(f"PMU{n}_"):
            pmu = n
            break

    if pmu is None:
        return {"action": "HOLD / REVIEW", "weights": weights}

    idx = pmu - 1
    if label.endswith("BAD_DATA"):
        weights[idx] = 0.10
        action = f"DOWN-WEIGHT PMU{pmu}"
    elif label.endswith("SYNC"):
        weights[idx] = 0.20
        action = f"DOWN-WEIGHT PHASE DATA OF PMU{pmu}"
    elif label.endswith("CLOCK_DRIFT"):
        weights[idx] = 0.20
        action = f"DOWN-WEIGHT PMU{pmu} AND APPLY TIMING CHECK"
    else:
        weights[idx] = 0.0
        action = f"ISOLATE PMU{pmu}"

    return {"action": action, "weights": weights}


def predict_window(df_window: pd.DataFrame, bundle: dict) -> dict:
    features = extract_window_features(df_window)
    X = pd.DataFrame([[features[n] for n in bundle["feature_names"]]], columns=bundle["feature_names"])
    model = bundle["model"]
    probabilities = model.predict_proba(X)[0]
    classes = model.named_steps["mlp"].classes_
    i = int(np.argmax(probabilities))
    label = str(classes[i])
    confidence = float(probabilities[i])
    action = management_action(label, confidence)
    return {
        "fault_class": label,
        "confidence": confidence,
        "action": action["action"],
        "weights": action["weights"],
    }


def scan_csv(csv_path: str, model_path: str, output_path: str = "Neural_PDC_Results.csv"):
    bundle = load_controller(model_path)
    df = pd.read_csv(csv_path)
    results = []
    required = []
    for pmu in (1, 2, 3):
        required += [
            f"PMU{pmu} Voltage Magnitude",
            f"PMU{pmu} Voltage Phase",
            f"PMU{pmu} Current Magnitude",
            f"PMU{pmu} Current Phase",
        ]

    for end in range(WINDOW - 1, len(df), WINDOW):
        window = df.iloc[end - WINDOW + 1 : end + 1]
        if len(window) != WINDOW or window[required].isna().any().any():
            continue
        result = predict_window(window, bundle)
        result["time_s"] = float(window["Time (s)"].iloc[-1])
        results.append(result)

    out = pd.DataFrame(results)
    out.to_csv(output_path, index=False)
    return out
