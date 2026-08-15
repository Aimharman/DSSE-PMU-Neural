"""Inference companion for neural_active_controller_v2.joblib.

IMPORTANT:
- Uses feature_extractor_v2, matching the v2 training feature set.
- Does not use simulator fault flags for neural inference.
- Fault flags are copied only into raw_* columns for post-run comparison.
"""

from __future__ import annotations

import ast
import joblib
import numpy as np
import pandas as pd

from feature_extractor_v2 import (
    WINDOW,
    TIMING_LONG,
    extract_window_features,
    window_label,
)

REQUIRED = []
for p in (1, 2, 3):
    REQUIRED += [
        f"PMU{p} Voltage Magnitude",
        f"PMU{p} Voltage Phase",
        f"PMU{p} Current Magnitude",
        f"PMU{p} Current Phase",
    ]


def _truth(v):
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def raw_fault_info(window):
    active = []
    for pmu in (1, 2, 3):
        if any(_truth(v) for v in window.get(f"PMU{pmu} Bad Data", [])):
            active.append((f"PMU{pmu}", "BAD_DATA"))
        if any(_truth(v) for v in window.get(f"PMU{pmu} Sync Fault Active", [])):
            active.append((f"PMU{pmu}", "SYNC"))
        if any(_truth(v) for v in window.get(f"PMU{pmu} Clock Drift Fault", [])):
            active.append((f"PMU{pmu}", "CLOCK_DRIFT"))

    if not active:
        return "NORMAL", "NONE"
    if len(active) == 1:
        return active[0][1], active[0][0]
    return "MIXED", ",".join(x[0] for x in active)


def _management(fault_type, pmu, confidence):
    if confidence < 0.70:
        return (
            "HOLD / REQUEST MORE DATA",
            [1.0, 1.0, 1.0],
            [1.0] * 12,
        )

    weights = [1.0, 1.0, 1.0]
    measurement = [1.0] * 12

    if fault_type == "NORMAL" or pmu == "NONE":
        return "ACCEPT ALL PMUs", weights, measurement

    p = int(pmu[-1])
    weights[p - 1] = 0.10

    # Measurement order:
    # PMU1 Vmag,Vphase,Imag,Iphase, PMU2..., PMU3...
    base = (p - 1) * 4

    if fault_type == "BAD_DATA":
        measurement[base:base + 4] = [0.10] * 4
        return f"DOWN-WEIGHT PMU{p}", weights, measurement

    if fault_type == "SYNC":
        # Preserve magnitude/current information, but reduce phase-sensitive
        # measurements for the affected PMU.
        measurement[base + 1] = 0.10
        measurement[base + 3] = 0.10
        return f"DOWN-WEIGHT PMU{p} AND APPLY PHASE CHECK", weights, measurement

    if fault_type == "CLOCK_DRIFT":
        measurement[base + 1] = 0.10
        measurement[base + 3] = 0.10
        return f"DOWN-WEIGHT PMU{p} AND APPLY TIMING CHECK", weights, measurement

    return "HOLD / REQUEST MORE DATA", [1.0, 1.0, 1.0], [1.0] * 12


def predict_window(window, history, bundle):
    features = extract_window_features(window, history=history)

    names = bundle["feature_names"]
    missing = [n for n in names if n not in features]
    if missing:
        raise KeyError(
            "v2 feature extractor/model mismatch. Missing features: "
            + ", ".join(missing[:20])
        )

    X = pd.DataFrame([[features[n] for n in names]], columns=names)

    type_model = bundle["type_model"]
    pmu_model = bundle["pmu_model"]

    type_prob = type_model.predict_proba(X)[0]
    type_classes = list(type_model.named_steps["mlp"].classes_)
    ti = int(np.argmax(type_prob))
    fault_type = str(type_classes[ti])
    type_conf = float(type_prob[ti])

    if fault_type == "NORMAL":
        pmu = "NONE"
        pmu_conf = 1.0
    else:
        pmu_prob = pmu_model.predict_proba(X)[0]
        pmu_classes = list(pmu_model.named_steps["mlp"].classes_)
        pi = int(np.argmax(pmu_prob))
        pmu = str(pmu_classes[pi])
        pmu_conf = float(pmu_prob[pi])

    confidence = min(type_conf, pmu_conf)
    action, weights, measurement_weights = _management(
        fault_type, pmu, confidence
    )

    return {
        "fault_type": fault_type,
        "faulty_pmu": pmu,
        "type_confidence": type_conf,
        "pmu_confidence": pmu_conf,
        "neural_confidence": confidence,
        "management_state": (
            "NORMAL" if fault_type == "NORMAL" else
            "FAULT" if confidence >= 0.70 else
            "UNCERTAIN"
        ),
        "management_action": action,
        "pmu_weights": weights,
        "measurement_weights": measurement_weights,
    }


def scan_csv(csv_path, model_path, out_path=None):
    bundle = joblib.load(model_path)
    df = pd.read_csv(csv_path).reset_index(drop=True)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PMU measurement columns: {missing}")

    rows = []

    # Match v2 training: provide 200 ms timing history.
    first_end = max(WINDOW - 1, TIMING_LONG - 1)

    for end in range(first_end, len(df), WINDOW):
        window = df.iloc[end - WINDOW + 1:end + 1]
        if len(window) != WINDOW:
            continue
        if window[REQUIRED].isna().any().any():
            continue

        hist_start = max(0, end - TIMING_LONG + 1)
        history = df.iloc[hist_start:end + 1]

        raw_type, raw_pmu = raw_fault_info(window)
        pred = predict_window(window, history, bundle)

        rows.append({
            "time_s": float(window["Time (s)"].iloc[-1])
                if "Time (s)" in window.columns else end / 1000.0,
            "raw_fault_type": raw_type,
            "raw_faulty_pmu": raw_pmu,
            "type_confidence": pred["type_confidence"],
            "pmu_confidence": pred["pmu_confidence"],
            "neural_confidence": pred["neural_confidence"],
            "management_state": pred["management_state"],
            "active_fault_type": pred["fault_type"],
            "active_faulty_pmu": pred["faulty_pmu"],
            "management_action": pred["management_action"],
            "pmu_weights": str(pred["pmu_weights"]),
            "measurement_weights": str(pred["measurement_weights"]),
        })

    out = pd.DataFrame(rows)
    if out_path:
        out.to_csv(out_path, index=False)
    return out
