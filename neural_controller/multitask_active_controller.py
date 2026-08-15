"""Primary Neural Active Fault Management Controller.

Two-stage neural inference:
    1. fault type: NORMAL / BAD_DATA / SYNC / CLOCK_DRIFT
    2. affected PMU: PMU1 / PMU2 / PMU3

The neural models receive measurement-derived features only. Simulator truth
columns are used during training, never during inference.

A temporal management layer prevents one-window predictions from rapidly
changing the active protection state.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from feature_extractor import WINDOW, extract_window_features


# ----------------------------- Policy settings -----------------------------

CONFIDENCE_THRESHOLD = 0.70
ACTIVATION_WINDOWS = 2       # consecutive fault predictions before action
RECOVERY_WINDOWS = 3         # consecutive NORMAL predictions before recovery

BAD_DATA_PMU_WEIGHT = 0.10
SYNC_PHASE_WEIGHT = 0.10     # applied to phase measurements only
CLOCK_DRIFT_PMU_WEIGHT = 0.20


def measurement_weights_for_action(fault_type: str, pmu: int | None):
    """Return both PMU-level and 12-measurement weights.

    Measurement order:
    [Vmag1,Vang1,Imag1,Iang1, Vmag2,Vang2,Imag2,Iang2, Vmag3,Vang3,Imag3,Iang3]

    For SYNC, only phase measurements are reduced because synchronization
    error primarily corrupts the reported phase reference.
    """
    pmu_weights = [1.0, 1.0, 1.0]
    measurement_weights = [1.0] * 12

    if fault_type == "NORMAL" or pmu is None:
        return pmu_weights, measurement_weights

    idx = pmu - 1
    base = 4 * idx

    if fault_type == "BAD_DATA":
        pmu_weights[idx] = BAD_DATA_PMU_WEIGHT
        measurement_weights[base:base + 4] = [BAD_DATA_PMU_WEIGHT] * 4

    elif fault_type == "SYNC":
        # V phase and I phase only.
        measurement_weights[base + 1] = SYNC_PHASE_WEIGHT
        measurement_weights[base + 3] = SYNC_PHASE_WEIGHT

    elif fault_type == "CLOCK_DRIFT":
        pmu_weights[idx] = CLOCK_DRIFT_PMU_WEIGHT
        measurement_weights[base:base + 4] = [CLOCK_DRIFT_PMU_WEIGHT] * 4

    else:
        pmu_weights[idx] = 0.0
        measurement_weights[base:base + 4] = [0.0] * 4

    return pmu_weights, measurement_weights


def raw_management_action(fault_type: str, pmu: int | None, confidence: float):
    """Convert one neural prediction into a candidate management action."""
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "state": "HOLD",
            "action": "HOLD / REQUEST MORE DATA",
            "pmu_weights": [1.0, 1.0, 1.0],
            "measurement_weights": [1.0] * 12,
        }

    if fault_type == "NORMAL":
        return {
            "state": "NORMAL",
            "action": "ACCEPT ALL PMUs",
            "pmu_weights": [1.0, 1.0, 1.0],
            "measurement_weights": [1.0] * 12,
        }

    if pmu not in (1, 2, 3):
        return {
            "state": "HOLD",
            "action": "HOLD / REVIEW",
            "pmu_weights": [1.0, 1.0, 1.0],
            "measurement_weights": [1.0] * 12,
        }

    pmu_weights, measurement_weights = measurement_weights_for_action(fault_type, pmu)

    if fault_type == "BAD_DATA":
        action = f"DOWN-WEIGHT PMU{pmu}"
    elif fault_type == "SYNC":
        action = f"DOWN-WEIGHT PHASE DATA OF PMU{pmu}"
    elif fault_type == "CLOCK_DRIFT":
        action = f"DOWN-WEIGHT PMU{pmu} AND APPLY TIMING CHECK"
    else:
        action = f"ISOLATE PMU{pmu}"

    return {
        "state": "FAULT",
        "action": action,
        "pmu_weights": pmu_weights,
        "measurement_weights": measurement_weights,
    }


class TemporalManager:
    """Apply persistence/hysteresis to neural management decisions."""

    def __init__(self, activation_windows=ACTIVATION_WINDOWS,
                 recovery_windows=RECOVERY_WINDOWS):
        self.activation_windows = activation_windows
        self.recovery_windows = recovery_windows
        self.active_fault = None
        self.pending_fault = None
        self.pending_count = 0
        self.normal_count = 0

    def update(self, candidate_fault, candidate_pmu, confidence):
        candidate = None if candidate_fault == "NORMAL" else (
            candidate_fault, candidate_pmu
        )

        # Low-confidence inference does not change the active state.
        if confidence < CONFIDENCE_THRESHOLD:
            self.pending_fault = None
            self.pending_count = 0
            self.normal_count = 0
            if self.active_fault is None:
                return "HOLD", None
            return "ACTIVE", self.active_fault

        # Normal prediction: require consecutive normal windows before recovery.
        if candidate is None:
            self.pending_fault = None
            self.pending_count = 0

            if self.active_fault is None:
                return "NORMAL", None

            self.normal_count += 1
            if self.normal_count >= self.recovery_windows:
                self.active_fault = None
                self.normal_count = 0
                return "NORMAL", None

            return "ACTIVE", self.active_fault

        # A fault prediction resets recovery.
        self.normal_count = 0

        if self.active_fault == candidate:
            self.pending_fault = None
            self.pending_count = 0
            return "ACTIVE", self.active_fault

        if self.pending_fault == candidate:
            self.pending_count += 1
        else:
            self.pending_fault = candidate
            self.pending_count = 1

        if self.pending_count >= self.activation_windows:
            self.active_fault = candidate
            self.pending_fault = None
            self.pending_count = 0
            return "ACTIVE", self.active_fault

        return "HOLD", self.active_fault


def load(path: str):
    return joblib.load(path)


def predict_window(window: pd.DataFrame, bundle: dict) -> dict:
    """Run the two neural models for one 20-sample PDC window."""
    f = extract_window_features(window)
    X = pd.DataFrame(
        [[f[n] for n in bundle["feature_names"]]],
        columns=bundle["feature_names"],
    )

    type_model = bundle["type_model"]
    type_probs = type_model.predict_proba(X)[0]
    ti = int(np.argmax(type_probs))
    fault_type = str(type_model.named_steps["mlp"].classes_[ti])
    type_conf = float(type_probs[ti])

    if fault_type == "NORMAL":
        faulty_pmu = None
        pmu_conf = 1.0
    else:
        pmu_model = bundle["pmu_model"]
        pmu_probs = pmu_model.predict_proba(X)[0]
        pi = int(np.argmax(pmu_probs))
        faulty_pmu = str(pmu_model.named_steps["mlp"].classes_[pi])
        pmu_conf = float(pmu_probs[pi])

    confidence = min(type_conf, pmu_conf)

    return {
        "fault_type": fault_type,
        "faulty_pmu": faulty_pmu,
        "confidence": confidence,
        "type_confidence": type_conf,
        "pmu_confidence": pmu_conf,
    }


def scan_csv(
    csv_path: str,
    model_path: str,
    output_path: str = "Neural_Active_PDC_Results.csv",
    activation_windows: int = ACTIVATION_WINDOWS,
    recovery_windows: int = RECOVERY_WINDOWS,
):
    """Run neural inference and stateful active management over a CSV."""
    bundle = load(model_path)
    df = pd.read_csv(csv_path)

    required = []
    for pmu in (1, 2, 3):
        required += [
            f"PMU{pmu} Voltage Magnitude",
            f"PMU{pmu} Voltage Phase",
            f"PMU{pmu} Current Magnitude",
            f"PMU{pmu} Current Phase",
        ]

    manager = TemporalManager(activation_windows, recovery_windows)
    rows = []

    for end in range(WINDOW - 1, len(df), WINDOW):
        window = df.iloc[end - WINDOW + 1:end + 1]

        if len(window) != WINDOW or window[required].isna().any().any():
            continue

        pred = predict_window(window, bundle)

        raw_type = pred["fault_type"]
        raw_pmu = pred["faulty_pmu"]
        confidence = pred["confidence"]

        # Candidate PMU as integer for management.
        candidate_pmu = (
            int(raw_pmu[-1]) if raw_pmu is not None else None
        )

        state, active_fault = manager.update(
            raw_type, candidate_pmu, confidence
        )

        if state == "ACTIVE" and active_fault is not None:
            active_type, active_pmu = active_fault
            action_data = raw_management_action(
                active_type, active_pmu, max(confidence, CONFIDENCE_THRESHOLD)
            )
        elif state == "NORMAL":
            action_data = raw_management_action("NORMAL", None, 1.0)
        else:
            action_data = {
                "state": "HOLD",
                "action": "HOLD / REQUEST MORE DATA",
                "pmu_weights": (
                    measurement_weights_for_action(
                        active_fault[0], active_fault[1]
                    )[0]
                    if active_fault is not None
                    else [1.0, 1.0, 1.0]
                ),
                "measurement_weights": (
                    measurement_weights_for_action(
                        active_fault[0], active_fault[1]
                    )[1]
                    if active_fault is not None
                    else [1.0] * 12
                ),
            }

        rows.append({
            "time_s": float(window["Time (s)"].iloc[-1]),
            "raw_fault_type": raw_type,
            "raw_faulty_pmu": raw_pmu or "NONE",
            "type_confidence": pred["type_confidence"],
            "pmu_confidence": pred["pmu_confidence"],
            "neural_confidence": confidence,
            "management_state": state,
            "active_fault_type": (
                active_fault[0] if active_fault else "NONE"
            ),
            "active_faulty_pmu": (
                f"PMU{active_fault[1]}" if active_fault else "NONE"
            ),
            "management_action": action_data["action"],
            "pmu_weights": json.dumps(action_data["pmu_weights"]),
            "measurement_weights": json.dumps(
                action_data["measurement_weights"]
            ),
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Neural active fault management controller"
    )
    ap.add_argument("csv")
    ap.add_argument("model")
    ap.add_argument("--output", default="Neural_Active_PDC_Results.csv")
    ap.add_argument("--activation-windows", type=int, default=2)
    ap.add_argument("--recovery-windows", type=int, default=3)
    args = ap.parse_args()

    result = scan_csv(
        args.csv,
        args.model,
        args.output,
        args.activation_windows,
        args.recovery_windows,
    )
    print(result.to_string(index=False))
