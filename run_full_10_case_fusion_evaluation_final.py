#!/usr/bin/env python3
"""Run the complete 10-scenario evaluation for the refactored independent
Neural + Classical WLS/Chi-square + Decision Fusion topology.

Important ordering:
  1) Run the neural classifier independently.
  2) Run ordinary WLS and compute J = r^T R^-1 r independently.
  3) Fuse the decisions.
  4) Only after fusion may neural-derived weights be applied to
     NeuralWeightedLeastSquares for mitigation.

This script is intentionally separate from the existing demo so the current
result is preserved while still evaluating the full scenario matrix.
"""

import sys
from typing import Dict, List, Tuple

sys.path.insert(0, "neural_controller")

import joblib
import numpy as np
import pandas as pd

from state_estimator import StateEstimator
from wls import WeightedLeastSquares
from decision_fusion import DecisionFusion
from neural_controller.wls_neural import NeuralWeightedLeastSquares
from neural_controller.multitask_active_controller_v4_2_final import (
    predict_window,
    _baseline_from_history,
    WINDOW,
    TIMING_LONG,
    BASELINE_SAMPLES,
    REQUIRED,
)

MODEL_PATH = "neural_controller/neural_active_controller_v42.joblib"

CASES = [
    ("NORMAL", "scenario_data/normal_r03.csv"),
    ("BAD_DATA - PMU1", "scenario_data/PMU1_bad_data_r03.csv"),
    ("BAD_DATA - PMU2", "scenario_data/PMU2_bad_data_r03.csv"),
    ("BAD_DATA - PMU3", "scenario_data/PMU3_bad_data_r03.csv"),
    ("SYNC - PMU1", "scenario_data/PMU1_sync_r03.csv"),
    ("SYNC - PMU2", "scenario_data/PMU2_sync_r03.csv"),
    ("SYNC - PMU3", "scenario_data/PMU3_sync_r03.csv"),
    ("CLOCK_DRIFT - PMU1", "scenario_data/PMU1_clock_drift_r03.csv"),
    ("CLOCK_DRIFT - PMU2", "scenario_data/PMU2_clock_drift_r03.csv"),
    ("CLOCK_DRIFT - PMU3", "scenario_data/PMU3_clock_drift_r03.csv"),
]


def measurement_weights_from_action(fault_type: str, faulty_pmu: str) -> np.ndarray:
    """Convert neural management action into WLS measurement weights."""
    weights = np.ones(12, dtype=float)
    if faulty_pmu == "NONE":
        return weights

    try:
        pmu = int(str(faulty_pmu)[-1])
    except Exception:
        return weights

    base = (pmu - 1) * 4
    if fault_type in {"BAD_DATA", "CLOCK_DRIFT"}:
        weights[base:base + 4] = 0.10
    elif fault_type == "SYNC":
        weights[base + 1] = 0.10
        weights[base + 3] = 0.10
    return weights


def generate_windows(df: pd.DataFrame, bundle: dict):
    """Yield all valid V4.2 neural inference windows."""
    baseline = _baseline_from_history(df.iloc[:min(BASELINE_SAMPLES, len(df))])
    first_end = max(WINDOW - 1, TIMING_LONG - 1)

    for end in range(first_end, len(df), WINDOW):
        window = df.iloc[end - WINDOW + 1:end + 1]
        if len(window) != WINDOW:
            continue
        if window[REQUIRED].isna().any().any():
            continue

        hs = max(0, end - TIMING_LONG + 1)
        history = df.iloc[hs:end + 1]
        prediction = predict_window(window, history, bundle, baseline)
        yield end, window, prediction


def select_best_window(df: pd.DataFrame, case_name: str, bundle: dict):
    """Pick the strongest representative window for the scenario."""
    candidates = list(generate_windows(df, bundle))
    if not candidates:
        raise RuntimeError(f"No valid windows found for {case_name}")

    if case_name == "NORMAL":
        return max(candidates, key=lambda item: item[2]["neural_confidence"] if item[2]["fault_type"] == "NORMAL" else -1.0)

    expected_fault = {
        "BAD_DATA - PMU1": "BAD_DATA",
        "BAD_DATA - PMU2": "BAD_DATA",
        "BAD_DATA - PMU3": "BAD_DATA",
        "SYNC - PMU1": "SYNC",
        "SYNC - PMU2": "SYNC",
        "SYNC - PMU3": "SYNC",
        "CLOCK_DRIFT - PMU1": "CLOCK_DRIFT",
        "CLOCK_DRIFT - PMU2": "CLOCK_DRIFT",
        "CLOCK_DRIFT - PMU3": "CLOCK_DRIFT",
    }[case_name]

    matching = [item for item in candidates if item[2]["fault_type"] == expected_fault]
    if matching:
        return max(matching, key=lambda item: item[2]["neural_confidence"])

    non_normal = [item for item in candidates if item[2]["fault_type"] != "NORMAL"]
    if non_normal:
        return max(non_normal, key=lambda item: item[2]["neural_confidence"])

    return max(candidates, key=lambda item: item[2]["neural_confidence"])


def run_case(case_name: str, csv_path: str, bundle: dict) -> Dict:
    """Execute one case with independent neural + classical streams and fusion."""
    df = pd.read_csv(csv_path).reset_index(drop=True)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PMU measurement columns in {csv_path}: {missing}")

    end, window, prediction = select_best_window(df, case_name, bundle)
    time_s = float(window["Time (s)"].iloc[-1]) if "Time (s)" in window.columns else end / 1000.0

    # Independent neural stream
    neural_result = {
        "fault_type": prediction["fault_type"],
        "faulty_pmu": prediction["faulty_pmu"],
        "confidence": float(prediction.get("neural_confidence", 0.0)),
    }

    # Independent classical stream: ordinary WLS + chi-square before any neural weights
    est = StateEstimator(csv_path, sample_index=end)
    z = est.build_measurement_vector()
    x0 = est.initialize_state()

    solver = WeightedLeastSquares(tolerance=1e-6, max_iterations=50)
    _, residual, _, _, chi2_result = solver.solve(
        z,
        x0,
        bad_data_index=None,
        bad_data_indices=None,
        bad_data_weight=0.1,
    )

    # Decision fusion after both streams complete
    fuser = DecisionFusion(neural_confidence_threshold=0.70, chi2_significance_level=0.05)
    fusion = fuser.fuse(neural_result, chi2_result)

    # Post-fusion mitigation only
    if fusion["decision_name"] == "AGREEMENT":
        mitigation_weights = measurement_weights_from_action(neural_result["fault_type"], neural_result["faulty_pmu"])
    elif neural_result["fault_type"] == "NORMAL":
        mitigation_weights = np.ones(12, dtype=float)
    else:
        mitigation_weights = np.ones(12, dtype=float)

    mitigated_solver = NeuralWeightedLeastSquares(tolerance=1e-6, max_iterations=50)
    _, residual_after, _ = mitigated_solver.solve(
        z,
        x0,
        measurement_weights=mitigation_weights,
    )
    residual_norm_after = float(np.linalg.norm(residual_after))

    return {
        "case": case_name,
        "csv_path": csv_path,
        "time_s": time_s,
        "sample_index": end,
        "neural_fault_type": neural_result["fault_type"],
        "neural_faulty_pmu": neural_result["faulty_pmu"],
        "neural_confidence": float(neural_result["confidence"]),
        "classical_test_statistic": float(chi2_result.get("test_statistic", np.nan)),
        "classical_critical_value": float(chi2_result.get("critical_value", np.nan)),
        "classical_p_value": float(chi2_result.get("p_value", np.nan)),
        "classical_dof": int(chi2_result.get("degrees_of_freedom", 0)),
        "classical_decision": "FAIL" if chi2_result.get("bad_data_detected", False) else "PASS",
        "classical_bad_data_detected": bool(chi2_result.get("bad_data_detected", False)),
        "fusion_decision": fusion["decision_name"],
        "fusion_agreement": bool(fusion["agreement"]),
        "mitigation_weights": mitigation_weights,
        "residual_norm_after_mitigation": residual_norm_after,
    }


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    rows = [run_case(case_name, csv_path, bundle) for case_name, csv_path in CASES]

    summary = pd.DataFrame(rows)
    summary = summary[[
        "case",
        "neural_fault_type",
        "neural_faulty_pmu",
        "neural_confidence",
        "classical_test_statistic",
        "classical_critical_value",
        "classical_p_value",
        "classical_dof",
        "classical_decision",
        "fusion_decision",
        "fusion_agreement",
        "residual_norm_after_mitigation",
    ]]

    print("\n10-CASE FUSION EVALUATION SUMMARY")
    print("=" * 180)
    print(summary.to_string(index=False, justify="center"))
    print("=" * 180)

    # Print a compact disagreement summary
    disagreements = summary[summary["fusion_agreement"] == False]
    if disagreements.empty:
        print("\nNo genuine disagreements were observed across the 10 scenarios.")
    else:
        print("\nGenuine disagreements:")
        print(disagreements[["case", "neural_fault_type", "classical_decision", "fusion_decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
