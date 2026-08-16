#!/usr/bin/env python3
"""
===========================================================
run_fused_topology_demo.py

Complete Topology Demonstration

Demonstrates the full decision fusion topology:

    PMU DATA
        │
    ┌───┴───┐
    ▼       ▼
  Neural   WLS +
  Stream   χ² Test
    │       │
    └───┬───┘
        ▼
    Decision Fusion
        │
    ┌───┼───┐
    ▼   ▼   ▼
  AGREE DISAGREE NORMAL

Shows agreement/disagreement patterns between:
- Neural fault detection
- Classical statistical (χ²) bad data detection
===========================================================
"""

import sys
sys.path.insert(0, "neural_controller")

import numpy as np
import pandas as pd
import joblib

from state_estimator import StateEstimator
from wls import WeightedLeastSquares
from decision_fusion import DecisionFusion
from neural_controller.wls_neural import NeuralWeightedLeastSquares

from neural_controller.multitask_active_controller_v4_2 import (
    predict_window,
    _baseline_from_history,
    WINDOW,
    TIMING_LONG,
    BASELINE_SAMPLES,
    REQUIRED,
)


MODEL = "neural_controller/neural_active_controller_v42.joblib"

CASES = [
    ("NORMAL", "scenario_data/normal_r03.csv"),
    ("BAD_DATA - PMU2", "scenario_data/PMU2_bad_data_r03.csv"),
    ("SYNC - PMU2", "scenario_data/PMU2_sync_r03.csv"),
    ("CLOCK_DRIFT - PMU2", "scenario_data/PMU2_clock_drift_r03.csv"),
]


def measurement_weights_from_action(fault_type, faulty_pmu):
    """Convert neural management decision into WLS measurement weights."""
    weights = np.ones(12, dtype=float)

    if faulty_pmu == "NONE":
        return weights

    try:
        pmu = int(str(faulty_pmu)[-1])
    except Exception:
        return weights

    base = (pmu - 1) * 4

    if fault_type == "BAD_DATA":
        weights[base:base + 4] = 0.10
    elif fault_type == "CLOCK_DRIFT":
        weights[base:base + 4] = 0.10
    elif fault_type == "SYNC":
        weights[base + 1] = 0.10
        weights[base + 3] = 0.10

    return weights


def generate_windows(df):
    """Build all valid V4.2 inference windows."""
    baseline = _baseline_from_history(
        df.iloc[:min(BASELINE_SAMPLES, len(df))]
    )

    first_end = max(WINDOW - 1, TIMING_LONG - 1)

    for end in range(first_end, len(df), WINDOW):
        window = df.iloc[end - WINDOW + 1:end + 1]

        if len(window) != WINDOW:
            continue

        if window[REQUIRED].isna().any().any():
            continue

        hs = max(0, end - TIMING_LONG + 1)
        history = df.iloc[hs:end + 1]

        prediction = predict_window(
            window,
            history,
            BUNDLE,
            baseline
        )

        yield end, window, prediction


def select_best_window(df, case_name):
    """Select the strongest representative window."""
    candidates = list(generate_windows(df))

    if not candidates:
        raise RuntimeError("No valid V4.2 inference windows found.")

    if case_name == "NORMAL":
        best = max(
            candidates,
            key=lambda item:
                item[2]["neural_confidence"]
                if item[2]["fault_type"] == "NORMAL"
                else -1.0
        )
        return best

    expected_fault = {
        "BAD_DATA - PMU2": "BAD_DATA",
        "SYNC - PMU2": "SYNC",
        "CLOCK_DRIFT - PMU2": "CLOCK_DRIFT",
    }[case_name]

    matching = [
        item for item in candidates
        if item[2]["fault_type"] == expected_fault
    ]

    if matching:
        return max(
            matching,
            key=lambda item: item[2]["neural_confidence"]
        )

    non_normal = [
        item for item in candidates
        if item[2]["fault_type"] != "NORMAL"
    ]

    if non_normal:
        return max(
            non_normal,
            key=lambda item: item[2]["neural_confidence"]
        )

    return max(
        candidates,
        key=lambda item: item[2]["neural_confidence"]
    )


def print_state(x):
    """Display estimated bus state."""
    print("\nEstimated bus states:")
    for bus in range(3):
        vm = x[2 * bus]
        angle_deg = np.rad2deg(x[2 * bus + 1])
        print(
            f"  Bus {bus + 1}: "
            f"|V| = {vm:.6f} pu, "
            f"angle = {angle_deg:.6f} deg"
        )


def run_case(case_name, csv_path):
    """Run one complete demonstration case."""
    print("\n")
    print("=" * 80)
    print(f" CASE: {case_name}")
    print("=" * 80)

    print(f"CSV: {csv_path}")

    df = pd.read_csv(csv_path).reset_index(drop=True)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PMU measurement columns: {missing}")

    # ────────────────────────────────────────────────────────────────
    # Select strongest representative window
    # ────────────────────────────────────────────────────────────────

    end, window, prediction = select_best_window(df, case_name)

    time_s = (
        float(window["Time (s)"].iloc[-1])
        if "Time (s)" in window.columns
        else end / 1000.0
    )

    print("\nDEMONSTRATION / REPRESENTATIVE WINDOW")
    print("-" * 80)
    print(f"Sample index     : {end}")
    print(f"Time             : {time_s:.6f} s")
    print(f"PDC window       : {WINDOW} samples")
    print("Note: this window is selected for demonstration; it is not a formal evaluation protocol.")

    # ────────────────────────────────────────────────────────────────
    # NEURAL STREAM
    # ────────────────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print(" NEURAL STREAM")
    print("=" * 80)

    print(f"Fault type       : {prediction['fault_type']}")
    print(f"Faulty PMU       : {prediction['faulty_pmu']}")
    print(f"Neural confidence: {prediction['neural_confidence']:.4f}")

    # ────────────────────────────────────────────────────────────────
    # CLASSICAL STREAM (INDependent ordinary WLS + χ²)
    # ────────────────────────────────────────────────────────────────

    est = StateEstimator(csv_path, sample_index=end)
    z = est.build_measurement_vector()
    x0 = est.initialize_state()

    solver = WeightedLeastSquares(tolerance=1e-6, max_iterations=50)
    x_classical, residual, W, active_indices, chi2_result = solver.solve(
        z,
        x0,
        bad_data_index=None,
        bad_data_indices=None,
        bad_data_weight=0.1,
    )

    print("\n" + "=" * 80)
    print(" CLASSICAL STREAM")
    print("=" * 80)
    print("Method             : Ordinary WLS")
    print(f"Measurement count  : {len(z)}")
    print(f"State count        : {len(x0)}")
    print(f"Degrees of freedom : {chi2_result['degrees_of_freedom']}")
    print(f"Chi-square statistic: {chi2_result['test_statistic']:.8f}")
    print(f"Critical value     : {chi2_result['critical_value']:.8f}")
    print(f"P-value            : {chi2_result['p_value']:.8f}")
    print(f"Decision           : {'FAIL' if chi2_result['bad_data_detected'] else 'PASS'}")
    print(f"Status             : {'STATISTICALLY INCONSISTENT' if chi2_result['bad_data_detected'] else 'CONSISTENT'}")

    print("\nEstimated State (ordinary WLS):")
    print_state(x_classical)

    # ────────────────────────────────────────────────────────────────
    # DECISION FUSION (after both streams have completed)
    # ────────────────────────────────────────────────────────────────

    neural_result = {
        "fault_type": prediction["fault_type"],
        "faulty_pmu": prediction["faulty_pmu"],
        "confidence": float(prediction.get("neural_confidence", 0.0)),
    }

    fuser = DecisionFusion(
        neural_confidence_threshold=0.70,
        chi2_significance_level=0.05,
    )
    fusion_decision = fuser.fuse(neural_result, chi2_result)

    print("\n" + "=" * 80)
    print(" DECISION FUSION")
    print("=" * 80)
    print(f"Neural decision     : {neural_result['fault_type']}")
    print(f"Classical decision  : {'ANOMALY' if chi2_result['bad_data_detected'] else 'NORMAL'}")
    print(f"Agreement           : {'YES' if fusion_decision['agreement'] else 'NO'}")
    print(f"Final diagnosis     : {fusion_decision['decision_name']}")
    print(f"Fusion action       : {fusion_decision['action']}")
    print(f"Reason              : {fusion_decision['reasoning']}")

    # ────────────────────────────────────────────────────────────────
    # NEURAL WLS MITIGATION (post-fusion)
    # ────────────────────────────────────────────────────────────────

    mitigation_weights = measurement_weights_from_action(
        prediction["fault_type"],
        prediction["faulty_pmu"],
    )

    if fusion_decision["decision_name"] == "AGREEMENT":
        mitigation_weights = measurement_weights_from_action(
            prediction["fault_type"],
            prediction["faulty_pmu"],
        )
    elif prediction["fault_type"] == "NORMAL":
        mitigation_weights = np.ones(12, dtype=float)
    else:
        mitigation_weights = np.ones(12, dtype=float)

    print("\n" + "=" * 80)
    print(" NEURAL WLS MITIGATION")
    print("=" * 80)
    print("PMU1 weights :", mitigation_weights[0:4].tolist())
    print("PMU2 weights :", mitigation_weights[4:8].tolist())
    print("PMU3 weights :", mitigation_weights[8:12].tolist())

    mitigated_solver = NeuralWeightedLeastSquares(tolerance=1e-6, max_iterations=50)
    x_mitigated, residual_after, _ = mitigated_solver.solve(
        z,
        x0,
        measurement_weights=mitigation_weights,
    )

    residual_norm_after = float(np.linalg.norm(residual_after))
    print(f"Residual norm after mitigation : {residual_norm_after:.8f}")
    print("Post-mitigation state estimate:")
    print_state(x_mitigated)

    print("\n" + "=" * 80)
    print(f" COMPLETED: {case_name}")
    print("=" * 80)

    return {
        "case": case_name,
        "neural_result": neural_result,
        "classical_result": chi2_result,
        "fusion_decision": fusion_decision,
        "mitigation_weights": mitigation_weights,
        "residual_norm_after": residual_norm_after,
    }

    # ────────────────────────────────────────────────────────────────
    # DECISION FUSION
    # ────────────────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print(" DECISION FUSION")
    print("=" * 80)

    fuser = DecisionFusion(
        neural_confidence_threshold=0.70,
        chi2_significance_level=0.05
    )

    # Prepare neural result for fusion
    neural_result = {
        "fault_type": prediction["fault_type"],
        "confidence": prediction["neural_confidence"],
        "faulty_pmu": prediction["faulty_pmu"],
    }

    # Perform fusion
    fusion_decision = fuser.fuse(neural_result, chi2_result)

    # Print fusion report
    fuser.print_fusion_report(fusion_decision)

    print("\n" + "=" * 80)
    print(f" COMPLETED: {case_name}")
    print("=" * 80)

    return {
        "case": case_name,
        "neural_result": neural_result,
        "chi2_result": chi2_result,
        "fusion_decision": fusion_decision,
    }


def main():
    """Run all demonstration cases."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + " FUSED TOPOLOGY DEMONSTRATION: Neural + Classical + Decision Fusion".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # Load neural model
    global BUNDLE
    BUNDLE = joblib.load(MODEL)

    results = []
    for case_name, csv_path in CASES:
        try:
            result = run_case(case_name, csv_path)
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR in case '{case_name}':")
            print(f"   {e}")
            import traceback
            traceback.print_exc()

    # ────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " TOPOLOGY SUMMARY".center(78) + "║")
    print("╚" + "=" * 78 + "╝")

    for result in results:
        case = result["case"]
        decision = result["fusion_decision"]["decision_name"]
        agreement = "✓" if result["fusion_decision"]["agreement"] else "✗"

        print(f"\n{case:25s} → {decision:15s} ({agreement})")


if __name__ == "__main__":
    main()
