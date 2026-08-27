#!/usr/bin/env python3

import sys
sys.path.insert(0, "neural_controller")

import numpy as np
import pandas as pd
import joblib

from state_estimator import StateEstimator
from neural_controller.wls_neural import NeuralWeightedLeastSquares

from neural_controller.multitask_active_controller_v4_2_final import (
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


# ---------------------------------------------------------------------------
# Convert neural management decision into individual WLS measurement weights
# ---------------------------------------------------------------------------

def measurement_weights_from_action(fault_type, faulty_pmu):

    weights = np.ones(12, dtype=float)

    if faulty_pmu == "NONE":
        return weights

    try:
        pmu = int(str(faulty_pmu)[-1])
    except Exception:
        return weights

    base = (pmu - 1) * 4

    if fault_type == "BAD_DATA":
        # Vmag, Vangle, Imag, Iangle
        weights[base:base + 4] = 0.10

    elif fault_type == "CLOCK_DRIFT":
        # Timing problem: down-weight the complete PMU
        weights[base:base + 4] = 0.10

    elif fault_type == "SYNC":
        # Synchronization error primarily affects phase.
        # Retain magnitude measurements.
        weights[base + 1] = 0.10
        weights[base + 3] = 0.10

    return weights


# ---------------------------------------------------------------------------
# Build all valid V4.2 inference windows
# ---------------------------------------------------------------------------

def generate_windows(df):

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


# ---------------------------------------------------------------------------
# Select the strongest representative window
# ---------------------------------------------------------------------------

def select_best_window(df, case_name):

    candidates = list(generate_windows(df))

    if not candidates:
        raise RuntimeError("No valid V4.2 inference windows found.")

    # NORMAL:
    # Select the window with the highest NORMAL confidence.
    if case_name == "NORMAL":

        best = max(
            candidates,
            key=lambda item:
                item[2]["neural_confidence"]
                if item[2]["fault_type"] == "NORMAL"
                else -1.0
        )

        return best

    # Fault cases:
    # Prefer a window where V4.2 actually detected the expected fault.
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

        # Highest confidence among correctly classified windows.
        return max(
            matching,
            key=lambda item: item[2]["neural_confidence"]
        )

    # If the expected fault was never detected,
    # select the strongest non-normal prediction.
    non_normal = [
        item for item in candidates
        if item[2]["fault_type"] != "NORMAL"
    ]

    if non_normal:

        return max(
            non_normal,
            key=lambda item: item[2]["neural_confidence"]
        )

    # Last resort: strongest overall prediction.
    return max(
        candidates,
        key=lambda item: item[2]["neural_confidence"]
    )


# ---------------------------------------------------------------------------
# Display estimated state
# ---------------------------------------------------------------------------

def print_state(x):

    print("\nEstimated bus states:")

    for bus in range(3):

        vm = x[2 * bus]
        angle_deg = np.rad2deg(x[2 * bus + 1])

        print(
            f"  Bus {bus + 1}: "
            f"|V| = {vm:.6f} pu, "
            f"angle = {angle_deg:.6f} deg"
        )


# ---------------------------------------------------------------------------
# Run one complete Neural -> WLS demonstration
# ---------------------------------------------------------------------------

def run_case(case_name, csv_path):

    print("\n")
    print("=" * 78)
    print(f" CASE: {case_name}")
    print("=" * 78)

    print(f"CSV: {csv_path}")

    df = pd.read_csv(csv_path).reset_index(drop=True)

    missing = [c for c in REQUIRED if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing PMU measurement columns: {missing}"
        )

    # ------------------------------------------------------------
    # Automatically locate strongest representative window
    # ------------------------------------------------------------

    end, window, prediction = select_best_window(
        df,
        case_name
    )

    time_s = (
        float(window["Time (s)"].iloc[-1])
        if "Time (s)" in window.columns
        else end / 1000.0
    )

    print("\nSELECTED V4.2 INFERENCE WINDOW")
    print("-" * 78)
    print(f"Sample index     : {end}")
    print(f"Time             : {time_s:.6f} s")
    print(f"PDC window       : {WINDOW} samples")

    # ------------------------------------------------------------
    # Neural controller
    # ------------------------------------------------------------

    print("\nNEURAL CONTROLLER")
    print("-" * 78)

    print(f"Fault type       : {prediction['fault_type']}")
    print(f"Faulty PMU       : {prediction['faulty_pmu']}")
    print(
        f"Type confidence  : "
        f"{prediction['type_confidence']:.4f}"
    )
    print(
        f"Timing confidence: "
        f"{prediction['timing_confidence']:.4f}"
    )
    print(
        f"PMU confidence   : "
        f"{prediction['pmu_confidence']:.4f}"
    )
    print(
        f"Neural confidence: "
        f"{prediction['neural_confidence']:.4f}"
    )

    print(
        f"Management state : "
        f"{prediction['management_state']}"
    )

    print(
        f"Action           : "
        f"{prediction['management_action']}"
    )

    # ------------------------------------------------------------
    # Neural -> WLS
    # ------------------------------------------------------------

    weights = measurement_weights_from_action(
        prediction["fault_type"],
        prediction["faulty_pmu"]
    )

    print("\nNEURAL → WLS WEIGHTS")
    print("-" * 78)

    print("Measurement order:")
    print("  [Vmag, Vangle, Imag, Iangle] per PMU")

    print()
    print("PMU1 :", weights[0:4].tolist())
    print("PMU2 :", weights[4:8].tolist())
    print("PMU3 :", weights[8:12].tolist())

    # ------------------------------------------------------------
    # State estimator
    #
    # IMPORTANT:
    # Use the SAME sample selected by the neural controller.
    # ------------------------------------------------------------

    est = StateEstimator(
        csv_path,
        sample_index=end
    )

    z = est.build_measurement_vector()
    x0 = est.initialize_state()

    print("\nSTATE ESTIMATION")
    print("-" * 78)

    print(
        f"Measurement dimension: {len(z)}"
    )

    print(
        f"State dimension      : {len(x0)}"
    )

    # ------------------------------------------------------------
    # Neural-weighted WLS
    # ------------------------------------------------------------

    solver = NeuralWeightedLeastSquares(
        tolerance=1e-6,
        max_iterations=50
    )

    x, residual, W = solver.solve(
        z,
        x0,
        measurement_weights=weights
    )

    print_state(x)

    # ------------------------------------------------------------
    # WLS result
    # ------------------------------------------------------------

    print("\nWLS RESULT")
    print("-" * 78)

    print(
        f"Residual norm : "
        f"{np.linalg.norm(residual):.8f}"
    )

    print("\nWLS diagonal:")
    print(np.diag(W))

    print("\n" + "=" * 78)
    print(f" COMPLETED: {case_name}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    global BUNDLE

    print("=" * 78)
    print(" NEURAL ACTIVE FAULT MANAGEMENT + WLS DEMONSTRATION")
    print(" V4.2 — AUTOMATIC FAULT-WINDOW SELECTION")
    print("=" * 78)

    print("\nLoading model:")
    print(MODEL)

    BUNDLE = joblib.load(MODEL)

    print("Model loaded successfully.")

    for case_name, csv_path in CASES:

        run_case(
            case_name,
            csv_path
        )

    print("\n")
    print("=" * 78)
    print(" ALL FOUR NEURAL → WLS CASES COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()