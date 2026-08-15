#!/usr/bin/env python3

import sys
sys.path.insert(0, "neural_controller")

import numpy as np
import pandas as pd
import joblib

from state_estimator import StateEstimator
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

    weights = np.ones(12, dtype=float)

    if faulty_pmu == "NONE":
        return weights

    try:
        pmu = int(faulty_pmu[-1])
    except Exception:
        return weights

    base = (pmu - 1) * 4

    if fault_type == "BAD_DATA":
        weights[base:base + 4] = 0.10

    elif fault_type == "SYNC":
        # Retain magnitude measurements.
        # Reduce influence of phase measurements.
        weights[base + 1] = 0.10
        weights[base + 3] = 0.10

    elif fault_type == "CLOCK_DRIFT":
        weights[base:base + 4] = 0.10

    return weights


def select_demo_window(df, case_name):
    """
    Select the final valid PDC window for the demonstration.

    The scenario CSV contains only PMU measurements. Ground-truth fault
    labels are NOT stored in the input CSV and are never supplied to the
    neural controller.

    For the demonstration we use the final valid window because the
    injected fault in the fault scenarios is active by the end of the
    scenario.
    """

    first_end = max(WINDOW - 1, TIMING_LONG - 1)

    if len(df) <= first_end:
        raise RuntimeError(
            f"Insufficient samples for {case_name}: "
            f"{len(df)} samples."
        )

    target = len(df) - 1

    window = df.iloc[target - WINDOW + 1:target + 1].copy()

    if len(window) != WINDOW:
        raise RuntimeError(
            f"Could not construct {WINDOW}-sample PDC window."
        )

    return target, window

def get_neural_prediction(df, target, bundle):

    baseline = _baseline_from_history(
        df.iloc[:min(BASELINE_SAMPLES, len(df))]
    )

    hs = max(0, target - TIMING_LONG + 1)

    history = df.iloc[hs:target + 1].copy()

    prediction = predict_window(
        df.iloc[target - WINDOW + 1:target + 1].copy(),
        history,
        bundle,
        baseline,
    )

    return prediction


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


def run_case(case_name, csv_path, bundle):

    print("\n")
    print("=" * 78)
    print(f" CASE: {case_name}")
    print("=" * 78)

    print(f"CSV: {csv_path}")

    # ------------------------------------------------------------
    # Load complete scenario
    # ------------------------------------------------------------

    df = pd.read_csv(csv_path).reset_index(drop=True)

    missing = [c for c in REQUIRED if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing PMU measurement columns: {missing}"
        )

    # ------------------------------------------------------------
    # Select demonstration point
    # ------------------------------------------------------------

    target, window = select_demo_window(df, case_name)

    time_s = (
        float(df["Time (s)"].iloc[target])
        if "Time (s)" in df.columns
        else target / 1000.0
    )

    print(f"Demo sample index : {target}")
    print(f"Demo time         : {time_s:.6f} s")
    print(f"PDC window        : {WINDOW} samples")

    # ------------------------------------------------------------
    # Neural controller
    # ------------------------------------------------------------

    prediction = get_neural_prediction(
        df,
        target,
        bundle
    )

    print("\nNEURAL CONTROLLER")
    print("-" * 78)

    print(f"Fault type       : {prediction['fault_type']}")
    print(f"Faulty PMU       : {prediction['faulty_pmu']}")
    print(f"Type confidence  : {prediction['type_confidence']:.4f}")
    print(f"Timing confidence: {prediction['timing_confidence']:.4f}")
    print(f"PMU confidence   : {prediction['pmu_confidence']:.4f}")
    print(f"Neural confidence: {prediction['neural_confidence']:.4f}")

    print(f"Management state : {prediction['management_state']}")
    print(f"Action           : {prediction['management_action']}")

    # ------------------------------------------------------------
    # Neural → WLS measurement weights
    # ------------------------------------------------------------

    weights = np.asarray(
        prediction["measurement_weights"],
        dtype=float
    )

    # Safety fallback in case an older controller bundle does not
    # provide measurement_weights.
    if weights.size != 12:
        weights = measurement_weights_from_action(
            prediction["fault_type"],
            prediction["faulty_pmu"]
        )

    print("\nNEURAL → WLS WEIGHTS")
    print("-" * 78)

    print("PMU1 :", weights[0:4].tolist())
    print("PMU2 :", weights[4:8].tolist())
    print("PMU3 :", weights[8:12].tolist())

    # ------------------------------------------------------------
    # State estimator
    #
    # IMPORTANT:
    # Use exactly the same sample that produced the neural decision.
    # ------------------------------------------------------------

    est = StateEstimator(
        csv_path,
        sample_index=target
    )

    z = est.build_measurement_vector()
    x0 = est.initialize_state()

    print("\nSTATE ESTIMATION")
    print("-" * 78)

    print(f"Measurement dimension: {len(z)}")
    print(f"State dimension      : {len(x0)}")

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

    print("\nWLS RESULT")
    print("-" * 78)

    print(f"Residual norm : {np.linalg.norm(residual):.8f}")

    print("\nWLS diagonal:")
    print(np.diag(W))

    print("\n" + "=" * 78)
    print(f" COMPLETED: {case_name}")
    print("=" * 78)


def main():

    print("=" * 78)
    print(" NEURAL ACTIVE FAULT MANAGEMENT + WLS DEMONSTRATION")
    print(" V4.2")
    print("=" * 78)

    print("\nLoading model:")
    print(MODEL)

    bundle = joblib.load(MODEL)

    print("Model loaded successfully.")

    for case_name, csv_path in CASES:
        run_case(
            case_name,
            csv_path,
            bundle
        )

    print("\n")
    print("=" * 78)
    print(" ALL FOUR NEURAL → WLS CASES COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
