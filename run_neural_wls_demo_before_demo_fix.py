#!/usr/bin/env python3
import sys
sys.path.insert(0, "neural_controller")

import ast
import numpy as np
import pandas as pd

from state_estimator import StateEstimator
#from neural_controller.multitask_active_controller_v4_2 import predict_window
from neural_controller.wls_neural import NeuralWeightedLeastSquares
import joblib

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
    """
    Convert neural active-management decision into
    12 individual WLS measurement weights.

    Ordering:
      PMU1: Vmag Vphase Imag Iphase
      PMU2: Vmag Vphase Imag Iphase
      PMU3: Vmag Vphase Imag Iphase
    """

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
        # Retain magnitude information.
        # Down-weight phase measurements.
        weights[base + 1] = 0.10
        weights[base + 3] = 0.10

    elif fault_type == "CLOCK_DRIFT":
        weights[base:base + 4] = 0.10

    return weights


# def get_neural_prediction(csv_path, bundle):
#     """
#     Run the V4.2 controller on the final 20-sample PDC window.
#     """

#     df = pd.read_csv(csv_path)

#     window = df.iloc[-20:].copy()

#     # pred = predict_window(window, bundle)
#     pred = predict_window(window, history, bundle, baseline)

#     return pred

def get_neural_prediction(csv_path, bundle):
    df = pd.read_csv(csv_path).reset_index(drop=True)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PMU measurement columns: {missing}")

    baseline = _baseline_from_history(
        df.iloc[:min(BASELINE_SAMPLES, len(df))]
    )

    first_end = max(WINDOW - 1, TIMING_LONG - 1)

    # Use the first valid V4.2 inference window.
    end = first_end

    window = df.iloc[end-WINDOW+1:end+1]

    if len(window) != WINDOW:
        raise ValueError("Insufficient samples for V4.2 inference window.")

    hs = max(0, end - TIMING_LONG + 1)
    history = df.iloc[hs:end+1]

    prediction = predict_window(
        window,
        history,
        bundle,
        baseline
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
    # Neural controller
    # ------------------------------------------------------------

    prediction = get_neural_prediction(csv_path, bundle)

    print("\nNEURAL CONTROLLER")
    print("-" * 78)

    print(f"Fault type       : {prediction['fault_type']}")
    print(f"Faulty PMU       : {prediction['faulty_pmu']}")
    print(f"Type confidence  : {prediction['type_confidence']:.4f}")
    print(f"PMU confidence   : {prediction['pmu_confidence']:.4f}")
    print(f"Neural confidence: {prediction['neural_confidence']:.4f}")

    print(f"Management state : {prediction['management_state']}")
    print(f"Action           : {prediction['management_action']}")

    # ------------------------------------------------------------
    # Convert neural decision into WLS weights
    # ------------------------------------------------------------

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
    # ------------------------------------------------------------

    est = StateEstimator(csv_path, sample_index=len(pd.read_csv(csv_path)) - 1)

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
        run_case(case_name, csv_path, bundle)

    print("\n")
    print("=" * 78)
    print(" ALL FOUR NEURAL → WLS CASES COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
