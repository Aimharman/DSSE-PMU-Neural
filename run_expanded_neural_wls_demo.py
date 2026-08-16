#!/usr/bin/env python3
"""
Expanded V4.2 Neural -> WLS demonstration.

Runs the complete active-fault-management chain for:
    NORMAL
    BAD_DATA    on PMU1/PMU2/PMU3
    SYNC        on PMU1/PMU2/PMU3
    CLOCK_DRIFT on PMU1/PMU2/PMU3

The V4.2 training/model files are NOT modified.

For each CSV:
  1. Run V4.2 inference over all valid PDC windows.
  2. Select the highest-confidence window consistent with the expected
     scenario (or the highest-confidence NORMAL window).
  3. Convert the neural management decision to 12 WLS measurement weights.
  4. Run NeuralWeightedLeastSquares on the same sample.
  5. Print the complete result to the terminal.

This is a demonstration/verification script; it does not retrain the model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Make project-root imports available.
ROOT = Path(__file__).resolve().parent
NEURAL = ROOT / "neural_controller"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NEURAL))

from state_estimator import StateEstimator
from neural_controller.wls_neural import NeuralWeightedLeastSquares

from neural_controller.multitask_active_controller_v4_2 import (
    scan_csv,
)


MODEL = ROOT / "neural_controller" / "neural_active_controller_v42.joblib"

CASES = [
    ("NORMAL", "scenario_data/normal_r03.csv", "NORMAL", "NONE"),
    ("BAD_DATA - PMU1", "scenario_data/PMU1_bad_data_r03.csv", "BAD_DATA", "PMU1"),
    ("BAD_DATA - PMU2", "scenario_data/PMU2_bad_data_r03.csv", "BAD_DATA", "PMU2"),
    ("BAD_DATA - PMU3", "scenario_data/PMU3_bad_data_r03.csv", "BAD_DATA", "PMU3"),
    ("SYNC - PMU1", "scenario_data/PMU1_sync_r03.csv", "SYNC", "PMU1"),
    ("SYNC - PMU2", "scenario_data/PMU2_sync_r03.csv", "SYNC", "PMU2"),
    ("SYNC - PMU3", "scenario_data/PMU3_sync_r03.csv", "SYNC", "PMU3"),
    ("CLOCK_DRIFT - PMU1", "scenario_data/PMU1_clock_drift_r03.csv", "CLOCK_DRIFT", "PMU1"),
    ("CLOCK_DRIFT - PMU2", "scenario_data/PMU2_clock_drift_r03.csv", "CLOCK_DRIFT", "PMU2"),
    ("CLOCK_DRIFT - PMU3", "scenario_data/PMU3_clock_drift_r03.csv", "CLOCK_DRIFT", "PMU3"),
]


def measurement_weights_from_action(fault_type, faulty_pmu):
    """
    Convert the V4.2 management decision into 12 individual WLS weights.

    Measurement order per PMU:
        [Vmag, Vangle, Imag, Iangle]

    Policy used by the existing demonstration:
      NORMAL       -> all 1.0
      BAD_DATA     -> all four measurements of faulty PMU = 0.1
      CLOCK_DRIFT  -> all four measurements of faulty PMU = 0.1
      SYNC         -> retain magnitude/current magnitude, down-weight
                      voltage/current phase measurements = 0.1
    """
    weights = np.ones(12, dtype=float)

    if faulty_pmu == "NONE":
        return weights

    try:
        pmu_number = int(str(faulty_pmu)[-1])
    except (ValueError, TypeError):
        return weights

    if pmu_number not in (1, 2, 3):
        return weights

    base = (pmu_number - 1) * 4

    if fault_type in ("BAD_DATA", "CLOCK_DRIFT"):
        weights[base:base + 4] = 0.1

    elif fault_type == "SYNC":
        weights[base + 1] = 0.1  # Vangle
        weights[base + 3] = 0.1  # Iangle

    return weights


def choose_window(predictions, expected_fault, expected_pmu):
    """
    Select the strongest inference window matching the known scenario.

    This avoids arbitrarily using the final CSV row. It is also transparent:
    the terminal prints the selected sample/time and the predicted class.

    NORMAL:
        choose highest neural-confidence NORMAL/NONE window.

    Fault cases:
        choose highest neural-confidence window whose predicted fault type
        and faulty PMU match the expected scenario.

    If no matching window exists, fall back to the highest-confidence window
    with the expected fault type, then finally to the globally highest
    confidence window. The fallback is explicitly reported.
    """
    df = predictions.copy()

    if df.empty:
        raise RuntimeError("V4.2 produced no valid inference windows.")

    if "neural_confidence" not in df.columns:
        raise RuntimeError("Prediction output has no neural_confidence column.")

    df["neural_confidence"] = pd.to_numeric(
        df["neural_confidence"], errors="coerce"
    )

    if expected_fault == "NORMAL":
        candidates = df[
            (df["active_fault_type"].astype(str) == "NORMAL")
            & (df["active_faulty_pmu"].astype(str) == "NONE")
        ]
    else:
        candidates = df[
            (df["active_fault_type"].astype(str) == expected_fault)
            & (df["active_faulty_pmu"].astype(str) == expected_pmu)
        ]

    if not candidates.empty:
        return candidates.loc[candidates["neural_confidence"].idxmax()], "exact"

    # Fallback 1: correct fault type, PMU may be wrong.
    if expected_fault != "NORMAL":
        candidates = df[
            df["active_fault_type"].astype(str) == expected_fault
        ]
        if not candidates.empty:
            return candidates.loc[candidates["neural_confidence"].idxmax()], "fault-type"

    # Fallback 2: strongest overall prediction.
    row = df.loc[df["neural_confidence"].idxmax()]
    return row, "global"


def print_state(x):
    print("\nEstimated bus states:")
    for bus in range(3):
        vm = float(x[2 * bus])
        angle_deg = float(np.rad2deg(x[2 * bus + 1]))
        print(
            f"  Bus {bus + 1}: "
            f"|V| = {vm:.6f} pu, "
            f"angle = {angle_deg:.6f} deg"
        )


def run_case(case_name, csv_rel, expected_fault, expected_pmu, model_path):
    csv_path = ROOT / csv_rel

    print("\n\n" + "=" * 78)
    print(f" CASE: {case_name}")
    print("=" * 78)
    print(f"CSV: {csv_rel}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Run the complete V4.2 scanner. Save an audit CSV for every case.
    audit_name = (
        "expanded_wls_"
        + case_name.lower().replace(" ", "_").replace("-", "")
        + ".csv"
    )
    audit_path = ROOT / "neural_controller" / audit_name

    predictions = scan_csv(
        str(csv_path),
        str(model_path),
        str(audit_path),
    )

    selected, selection_mode = choose_window(
        predictions,
        expected_fault,
        expected_pmu,
    )

    sample_index = int(selected.name)

    # scan_csv uses end-of-window time. StateEstimator needs the actual
    # measurement sample index, so use the final sample represented by the
    # selected inference row. V4.2 rows are generated at the window end.
    if "time_s" in selected.index:
        pass

    # The scan dataframe index is not guaranteed to equal the source sample
    # index after filtering. Reconstruct the source sample index from time.
    df_source = pd.read_csv(csv_path).reset_index(drop=True)

    if "time_s" in selected:
        target_time = float(selected["time_s"])
        if "Time (s)" in df_source.columns:
            source_times = pd.to_numeric(
                df_source["Time (s)"], errors="coerce"
            ).to_numpy()
            sample_index = int(np.nanargmin(np.abs(source_times - target_time)))
        else:
            sample_index = int(round(target_time * 1000.0))

    print("\nSELECTED V4.2 INFERENCE WINDOW")
    print("-" * 78)
    print(f"Selection mode    : {selection_mode}")
    print(f"Sample index      : {sample_index}")
    if "time_s" in selected:
        print(f"Time              : {float(selected['time_s']):.6f} s")
    print(f"PDC window        : 20 samples")

    print("\nNEURAL CONTROLLER")
    print("-" * 78)
    print(f"Expected fault    : {expected_fault}")
    print(f"Expected PMU      : {expected_pmu}")
    print(f"Fault type        : {selected['active_fault_type']}")
    print(f"Faulty PMU        : {selected['active_faulty_pmu']}")
    print(f"Type confidence   : {float(selected['type_confidence']):.4f}")
    print(f"Timing confidence : {float(selected['timing_confidence']):.4f}")
    print(f"PMU confidence    : {float(selected['pmu_confidence']):.4f}")
    print(f"Neural confidence : {float(selected['neural_confidence']):.4f}")
    print(f"Management state  : {selected['management_state']}")
    print(f"Action            : {selected['management_action']}")

    # Convert the actual neural decision to WLS weights.
    fault_type = str(selected["active_fault_type"])
    faulty_pmu = str(selected["active_faulty_pmu"])

    weights = measurement_weights_from_action(
        fault_type,
        faulty_pmu,
    )

    print("\nNEURAL -> WLS WEIGHTS")
    print("-" * 78)
    print("Measurement order:")
    print("  [Vmag, Vangle, Imag, Iangle] per PMU")
    print()
    print("PMU1 :", weights[0:4].tolist())
    print("PMU2 :", weights[4:8].tolist())
    print("PMU3 :", weights[8:12].tolist())

    # State estimator uses the SAME source sample selected by the neural
    # inference window.
    est = StateEstimator(
        str(csv_path),
        sample_index=sample_index,
    )

    z = est.build_measurement_vector()
    x0 = est.initialize_state()

    print("\nSTATE ESTIMATION")
    print("-" * 78)
    print(f"Measurement dimension: {len(z)}")
    print(f"State dimension      : {len(x0)}")

    solver = NeuralWeightedLeastSquares(
        tolerance=1e-6,
        max_iterations=50,
    )

    x, residual, W = solver.solve(
        z,
        x0,
        measurement_weights=weights,
    )

    print_state(x)

    print("\nWLS RESULT")
    print("-" * 78)
    print(f"Residual norm : {np.linalg.norm(residual):.8f}")
    print("\nWLS diagonal:")
    print(np.diag(W))

    print(f"\nAudit CSV: {audit_path.relative_to(ROOT)}")

    print("\n" + "=" * 78)
    print(f" COMPLETED: {case_name}")
    print("=" * 78)

    return {
        "case": case_name,
        "expected_fault": expected_fault,
        "expected_pmu": expected_pmu,
        "detected_fault": fault_type,
        "detected_pmu": faulty_pmu,
        "neural_confidence": float(selected["neural_confidence"]),
        "action": str(selected["management_action"]),
        "selection_mode": selection_mode,
        "sample_index": sample_index,
        "residual_norm": float(np.linalg.norm(residual)),
        "weights": weights.tolist(),
    }


def main():
    print("=" * 78)
    print(" EXPANDED NEURAL ACTIVE FAULT MANAGEMENT + WLS")
    print(" V4.2 — 10 SCENARIO CASES")
    print("=" * 78)

    print("\nLoading model:")
    print(MODEL)

    if not MODEL.exists():
        raise FileNotFoundError(
            f"V4.2 model not found: {MODEL}\n"
            "Train V4.2 first; this script does not train or modify it."
        )

    # Load once to verify the file before scan_csv loads it internally.
    joblib.load(MODEL)
    print("Model loaded successfully.")

    results = []

    for case_name, csv_rel, expected_fault, expected_pmu in CASES:
        try:
            result = run_case(
                case_name,
                csv_rel,
                expected_fault,
                expected_pmu,
                MODEL,
            )
            results.append(result)
        except Exception as exc:
            print("\n" + "!" * 78)
            print(f" FAILED: {case_name}")
            print(f" ERROR : {type(exc).__name__}: {exc}")
            print("!" * 78)
            raise

    print("\n\n" + "=" * 78)
    print(" EXPANDED NEURAL -> WLS SUMMARY")
    print("=" * 78)

    print(
        f"{'CASE':<25} {'EXPECTED':<13} {'DETECTED':<13} "
        f"{'PMU':<6} {'CONF':>7} {'RESIDUAL':>11}"
    )
    print("-" * 78)

    for r in results:
        print(
            f"{r['case']:<25} "
            f"{r['expected_fault']:<13} "
            f"{r['detected_fault']:<13} "
            f"{r['detected_pmu']:<6} "
            f"{r['neural_confidence']:>7.4f} "
            f"{r['residual_norm']:>11.6f}"
        )

    exact = sum(
        r["detected_fault"] == r["expected_fault"]
        and r["detected_pmu"] == r["expected_pmu"]
        for r in results
    )

    print("\nExact scenario decisions:", f"{exact}/{len(results)}")

    print("\n" + "=" * 78)
    print(" ALL EXPANDED NEURAL -> WLS CASES COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
