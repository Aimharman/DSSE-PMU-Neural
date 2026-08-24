from __future__ import annotations

import numpy as np
import pandas as pd

from .timing_features import compute_timing_features


DEFAULT_METADATA_COLUMNS = {
    "PMU1 Sync Fault",
    "PMU1 Sync Fault Active",
    "PMU1 Clock Drift",
    "PMU1 Clock Drift Fault",
    "PMU1 Packet Loss",
    "PMU1 Bad Data",
    "PMU2 Sync Fault",
    "PMU2 Sync Fault Active",
    "PMU2 Clock Drift",
    "PMU2 Clock Drift Fault",
    "PMU2 Packet Loss",
    "PMU2 Bad Data",
    "PMU3 Sync Fault",
    "PMU3 Sync Fault Active",
    "PMU3 Clock Drift",
    "PMU3 Clock Drift Fault",
    "PMU3 Packet Loss",
    "PMU3 Bad Data",
}


def _phase_vector_stats(values):
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(5, dtype=float)
    arr = arr[finite]
    if arr.size <= 1:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    slope = 0.0
    try:
        slope = float(abs(np.polyfit(np.arange(arr.size), arr, 1)[0]))
    except np.linalg.LinAlgError:
        slope = 0.0
    return np.array([
        float(np.mean(np.abs(arr))),
        slope,
        float(np.var(arr)),
        float(np.max(np.abs(np.diff(arr)))),
        float(np.mean(np.abs(np.diff(arr)))),
    ], dtype=float)


def extract_window_features(window_df):
    """Compute a richer phase-aware feature vector retaining PMU and pairwise dynamics."""
    if window_df is None or window_df.empty:
        return np.zeros(30, dtype=float)

    feature_df = window_df.copy()
    for col in DEFAULT_METADATA_COLUMNS:
        if col in feature_df.columns:
            feature_df = feature_df.drop(columns=[col])

    timing = compute_timing_features(feature_df)
    vector = [
        float(timing["offset"]),
        float(timing["short_term_slope"]),
        float(timing["long_term_slope"]),
        float(timing["variance"]),
        float(timing["step_change"]),
        float(timing["persistence"]),
        float(timing["delta_phi_mean"]),
    ]

    pair_names = [("12", "pair12"), ("13", "pair13"), ("23", "pair23")]
    for _, pair_key in pair_names:
        vector.extend([
            float(timing[f"{pair_key}_offset"]),
            float(timing[f"{pair_key}_slope"]),
            float(timing[f"{pair_key}_variance"]),
            float(timing[f"{pair_key}_step"]),
            float(timing[f"{pair_key}_persistence"]),
        ])

    time = pd.to_numeric(feature_df.get("Time (s)", pd.Series(np.arange(len(feature_df)))), errors="coerce").to_numpy(dtype=float)
    if np.all(~np.isfinite(time)):
        time = np.arange(len(feature_df), dtype=float)
    for pmu in (1, 2, 3):
        phase_col = f"PMU{pmu} Voltage Phase"
        mag_col = f"PMU{pmu} Voltage Magnitude"
        phase_stats = np.zeros(5, dtype=float)
        mag_stats = np.zeros(5, dtype=float)
        if phase_col in feature_df.columns:
            phase_stats = _phase_vector_stats(feature_df[phase_col])
        if mag_col in feature_df.columns:
            mag_stats = _phase_vector_stats(feature_df[mag_col])
        vector.extend(np.concatenate([phase_stats, mag_stats]).tolist())

    return np.asarray(vector, dtype=float)


def build_feature_matrix(df):
    if isinstance(df, str):
        df = pd.read_csv(df)
    rows = [extract_window_features(df.iloc[i:i + 128]) for i in range(0, len(df), 20)]
    return np.vstack(rows) if rows else np.zeros((1, 30), dtype=float)


def extract_window_labels(window_df):
    labels = []
    for pmu in range(1, 4):
        if bool(window_df.get(f"PMU{pmu} Sync Fault Active", pd.Series(False)).fillna(False).astype(bool).any()):
            labels.append("SYNC")
        elif bool(window_df.get(f"PMU{pmu} Clock Drift Fault", pd.Series(False)).fillna(False).astype(bool).any()):
            labels.append("CLOCK_DRIFT")
        elif bool(window_df.get(f"PMU{pmu} Bad Data", pd.Series(False)).fillna(False).astype(bool).any()):
            labels.append("BAD_DATA")
    return labels[0] if labels else "NORMAL"
