from __future__ import annotations

import numpy as np
import pandas as pd


PHASE_COLUMNS = ["PMU1 Voltage Phase", "PMU2 Voltage Phase", "PMU3 Voltage Phase"]


def _as_time_vector(window_df):
    if "Time (s)" in window_df.columns:
        t = pd.to_numeric(window_df["Time (s)"], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isnan(t)):
            t = np.arange(len(window_df), dtype=float)
        return np.asarray(t, dtype=float)
    return np.arange(len(window_df), dtype=float)


def compute_timing_features(window_df):
    """Compute real phase-difference timing features from delta_phi(t).

    Preserves pairwise PMU-to-PMU phase dynamics so that drift direction and PMU
    localization remain observable instead of being collapsed into a single mean
    magnitude value.
    """
    if window_df is None or window_df.empty or not all(col in window_df.columns for col in PHASE_COLUMNS):
        return {
            "offset": 0.0,
            "short_term_slope": 0.0,
            "long_term_slope": 0.0,
            "variance": 0.0,
            "step_change": 0.0,
            "persistence": 0.0,
            "delta_phi_mean": 0.0,
            "pair12_offset": 0.0,
            "pair12_slope": 0.0,
            "pair12_variance": 0.0,
            "pair12_step": 0.0,
            "pair12_persistence": 0.0,
            "pair13_offset": 0.0,
            "pair13_slope": 0.0,
            "pair13_variance": 0.0,
            "pair13_step": 0.0,
            "pair13_persistence": 0.0,
            "pair23_offset": 0.0,
            "pair23_slope": 0.0,
            "pair23_variance": 0.0,
            "pair23_step": 0.0,
            "pair23_persistence": 0.0,
        }

    time = _as_time_vector(window_df)
    phases = window_df[PHASE_COLUMNS].astype(float).to_numpy()
    valid_mask = np.all(np.isfinite(phases), axis=1)
    if not np.any(valid_mask):
        return {
            "offset": 0.0,
            "short_term_slope": 0.0,
            "long_term_slope": 0.0,
            "variance": 0.0,
            "step_change": 0.0,
            "persistence": 0.0,
            "delta_phi_mean": 0.0,
            "pair12_offset": 0.0,
            "pair12_slope": 0.0,
            "pair12_variance": 0.0,
            "pair12_step": 0.0,
            "pair12_persistence": 0.0,
            "pair13_offset": 0.0,
            "pair13_slope": 0.0,
            "pair13_variance": 0.0,
            "pair13_step": 0.0,
            "pair13_persistence": 0.0,
            "pair23_offset": 0.0,
            "pair23_slope": 0.0,
            "pair23_variance": 0.0,
            "pair23_step": 0.0,
            "pair23_persistence": 0.0,
        }

    phases = phases[valid_mask]
    time = time[valid_mask]
    deltas = np.column_stack([
        phases[:, 1] - phases[:, 0],
        phases[:, 2] - phases[:, 0],
        phases[:, 2] - phases[:, 1],
    ])
    all_deltas = deltas.ravel()
    abs_deltas = np.abs(all_deltas)
    offset = float(np.mean(abs_deltas)) if abs_deltas.size else 0.0
    variance = float(np.var(abs_deltas)) if abs_deltas.size else 0.0
    step_change = float(np.max(np.abs(np.diff(all_deltas)))) if len(all_deltas) > 1 else 0.0
    persistence = float(np.mean(np.abs(np.diff(all_deltas)))) if len(all_deltas) > 1 else 0.0
    delta_phi_mean = float(np.mean(abs_deltas)) if abs_deltas.size else 0.0

    pair_stats = {}
    for idx, label in enumerate(["pair12", "pair13", "pair23"]):
        pair = deltas[:, idx]
        pair_abs = np.abs(pair)
        slope = 0.0
        if len(pair) >= 2:
            try:
                slope, _ = np.polyfit(time, pair, 1)
                slope = float(np.abs(slope))
            except np.linalg.LinAlgError:
                slope = 0.0
        pair_stats[f"{label}_offset"] = float(np.mean(pair_abs)) if pair_abs.size else 0.0
        pair_stats[f"{label}_slope"] = slope
        pair_stats[f"{label}_variance"] = float(np.var(pair_abs)) if pair_abs.size else 0.0
        pair_stats[f"{label}_step"] = float(np.max(np.abs(np.diff(pair)))) if len(pair) > 1 else 0.0
        pair_stats[f"{label}_persistence"] = float(np.mean(np.abs(np.diff(pair)))) if len(pair) > 1 else 0.0

    short_terms = []
    long_terms = []
    for pair in deltas.T:
        if len(pair) <= 1 or np.allclose(pair, pair[0]):
            short_terms.append(0.0)
            long_terms.append(0.0)
            continue
        if len(time) >= 2:
            half = max(2, len(time) // 2)
            if len(pair) >= half:
                s_short, _ = np.polyfit(time[:half], pair[:half], 1)
                s_long, _ = np.polyfit(time[half:], pair[half:], 1)
                short_terms.append(float(abs(s_short)))
                long_terms.append(float(abs(s_long)))
            else:
                s_full, _ = np.polyfit(time, pair, 1)
                short_terms.append(float(abs(s_full)))
                long_terms.append(float(abs(s_full)))
        else:
            short_terms.append(0.0)
            long_terms.append(0.0)

    short_term_slope = float(np.mean(short_terms)) if short_terms else 0.0
    long_term_slope = float(np.mean(long_terms)) if long_terms else 0.0

    return {
        "offset": offset,
        "short_term_slope": short_term_slope,
        "long_term_slope": long_term_slope,
        "variance": variance,
        "step_change": step_change,
        "persistence": persistence,
        "delta_phi_mean": delta_phi_mean,
        **pair_stats,
    }
