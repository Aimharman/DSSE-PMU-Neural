"""Feature extraction v4.1.

V4.1 is a targeted timing refinement over v4.  It keeps the v4 measurement
features and adds explicit, measurement-derived fixed-offset versus
persistent-slope descriptors for the SYNC/CLOCK_DRIFT decision.

No simulator fault flags are used as neural inputs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from feature_extractor_v4 import (
    PDC_RATE_HZ, ADC_RATE_HZ, WINDOW, TIMING_RECENT, TIMING_LONG,
    PMUS, PMU_PAIRS, _phase_difference, _mean, _std, _safe_slope,
    _baseline_from_history, BASELINE_SAMPLES, PRE_FAULT_SAMPLES,
    extract_window_features_v4, window_label,
)

EPS = 1e-6


def extract_window_features_v41(window, history=None, baseline=None):
    """Return v4 features plus robust physics-inspired timing features."""
    out = dict(extract_window_features_v4(window, history=history, baseline=baseline))
    if history is None:
        history = window
    if baseline is None:
        baseline = _baseline_from_history(history)

    dt = 1.0 / ADC_RATE_HZ

    for a, b in PMU_PAIRS:
        pair = f"{a}{b}"
        d = _phase_difference(
            history[f"PMU{a} Voltage Phase"].to_numpy(float),
            history[f"PMU{b} Voltage Phase"].to_numpy(float),
        )
        base = float(baseline.get((a, b), 0.0))

        # Work entirely relative to the normal measured PMU-to-PMU phase.
        rel = d - base
        recent = rel[-TIMING_RECENT:]
        long = rel[-TIMING_LONG:]
        pre = (rel[-(PRE_FAULT_SAMPLES + TIMING_RECENT):-TIMING_RECENT]
               if len(rel) >= PRE_FAULT_SAMPLES + TIMING_RECENT else np.array([]))

        recent_mean = _mean(recent)
        long_mean = _mean(long)
        pre_mean = _mean(pre)

        recent_slope = _safe_slope(recent, dt)
        long_slope = _safe_slope(long, dt)
        recent_std = _std(recent)

        # A fixed synchronization error produces a non-zero offset with a
        # small continuing slope.  A clock drift produces a persistent slope.
        offset = abs(recent_mean)
        slope = abs(recent_slope)
        long_s = abs(long_slope)
        step = abs(recent_mean - pre_mean)

        # Dimensionless separation features.
        stable_offset = offset / (1.0 + slope + recent_std)
        slope_dominance = (slope + 0.5 * long_s) / (1.0 + offset)
        offset_to_slope = offset / (EPS + slope + 0.5 * long_s)
        step_persistence = step / (1.0 + abs(recent_mean) + recent_std)

        # Explicit evidence scores. These are features, not hard labels.
        # Thresholds are deliberately mild because the final controller also
        # requires temporal persistence before acting.
        fixed_offset_evidence = (
            offset_to_slope *
            (1.0 / (1.0 + recent_std)) *
            (1.0 + min(step, 20.0) / 20.0)
        )
        drift_evidence = slope + 0.5 * long_s

        out[f"v41_{pair}_offset"] = offset
        out[f"v41_{pair}_slope"] = slope
        out[f"v41_{pair}_long_slope"] = long_s
        out[f"v41_{pair}_step"] = step
        out[f"v41_{pair}_stable_offset"] = stable_offset
        out[f"v41_{pair}_slope_dominance"] = slope_dominance
        out[f"v41_{pair}_offset_to_slope"] = offset_to_slope
        out[f"v41_{pair}_step_persistence"] = step_persistence
        out[f"v41_{pair}_fixed_offset_evidence"] = fixed_offset_evidence
        out[f"v41_{pair}_drift_evidence"] = drift_evidence

    return {k: float(v) for k, v in out.items()}


def build_dataset_v41(csv_paths, include_mixed=False):
    rows, labels, sources, times = [], [], [], []
    required = []
    for p in PMUS:
        required += [
            f"PMU{p} Voltage Magnitude", f"PMU{p} Voltage Phase",
            f"PMU{p} Current Magnitude", f"PMU{p} Current Phase",
        ]

    for path in csv_paths:
        df = pd.read_csv(path).reset_index(drop=True)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing PMU measurement columns in {path}: {missing}")

        first_end = max(WINDOW - 1, TIMING_LONG - 1)
        baseline_history = df.iloc[:min(BASELINE_SAMPLES, len(df))]
        baseline = _baseline_from_history(baseline_history)

        for end in range(first_end, len(df), WINDOW):
            window = df.iloc[end-WINDOW+1:end+1]
            if len(window) != WINDOW or window[required].isna().any().any():
                continue
            label = window_label(window)
            if label == "MIXED" and not include_mixed:
                continue
            hs = max(0, end - TIMING_LONG + 1)
            history = df.iloc[hs:end+1]
            feat = extract_window_features_v41(window, history, baseline)
            if not all(np.isfinite(v) for v in feat.values()):
                continue
            rows.append(feat)
            labels.append(label)
            sources.append(path)
            times.append(float(window["Time (s)"].iloc[-1])
                         if "Time (s)" in window.columns else end/ADC_RATE_HZ)

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="label")
    meta = pd.DataFrame({"source": sources, "time_s": times, "label": labels})
    return X, y, meta
