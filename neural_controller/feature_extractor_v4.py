"""Feature extraction v4 for the Neural Active Fault Management Controller.

v4 keeps the measurement-derived v2 feature set and adds baseline-relative
timing features.  The baseline is estimated from the beginning of each CSV
file, using only measured PMU phase relationships; simulator fault flags are
never inputs to the neural models.

The main v4 idea is:
    fixed SYNC fault -> phase-pair offset changes and then stabilizes
    CLOCK_DRIFT     -> phase-pair offset keeps changing with time

The baseline-relative features make that distinction explicit while retaining
the original v2 features for continuity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from feature_extractor_v2 import (
    PDC_RATE_HZ, ADC_RATE_HZ, WINDOW, TIMING_RECENT, TIMING_LONG,
    PMUS, PMU_PAIRS, _phase_difference, _safe_mean, _safe_std, _slope,
    extract_window_features as extract_v2,
)

BASELINE_SAMPLES = int(round(1.0 * ADC_RATE_HZ))
PRE_FAULT_SAMPLES = int(round(0.050 * ADC_RATE_HZ))
EPS = 1e-6


def _finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def _mean(a, default=0.0):
    x = _finite(a)
    return float(np.mean(x)) if x.size else default


def _std(a, default=0.0):
    x = _finite(a)
    return float(np.std(x)) if x.size else default


def _safe_slope(a, dt):
    return float(_slope(np.asarray(a, dtype=float), dt))


def _baseline_from_history(history: pd.DataFrame) -> dict[tuple[int,int], float]:
    """Estimate normal relative phase from the initial part of a file."""
    h = history.iloc[:min(BASELINE_SAMPLES, len(history))]
    result = {}
    for a, b in PMU_PAIRS:
        pa = h[f"PMU{a} Voltage Phase"].to_numpy(float)
        pb = h[f"PMU{b} Voltage Phase"].to_numpy(float)
        result[(a,b)] = _mean(_phase_difference(pa, pb))
    return result


def extract_window_features_v4(window: pd.DataFrame,
                               history: pd.DataFrame | None = None,
                               baseline: dict[tuple[int,int], float] | None = None):
    """Return v2 features plus baseline-relative timing features."""
    if history is None:
        history = window
    if baseline is None:
        baseline = _baseline_from_history(history)

    out = dict(extract_v2(window, history=history))
    dt = 1.0 / ADC_RATE_HZ

    for a, b in PMU_PAIRS:
        pair = f"{a}{b}"
        d = _phase_difference(
            history[f"PMU{a} Voltage Phase"].to_numpy(float),
            history[f"PMU{b} Voltage Phase"].to_numpy(float),
        )

        current = d[-WINDOW:]
        recent = d[-TIMING_RECENT:]
        long = d[-TIMING_LONG:]

        # Remove the normal file-specific PMU-to-PMU phase relationship.
        # This prevents a normal fixed PMU offset from looking like SYNC.
        base = float(baseline.get((a,b), 0.0))
        cur_rel = current - base
        recent_rel = recent - base
        long_rel = long - base

        # Previous 50 ms and current 50 ms are used to expose a fixed step.
        pre = d[-(PRE_FAULT_SAMPLES + TIMING_RECENT):-TIMING_RECENT] \
            if len(d) >= PRE_FAULT_SAMPLES + TIMING_RECENT else np.array([])
        pre_rel = pre - base

        cur_mean = _mean(cur_rel)
        recent_mean = _mean(recent_rel)
        long_mean = _mean(long_rel)
        pre_mean = _mean(pre_rel)

        cur_slope = _safe_slope(cur_rel, dt)
        recent_slope = _safe_slope(recent_rel, dt)
        long_slope = _safe_slope(long_rel, dt)

        recent_delta = (
            float(recent_rel[-1] - recent_rel[0])
            if len(_finite(recent_rel)) >= 2 else 0.0
        )

        # Baseline-relative descriptors.
        out[f"v4_{pair}_baseline"] = base
        out[f"v4_{pair}_current_offset"] = cur_mean
        out[f"v4_{pair}_recent_offset"] = recent_mean
        out[f"v4_{pair}_long_offset"] = long_mean
        out[f"v4_{pair}_offset_std"] = _std(recent_rel)
        out[f"v4_{pair}_current_slope"] = cur_slope
        out[f"v4_{pair}_recent_slope"] = recent_slope
        out[f"v4_{pair}_long_slope"] = long_slope
        out[f"v4_{pair}_slope_abs"] = abs(recent_slope)
        out[f"v4_{pair}_slope_persistence"] = 0.5 * (
            abs(recent_slope) + abs(long_slope)
        )
        out[f"v4_{pair}_recent_delta"] = recent_delta

        # A fixed synchronization step appears as a large change from the
        # pre-event reference followed by a small continuing slope.
        out[f"v4_{pair}_step_change"] = recent_mean - pre_mean
        out[f"v4_{pair}_step_abs"] = abs(recent_mean - pre_mean)

        # Large stable offset is a SYNC-like signature; persistent slope is a
        # DRIFT-like signature. These are still features, not hard decisions.
        out[f"v4_{pair}_sync_score"] = abs(recent_mean) / (
            1.0 + abs(recent_slope) + _std(recent_rel)
        )
        out[f"v4_{pair}_drift_score"] = (
            abs(recent_slope) + 0.5 * abs(long_slope)
        )
        out[f"v4_{pair}_stability_ratio"] = abs(recent_mean) / (
            EPS + abs(recent_slope) + _std(recent_rel)
        )
        out[f"v4_{pair}_slope_consistency"] = (
            1.0 - abs(recent_slope - long_slope) /
            (EPS + abs(recent_slope) + abs(long_slope))
        )

    return {k: float(v) for k, v in out.items()}


def window_label(window: pd.DataFrame) -> str:
    active = []
    for p in PMUS:
        def truth(v):
            if pd.isna(v):
                return False
            return str(v).strip().lower() in {"true","1","1.0","yes","y"}
        if any(truth(v) for v in window.get(f"PMU{p} Bad Data", [])):
            active.append(f"PMU{p}_BAD_DATA")
        if any(truth(v) for v in window.get(f"PMU{p} Sync Fault Active", [])):
            active.append(f"PMU{p}_SYNC")
        if any(truth(v) for v in window.get(f"PMU{p} Clock Drift Fault", [])):
            active.append(f"PMU{p}_CLOCK_DRIFT")
    unique = sorted(set(active))
    if not unique:
        return "NORMAL"
    if len(unique) == 1:
        return unique[0]
    return "MIXED"


def build_dataset_v4(csv_paths, include_mixed=False):
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
            feat = extract_window_features_v4(window, history, baseline)
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
