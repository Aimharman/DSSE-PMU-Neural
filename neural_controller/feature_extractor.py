"""Feature extraction for Neural Active Fault Management Controller.

The extractor uses only PMU measurements/phasors. Simulator fault flags are
used only to create training labels, never as neural-network inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PDC_RATE_HZ = 50.0
ADC_RATE_HZ = 1000.0
WINDOW = int(round(ADC_RATE_HZ / PDC_RATE_HZ))  # 20 raw samples

PMUS = (1, 2, 3)


def _unwrap(values: np.ndarray) -> np.ndarray:
    return np.unwrap(np.deg2rad(values)) * 180.0 / np.pi


def _slope(values: np.ndarray, dt: float) -> float:
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return np.nan
    x = np.arange(len(values), dtype=float) * dt
    return float(np.polyfit(x, values, 1)[0])


def _truth(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def window_label(window: pd.DataFrame) -> str:
    """Return a mutually-exclusive ground-truth class for one PDC window."""
    active = []
    for pmu in PMUS:
        bad = any(_truth(v) for v in window.get(f"PMU{pmu} Bad Data", []))
        sync = any(_truth(v) for v in window.get(f"PMU{pmu} Sync Fault Active", []))
        clock = any(_truth(v) for v in window.get(f"PMU{pmu} Clock Drift Fault", []))
        if bad:
            active.append(f"PMU{pmu}_BAD_DATA")
        if sync:
            active.append(f"PMU{pmu}_SYNC")
        if clock:
            active.append(f"PMU{pmu}_CLOCK_DRIFT")

    # Mixed simultaneous faults are not used in the first classifier because
    # they create an ambiguous single-label problem. They can be added later.
    unique = sorted(set(active))
    if not unique:
        return "NORMAL"
    if len(unique) == 1:
        return unique[0]
    return "MIXED"


def extract_window_features(window: pd.DataFrame) -> dict[str, float]:
    """Extract measurement-derived features from one 20-sample PDC window."""
    dt = 1.0 / ADC_RATE_HZ
    out: dict[str, float] = {}

    for pmu in PMUS:
        prefix = f"PMU{pmu}"
        vm = window[f"{prefix} Voltage Magnitude"].to_numpy(float)
        vp = window[f"{prefix} Voltage Phase"].to_numpy(float)
        im = window[f"{prefix} Current Magnitude"].to_numpy(float)
        ip = window[f"{prefix} Current Phase"].to_numpy(float)

        vp_u = _unwrap(vp)
        ip_u = _unwrap(ip)

        pairs = {
            "v_mag_mean": np.nanmean(vm),
            "v_mag_std": np.nanstd(vm),
            "v_mag_slope": _slope(vm, dt),
            "v_phase_mean": np.nanmean(vp_u),
            "v_phase_std": np.nanstd(vp_u),
            "v_phase_slope": _slope(vp_u, dt),
            "v_phase_delta": float(vp_u[-1] - vp_u[0]),
            "i_mag_mean": np.nanmean(im),
            "i_mag_std": np.nanstd(im),
            "i_mag_slope": _slope(im, dt),
            "i_phase_mean": np.nanmean(ip_u),
            "i_phase_std": np.nanstd(ip_u),
            "i_phase_slope": _slope(ip_u, dt),
            "i_phase_delta": float(ip_u[-1] - ip_u[0]),
        }
        for name, value in pairs.items():
            out[f"{prefix}_{name}"] = float(value)

    # Relative PMU phase relationships are particularly useful for detecting
    # fixed synchronization offsets.
    phases = {}
    for pmu in PMUS:
        phases[pmu] = _unwrap(window[f"PMU{pmu} Voltage Phase"].to_numpy(float))

    for a, b in ((1, 2), (1, 3), (2, 3)):
        d = phases[a] - phases[b]
        out[f"phase_diff_{a}{b}_mean"] = float(np.nanmean(d))
        out[f"phase_diff_{a}{b}_std"] = float(np.nanstd(d))
        out[f"phase_diff_{a}{b}_slope"] = _slope(d, dt)
        out[f"phase_diff_{a}{b}_end"] = float(d[-1])

    # End-of-window values preserve the instantaneous operating point.
    for pmu in PMUS:
        prefix = f"PMU{pmu}"
        out[f"{prefix}_v_mag_end"] = float(window[f"{prefix} Voltage Magnitude"].iloc[-1])
        out[f"{prefix}_v_phase_end"] = float(window[f"{prefix} Voltage Phase"].iloc[-1])
        out[f"{prefix}_i_mag_end"] = float(window[f"{prefix} Current Magnitude"].iloc[-1])
        out[f"{prefix}_i_phase_end"] = float(window[f"{prefix} Current Phase"].iloc[-1])

    return out


def build_dataset(csv_paths: list[str], include_mixed: bool = False):
    """Build X/y from one or more simulator CSV files."""
    rows = []
    labels = []
    sources = []
    times = []

    required = []
    for pmu in PMUS:
        required += [
            f"PMU{pmu} Voltage Magnitude",
            f"PMU{pmu} Voltage Phase",
            f"PMU{pmu} Current Magnitude",
            f"PMU{pmu} Current Phase",
        ]

    for path in csv_paths:
        df = pd.read_csv(path)
        df = df.reset_index(drop=True)
        if any(c not in df.columns for c in required):
            raise ValueError(f"Missing PMU measurement columns in {path}")

        for end in range(WINDOW - 1, len(df), WINDOW):
            window = df.iloc[end - WINDOW + 1 : end + 1]
            if len(window) != WINDOW or window[required].isna().any().any():
                continue

            label = window_label(window)
            if label == "MIXED" and not include_mixed:
                continue

            feat = extract_window_features(window)
            if not all(np.isfinite(v) for v in feat.values()):
                continue
            rows.append(feat)
            labels.append(label)
            sources.append(path)
            times.append(float(window["Time (s)"].iloc[-1]))

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="label")
    meta = pd.DataFrame({"source": sources, "time_s": times, "label": labels})
    return X, y, meta
