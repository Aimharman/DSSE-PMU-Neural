"""Feature extraction for the Neural Active Fault Management Controller.

The extractor uses PMU measurements/phasors only. Simulator fault flags are
used only to create supervised labels and are never used as neural inputs.

Timing-fault separation is strengthened with:
  * relative PMU phase error features
  * fixed-offset / trend features
  * linear-fit residuals
  * short-context timing trends across consecutive PDC windows

The basic decision window remains 20 raw samples (50-Hz PDC at 1-kHz ADC).
For training, a configurable number of preceding PDC windows is used only to
measure temporal evolution; the label belongs to the final PDC window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PDC_RATE_HZ = 50.0
ADC_RATE_HZ = 1000.0
WINDOW = int(round(ADC_RATE_HZ / PDC_RATE_HZ))  # 20 raw samples

# Number of PDC frames used for the timing-trend context.
# 10 frames = 200 ms.  The actual neural decision window is still 20 samples.
TIMING_CONTEXT_WINDOWS = 10
TIMING_CONTEXT_SAMPLES = WINDOW * TIMING_CONTEXT_WINDOWS

PMUS = (1, 2, 3)
PAIRS = ((1, 2), (1, 3), (2, 3))


def _unwrap(values: np.ndarray) -> np.ndarray:
    return np.unwrap(np.deg2rad(values)) * 180.0 / np.pi


def _slope(values: np.ndarray, dt: float) -> float:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if good.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype=float) * dt
    return float(np.polyfit(x[good], values[good], 1)[0])


def _linear_fit(values: np.ndarray, dt: float) -> tuple[float, float, float]:
    """Return slope, intercept and RMS residual of a linear trend."""
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if good.sum() < 2:
        return np.nan, np.nan, np.nan

    x = np.arange(len(values), dtype=float)[good] * dt
    y = values[good]
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return float(slope), float(intercept), rms


def _phase_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Difference of already-unwrapped phase traces."""
    return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)


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

    unique = sorted(set(active))
    if not unique:
        return "NORMAL"
    if len(unique) == 1:
        return unique[0]
    return "MIXED"


def _timing_features_from_context(
    context: pd.DataFrame,
    out: dict[str, float],
) -> None:
    """Extract timing-discriminating features from a multi-frame context.

    A fixed synchronization offset tends to have a large level but little
    temporal slope. Clock drift tends to create a sustained change in relative
    phase. These features therefore expose both the level and its evolution.
    """
    dt = 1.0 / ADC_RATE_HZ

    phases = {
        pmu: _unwrap(
            context[f"PMU{pmu} Voltage Phase"].to_numpy(float)
        )
        for pmu in PMUS
    }

    # A phase difference is evaluated directly from unwrapped PMU phases.
    # This avoids independently wrapping the difference at +/-180 degrees.
    for a, b in PAIRS:
        d = _phase_difference(phases[a], phases[b])

        slope, intercept, rms = _linear_fit(d, dt)
        half = max(1, len(d) // 2)

        first = float(np.nanmean(d[:half]))
        second = float(np.nanmean(d[-half:]))
        delta = second - first

        out[f"timing_diff_{a}{b}_mean"] = float(np.nanmean(d))
        out[f"timing_diff_{a}{b}_std"] = float(np.nanstd(d))
        out[f"timing_diff_{a}{b}_min"] = float(np.nanmin(d))
        out[f"timing_diff_{a}{b}_max"] = float(np.nanmax(d))
        out[f"timing_diff_{a}{b}_range"] = float(np.nanmax(d) - np.nanmin(d))
        out[f"timing_diff_{a}{b}_start"] = float(d[0])
        out[f"timing_diff_{a}{b}_end"] = float(d[-1])
        out[f"timing_diff_{a}{b}_change"] = delta
        out[f"timing_diff_{a}{b}_slope"] = float(slope)
        out[f"timing_diff_{a}{b}_slope_abs"] = float(abs(slope))
        out[f"timing_diff_{a}{b}_fit_rms"] = float(rms)

        # Compare the two halves. This is particularly useful for a drifting
        # timing error while remaining insensitive to absolute phase offset.
        out[f"timing_diff_{a}{b}_half_change_rate"] = float(
            delta / max((len(d) * dt) / 2.0, dt)
        )

    # Aggregate temporal behaviour across the three PMU pairs.
    slopes = [
        out[f"timing_diff_{a}{b}_slope"]
        for a, b in PAIRS
    ]
    changes = [
        out[f"timing_diff_{a}{b}_change"]
        for a, b in PAIRS
    ]

    out["timing_pair_slope_mean"] = float(np.nanmean(slopes))
    out["timing_pair_slope_std"] = float(np.nanstd(slopes))
    out["timing_pair_slope_abs_mean"] = float(np.nanmean(np.abs(slopes)))
    out["timing_pair_change_mean"] = float(np.nanmean(changes))
    out["timing_pair_change_abs_mean"] = float(np.nanmean(np.abs(changes)))


def extract_window_features(
    window: pd.DataFrame,
    context: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Extract measurement-derived features from one 20-sample PDC window.

    Parameters
    ----------
    window:
        The final 20 raw samples used for the current PDC decision.
    context:
        Optional longer measurement history ending at ``window``. If omitted,
        the current 20-sample window is used for timing features. Training and
        the stateful inference controller should supply the longer context.
    """
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

    # Relative phase relationships in the immediate 20-sample decision window.
    phases = {
        pmu: _unwrap(
            window[f"PMU{pmu} Voltage Phase"].to_numpy(float)
        )
        for pmu in PMUS
    }

    for a, b in PAIRS:
        d = _phase_difference(phases[a], phases[b])
        slope, _, rms = _linear_fit(d, dt)

        out[f"phase_diff_{a}{b}_mean"] = float(np.nanmean(d))
        out[f"phase_diff_{a}{b}_std"] = float(np.nanstd(d))
        out[f"phase_diff_{a}{b}_slope"] = float(slope)
        out[f"phase_diff_{a}{b}_end"] = float(d[-1])
        out[f"phase_diff_{a}{b}_start"] = float(d[0])
        out[f"phase_diff_{a}{b}_change"] = float(d[-1] - d[0])
        out[f"phase_diff_{a}{b}_range"] = float(np.nanmax(d) - np.nanmin(d))
        out[f"phase_diff_{a}{b}_fit_rms"] = float(rms)

    # Longer temporal context is the main addition for clock-drift separation.
    if context is None:
        context = window

    _timing_features_from_context(context, out)

    # End-of-window values preserve the instantaneous operating point.
    for pmu in PMUS:
        prefix = f"PMU{pmu}"
        out[f"{prefix}_v_mag_end"] = float(
            window[f"{prefix} Voltage Magnitude"].iloc[-1]
        )
        out[f"{prefix}_v_phase_end"] = float(
            window[f"{prefix} Voltage Phase"].iloc[-1]
        )
        out[f"{prefix}_i_mag_end"] = float(
            window[f"{prefix} Current Magnitude"].iloc[-1]
        )
        out[f"{prefix}_i_phase_end"] = float(
            window[f"{prefix} Current Phase"].iloc[-1]
        )

    return out


def build_dataset(csv_paths: list[str], include_mixed: bool = False):
    """Build X/y from simulator CSV files.

    The label is taken from the final 20-sample PDC window.  The feature vector
    includes up to TIMING_CONTEXT_WINDOWS consecutive PDC frames ending at that
    window so that a clock drift can be distinguished from a fixed offset.
    """
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
        df = pd.read_csv(path).reset_index(drop=True)

        if any(c not in df.columns for c in required):
            raise ValueError(f"Missing PMU measurement columns in {path}")

        # Preserve the existing 20-sample PDC cadence.
        for end in range(WINDOW - 1, len(df), WINDOW):
            window = df.iloc[end - WINDOW + 1 : end + 1]

            if len(window) != WINDOW or window[required].isna().any().any():
                continue

            label = window_label(window)
            if label == "MIXED" and not include_mixed:
                continue

            context_start = max(0, end - TIMING_CONTEXT_SAMPLES + 1)
            context = df.iloc[context_start : end + 1]

            # Do not create a special label for the context. It only supplies
            # measurement history to the final-window feature vector.
            feat = extract_window_features(window, context=context)

            if not all(np.isfinite(v) for v in feat.values()):
                continue

            rows.append(feat)
            labels.append(label)
            sources.append(path)
            times.append(float(window["Time (s)"].iloc[-1]))

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="label")
    meta = pd.DataFrame({
        "source": sources,
        "time_s": times,
        "label": labels,
    })
    return X, y, meta
