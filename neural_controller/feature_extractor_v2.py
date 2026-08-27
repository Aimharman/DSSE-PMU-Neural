"""Feature extraction v2 for the Neural Active Fault Management Controller.

Refactor v2 goals
-----------------
1. Preserve the original measurement-derived features.
2. Add multi-scale timing context:
      - 20 ms current PDC window
      - 100 ms recent history
      - 200 ms long history
3. Explicitly distinguish a fixed synchronization step from continuing clock drift
   using early/late slope and change features.
4. Never call nanmean/nanmin/nanmax on an empty/all-NaN array.
5. Simulator fault flags are used only to create labels, never as NN inputs.

The timing features are derived only from measured PMU voltage phase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PDC_RATE_HZ = 50.0
ADC_RATE_HZ = 1000.0
WINDOW = int(round(ADC_RATE_HZ / PDC_RATE_HZ))       # 20 raw samples = 20 ms

TIMING_RECENT = int(round(0.100 * ADC_RATE_HZ))      # 100 samples = 100 ms
TIMING_LONG = int(round(0.200 * ADC_RATE_HZ))        # 200 samples = 200 ms
TIMING_HALF_RECENT = TIMING_RECENT // 2
MIN_TIMING_VALID = 20

PMUS = (1, 2, 3)
PMU_PAIRS = ((1, 2), (1, 3), (2, 3))


def _unwrap(values: np.ndarray) -> np.ndarray:
    """Unwrap phase in degrees while preserving NaNs."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return out

    idx = np.flatnonzero(valid)
    # Unwrap contiguous valid runs separately so NaN gaps do not corrupt phase.
    split = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, split)
    for run in runs:
        out[run] = np.unwrap(np.deg2rad(values[run])) * 180.0 / np.pi
    return out


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    v = _finite(values)
    return float(np.mean(v)) if v.size else float(default)


def _safe_std(values: np.ndarray, default: float = 0.0) -> float:
    v = _finite(values)
    return float(np.std(v)) if v.size else float(default)


def _safe_min(values: np.ndarray, default: float = 0.0) -> float:
    v = _finite(values)
    return float(np.min(v)) if v.size else float(default)


def _safe_max(values: np.ndarray, default: float = 0.0) -> float:
    v = _finite(values)
    return float(np.max(v)) if v.size else float(default)


def _safe_delta(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    idx = np.flatnonzero(np.isfinite(values))
    if idx.size < 2:
        return float(default)
    return float(values[idx[-1]] - values[idx[0]])


def _slope(values: np.ndarray, dt: float) -> float:
    """Least-squares slope using only finite samples."""
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return 0.0

    x = np.flatnonzero(mask).astype(float) * dt
    y = values[mask]
    x = x - x.mean()

    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0

    return float(np.dot(x, y - y.mean()) / denom)


def _safe_ratio(num: float, den: float, eps: float = 1e-9) -> float:
    return float(num / (abs(den) + eps))


def _timing_stats(values: np.ndarray, dt: float) -> dict[str, float]:
    """Return robust multi-scale statistics for one timing-difference trace."""
    values = np.asarray(values, dtype=float)

    valid = np.flatnonzero(np.isfinite(values))
    if valid.size < MIN_TIMING_VALID:
        # Controlled neutral fallback.  build_dataset will still reject a
        # genuinely invalid measurement window; this only protects timing
        # history from early DFT NaNs / packet-loss gaps.
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
            "start": 0.0,
            "end": 0.0,
            "delta": 0.0,
            "slope": 0.0,
            "early_slope": 0.0,
            "late_slope": 0.0,
            "late_delta": 0.0,
            "slope_ratio": 0.0,
            "late_vs_long": 0.0,
        }

    # Work on the finite samples only for robust scalar statistics.
    v = values[valid]

    # Early/late sections retain their temporal ordering.
    half = len(values) // 2
    early = values[:half]
    late = values[-half:] if half else values

    slope = _slope(values, dt)
    early_slope = _slope(early, dt)
    late_slope = _slope(late, dt)

    late_idx = np.flatnonzero(np.isfinite(late))
    early_idx = np.flatnonzero(np.isfinite(early))

    if early_idx.size:
        early_end = float(early[early_idx[-1]])
    else:
        early_end = 0.0

    if late_idx.size:
        late_start = float(late[late_idx[0]])
        late_end = float(late[late_idx[-1]])
    else:
        late_start = late_end = 0.0

    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "range": float(np.max(v) - np.min(v)),
        "start": float(v[0]),
        "end": float(v[-1]),
        "delta": float(v[-1] - v[0]),
        "slope": slope,
        "early_slope": early_slope,
        "late_slope": late_slope,
        # This is intentionally a short-late-window change.  A fixed SYNC
        # offset should settle here; clock drift should continue moving.
        "late_delta": float(late_end - late_start),
        "slope_ratio": _safe_ratio(late_slope, slope),
        "late_vs_long": float(late_slope - slope),
    }


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


def _phase_difference(
    phase_a: np.ndarray,
    phase_b: np.ndarray,
) -> np.ndarray:
    """Create a stable relative phase trace in degrees."""
    a = _unwrap(phase_a)
    b = _unwrap(phase_b)
    d = a - b

    # Keep the actual relative phase offset.  Only remove whole-cycle
    # ambiguity; do NOT subtract the median because that would erase the
    # fixed synchronization offset we are trying to detect.
    d = (d + 180.0) % 360.0 - 180.0
    return d


def _add_timing_features(
    out: dict[str, float],
    phase_history: dict[int, np.ndarray],
    current_end: int,
    dt: float,
) -> None:
    """Add 20/100/200-ms timing features for all PMU phase pairs."""
    for a, b in PMU_PAIRS:
        d = _phase_difference(phase_history[a], phase_history[b])

        # Current 20 ms PDC window.
        current = d[-WINDOW:]
        current_stats = _timing_stats(current, dt)

        # Multi-scale history.
        recent = d[-TIMING_RECENT:]
        long = d[-TIMING_LONG:]

        recent_stats = _timing_stats(recent, dt)
        long_stats = _timing_stats(long, dt)

        pair = f"{a}{b}"

        # Preserve the original timing feature names from v1.
        out[f"timing_diff_{pair}_mean"] = recent_stats["mean"]
        out[f"timing_diff_{pair}_std"] = recent_stats["std"]
        out[f"timing_diff_{pair}_min"] = recent_stats["min"]
        out[f"timing_diff_{pair}_max"] = recent_stats["max"]
        out[f"timing_diff_{pair}_range"] = recent_stats["range"]

        # Current PDC-window timing behaviour.
        out[f"timing_current_{pair}_mean"] = current_stats["mean"]
        out[f"timing_current_{pair}_std"] = current_stats["std"]
        out[f"timing_current_{pair}_slope"] = current_stats["slope"]
        out[f"timing_current_{pair}_delta"] = current_stats["delta"]
        out[f"timing_current_{pair}_end"] = current_stats["end"]

        # 100-ms behaviour.
        out[f"timing_100ms_{pair}_mean"] = recent_stats["mean"]
        out[f"timing_100ms_{pair}_std"] = recent_stats["std"]
        out[f"timing_100ms_{pair}_slope"] = recent_stats["slope"]
        out[f"timing_100ms_{pair}_delta"] = recent_stats["delta"]
        out[f"timing_100ms_{pair}_early_slope"] = recent_stats["early_slope"]
        out[f"timing_100ms_{pair}_late_slope"] = recent_stats["late_slope"]
        out[f"timing_100ms_{pair}_late_delta"] = recent_stats["late_delta"]
        out[f"timing_100ms_{pair}_slope_ratio"] = recent_stats["slope_ratio"]
        out[f"timing_100ms_{pair}_late_vs_long"] = recent_stats["late_vs_long"]

        # 200-ms behaviour.
        out[f"timing_200ms_{pair}_mean"] = long_stats["mean"]
        out[f"timing_200ms_{pair}_std"] = long_stats["std"]
        out[f"timing_200ms_{pair}_slope"] = long_stats["slope"]
        out[f"timing_200ms_{pair}_delta"] = long_stats["delta"]
        out[f"timing_200ms_{pair}_early_slope"] = long_stats["early_slope"]
        out[f"timing_200ms_{pair}_late_slope"] = long_stats["late_slope"]
        out[f"timing_200ms_{pair}_late_delta"] = long_stats["late_delta"]
        out[f"timing_200ms_{pair}_slope_ratio"] = long_stats["slope_ratio"]
        out[f"timing_200ms_{pair}_late_vs_long"] = long_stats["late_vs_long"]

        # Explicit fixed-step-vs-drift discriminators.
        out[f"timing_{pair}_late_slope_abs"] = abs(recent_stats["late_slope"])
        out[f"timing_{pair}_long_slope_abs"] = abs(long_stats["slope"])
        out[f"timing_{pair}_late_slope_change"] = (
            recent_stats["late_slope"] - recent_stats["early_slope"]
        )
        out[f"timing_{pair}_late_to_long_ratio"] = _safe_ratio(
            recent_stats["late_slope"],
            long_stats["slope"],
        )
        out[f"timing_{pair}_late_change_abs"] = abs(recent_stats["late_delta"])

        # Direct fixed-offset indicator: the phase difference is large while
        # its late-window slope is small.
        out[f"timing_{pair}_offset_stability"] = _safe_ratio(
            abs(recent_stats["end"]),
            1.0 + abs(recent_stats["late_slope"]),
        )

        # Cross-scale consistency: clock drift should have a persistent
        # non-zero slope at both scales.
        out[f"timing_{pair}_persistent_slope"] = (
            0.5 * recent_stats["late_slope"] +
            0.5 * long_stats["late_slope"]
        )


def extract_window_features(
    window: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Extract original + v2 multi-scale timing features."""
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
            "v_mag_mean": _safe_mean(vm),
            "v_mag_std": _safe_std(vm),
            "v_mag_slope": _slope(vm, dt),
            "v_phase_mean": _safe_mean(vp_u),
            "v_phase_std": _safe_std(vp_u),
            "v_phase_slope": _slope(vp_u, dt),
            "v_phase_delta": _safe_delta(vp_u),
            "i_mag_mean": _safe_mean(im),
            "i_mag_std": _safe_std(im),
            "i_mag_slope": _slope(im, dt),
            "i_phase_mean": _safe_mean(ip_u),
            "i_phase_std": _safe_std(ip_u),
            "i_phase_slope": _slope(ip_u, dt),
            "i_phase_delta": _safe_delta(ip_u),
        }

        for name, value in pairs.items():
            out[f"{prefix}_{name}"] = float(value)

    # Relative PMU phase relationships for the current PDC window.
    phases = {
        pmu: _unwrap(
            window[f"PMU{pmu} Voltage Phase"].to_numpy(float)
        )
        for pmu in PMUS
    }

    for a, b in PMU_PAIRS:
        d = phases[a] - phases[b]
        out[f"phase_diff_{a}{b}_mean"] = _safe_mean(d)
        out[f"phase_diff_{a}{b}_std"] = _safe_std(d)
        out[f"phase_diff_{a}{b}_slope"] = _slope(d, dt)
        out[f"phase_diff_{a}{b}_end"] = (
            float(d[np.flatnonzero(np.isfinite(d))[-1]])
            if np.any(np.isfinite(d)) else 0.0
        )

    # Timing history is deliberately measured from the raw phase columns,
    # not from fault flags or simulator metadata.
    if history is None:
        history = window

    phase_history = {
        pmu: history[f"PMU{pmu} Voltage Phase"].to_numpy(float)
        for pmu in PMUS
    }

    _add_timing_features(
        out,
        phase_history=phase_history,
        current_end=len(history) - 1,
        dt=dt,
    )

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


def build_dataset(
    csv_paths: list[str],
    include_mixed: bool = False,
):
    """Build X/y using 20-ms PDC windows and up to 200-ms timing history."""
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

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing PMU measurement columns in {path}: {missing}"
            )

        # We need 200 ms of history so the v2 timing features have a real
        # long-term context.  This discards only the initial 180 ms or so.
        first_end = max(WINDOW - 1, TIMING_LONG - 1)

        for end in range(first_end, len(df), WINDOW):
            window = df.iloc[end - WINDOW + 1 : end + 1]

            if len(window) != WINDOW:
                continue

            if window[required].isna().any().any():
                continue

            label = window_label(window)
            if label == "MIXED" and not include_mixed:
                continue

            hist_start = max(0, end - TIMING_LONG + 1)
            history = df.iloc[hist_start : end + 1]

            feat = extract_window_features(window, history=history)

            # The extractor is required to produce a finite feature vector.
            if not all(np.isfinite(v) for v in feat.values()):
                continue

            rows.append(feat)
            labels.append(label)
            sources.append(path)

            if "Time (s)" in window.columns:
                times.append(float(window["Time (s)"].iloc[-1]))
            else:
                times.append(float(end) / ADC_RATE_HZ)

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="label")
    meta = pd.DataFrame(
        {"source": sources, "time_s": times, "label": labels}
    )

    return X, y, meta


if __name__ == "__main__":
    print("feature_extractor_v2.py")
    print(f"PDC window     : {WINDOW} samples ({WINDOW / ADC_RATE_HZ:.3f} s)")
    print(f"Recent timing  : {TIMING_RECENT} samples (0.100 s)")
    print(f"Long timing    : {TIMING_LONG} samples (0.200 s)")
    print(f"Minimum valid timing samples: {MIN_TIMING_VALID}")
