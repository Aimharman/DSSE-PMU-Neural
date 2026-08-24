"""Measurement model for the PMU state-estimation smoke tests."""

from __future__ import annotations

import numpy as np


def measurement_model(x):
    """Return a 12-measurement vector derived from a 6-state input."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 6:
        raise ValueError(f"measurement_model expects 6 state values, got {x.size}")

    return np.concatenate([
        x[:4],
        x[2:6],
        x[:4] + 0.1,
    ])


__all__ = ["measurement_model"]
