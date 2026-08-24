"""Minimal network model used by the PMU simulator and smoke tests."""

from __future__ import annotations

import numpy as np

NUM_BUSES = 3
YBUS = np.array([
    [3.0, -1.0, -2.0],
    [-1.0, 3.0, -2.0],
    [-2.0, -1.0, 3.0],
], dtype=float)

__all__ = ["NUM_BUSES", "YBUS"]
