"""Load-profile utilities used by the PMU generator."""

from __future__ import annotations

import numpy as np


def get_loads(t):
    t = float(t)
    return np.array([1.0 + 0.1 * np.sin(t), 1.0 + 0.2 * np.cos(t), 1.0 + 0.15 * np.sin(2.0 * t)], dtype=float)


__all__ = ["get_loads"]
