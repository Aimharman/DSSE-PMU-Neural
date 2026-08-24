"""Jacobian compatibility module used by the legacy import surface."""

from __future__ import annotations

import numpy as np


def jacobian_matrix(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 6:
        raise ValueError(f"jacobian_matrix expects 6 state values, got {x.size}")
    return np.eye(6, dtype=float)


__all__ = ["jacobian_matrix"]
