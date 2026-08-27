"""WLS extension for neural active measurement management.

Default behavior is unchanged when measurement_weights is None.

Supported weighting:
    - 3 PMU weights: each PMU's four measurements receive the same weight.
    - 12 measurement weights:
      [Vmag1,Vang1,Imag1,Iang1, Vmag2,Vang2,Imag2,Iang2,
       Vmag3,Vang3,Imag3,Iang3]

The 12-weight form allows synchronization errors to down-weight only
phase measurements while retaining magnitude measurements.
"""

from __future__ import annotations

import numpy as np

from measurement_model import measurement_model
from jacobian import compute_jacobian


class NeuralWeightedLeastSquares:
    def __init__(self, tolerance=1e-6, max_iterations=250):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    @staticmethod
    def pmu_weights_to_measurement_weights(pmu_weights):
        if len(pmu_weights) != 3:
            raise ValueError("Expected three PMU weights.")
        return np.repeat(np.asarray(pmu_weights, dtype=float), 4)

    @staticmethod
    def validate_measurement_weights(measurement_weights, measurement_count):
        mw = np.asarray(measurement_weights, dtype=float).reshape(-1)

        if mw.size == 3:
            mw = NeuralWeightedLeastSquares.pmu_weights_to_measurement_weights(mw)

        if mw.size != measurement_count:
            raise ValueError(
                f"Expected 3 PMU weights or {measurement_count} "
                f"measurement weights; got {mw.size}."
            )

        if not np.all(np.isfinite(mw)):
            raise ValueError("Measurement weights must be finite.")

        return np.clip(mw, 0.0, 1.0)

    def solve(self, z, x0, measurement_weights=None):
        x = np.asarray(x0, dtype=float).copy()
        z = np.asarray(z, dtype=float)

        # Existing project measurement-noise model.
        R = np.diag([
            1e-4, 1e-4, 1e-2, 1e-2,
            1e-4, 1e-4, 1e-2, 1e-2,
            1e-4, 1e-4, 1e-2, 1e-2,
        ])

        if len(z) != len(R):
            raise ValueError(
                f"Expected {len(R)} measurements, got {len(z)}."
            )

        base_diag = np.diag(np.linalg.inv(R))

        if measurement_weights is None:
            mw = np.ones(len(z), dtype=float)
        else:
            mw = self.validate_measurement_weights(
                measurement_weights, len(z)
            )

        for _ in range(self.max_iterations):
            h = measurement_model(x)
            r = z - h
            H = compute_jacobian(x)

            # Squared action weights preserve the existing implementation's
            # interpretation of a management weight as a confidence factor.
            W = np.diag(base_diag * (mw ** 2))

            G = H.T @ W @ H
            g = H.T @ W @ r

            try:
                dx = np.linalg.solve(G, g)
            except np.linalg.LinAlgError:
                return x, r, W

            x = x + dx

            if np.linalg.norm(dx) < self.tolerance:
                break

        return x, r, W
