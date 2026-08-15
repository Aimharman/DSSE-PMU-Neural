"""WLS extension for neural active measurement management.

This is an additive modification of the existing WeightedLeastSquares solver.
The existing solver's default behavior is preserved when measurement_weights
is None.

Measurement ordering is the project's existing 12-element order:
[Vmag1,Vang1,Imag1,Iang1, Vmag2,Vang2,Imag2,Iang2, Vmag3,Vang3,Imag3,Iang3].
"""
from __future__ import annotations
import numpy as np
from measurement_model import measurement_model, state_to_voltage, compute_currents
from jacobian import compute_jacobian
from network_model import NUM_BUSES

class NeuralWeightedLeastSquares:
    def __init__(self, tolerance=1e-6, max_iterations=250):
        self.tolerance=tolerance
        self.max_iterations=max_iterations

    @staticmethod
    def pmu_weights_to_measurement_weights(pmu_weights):
        """Expand [w1,w2,w3] to the 12 measurement weights."""
        if len(pmu_weights)!=3:
            raise ValueError('Expected three PMU weights.')
        return np.repeat(np.asarray(pmu_weights,dtype=float),4)

    def solve(self, z, x0, measurement_weights=None):
        x=x0.copy()
        R=np.diag([
            1e-4,1e-4,1e-2,1e-2,
            1e-4,1e-4,1e-2,1e-2,
            1e-4,1e-4,1e-2,1e-2])
        base_diag=np.diag(np.linalg.inv(R))
        if measurement_weights is None:
            mw=np.ones(len(z))
        else:
            mw=np.asarray(measurement_weights,dtype=float)
            if mw.size==3:
                mw=self.pmu_weights_to_measurement_weights(mw)
            if mw.size!=len(z):
                raise ValueError(f'Expected 3 PMU weights or {len(z)} measurement weights.')
            mw=np.clip(mw,0.0,1.0)

        for _ in range(self.max_iterations):
            h=measurement_model(x)
            r=z-h
            H=compute_jacobian(x)
            W=np.diag(base_diag*(mw**2))
            G=H.T@W@H
            g=H.T@W@r
            try:
                dx=np.linalg.solve(G,g)
            except np.linalg.LinAlgError:
                return x,r,W
            x=x+dx
            if np.linalg.norm(dx)<self.tolerance:
                break
        return x,r,W
