"""
===========================================================
wls.py

Weighted Least Squares Solver

Implements iterative Gauss-Newton State Estimation.

Faulty measurements can be isolated by assigning them
zero weight. The returned active_indices identify the
measurements that actually contribute to the WLS solution.
===========================================================
"""

import numpy as np

from measurement_model import (
    measurement_model,
    state_to_voltage,
    compute_currents,
)

from jacobian import compute_jacobian
from network_model import NUM_BUSES


###########################################################################
# CONFIGURATION
###########################################################################

MAX_ITERATIONS = 250
TOLERANCE = 1e-6


class WeightedLeastSquares:

    def __init__(
        self,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    ):

        self.tolerance = tolerance
        self.max_iterations = max_iterations

    #######################################################################
    # Build covariance matrix
    #######################################################################

    @staticmethod
    def _build_covariance_matrix():

        return np.diag([
            1e-4, 1e-4, 1e-2, 1e-2,
            1e-4, 1e-4, 1e-2, 1e-2,
            1e-4, 1e-4, 1e-2, 1e-2,
        ])

    #######################################################################
    # Normalize bad-data indices
    #######################################################################

    @staticmethod
    def _normalize_bad_data_indices(
        bad_data_index,
        bad_data_indices,
    ):

        indices = []

        if bad_data_indices is not None:
            indices.extend(
                int(idx)
                for idx in bad_data_indices
            )

        if bad_data_index is not None:
            indices.append(
                int(bad_data_index)
            )

        # Remove duplicates while preserving order.
        return list(dict.fromkeys(indices))

    #######################################################################
    # Build weight matrix
    #######################################################################

    @staticmethod
    def _build_weight_matrix(
        R,
        bad_data_indices,
        bad_data_weight,
    ):

        base_diag = np.diag(
            np.linalg.inv(R)
        )

        weights = np.ones(
            len(base_diag)
        )

        for idx in bad_data_indices:
            if 0 <= idx < len(weights):
                weights[idx] = (
                    bad_data_weight ** 2
                )

        W = np.diag(
            base_diag * weights
        )

        return W

    #######################################################################
    # Solve WLS
    #######################################################################

    def solve(
        self,
        z,
        x0,
        bad_data_index=None,
        bad_data_indices=None,
        bad_data_weight=0.1,
    ):
        """
        Solve the nonlinear WLS state-estimation problem.

        Parameters
        ----------
        z : ndarray
            Measurement vector.

        x0 : ndarray
            Initial state vector.

        bad_data_index : int, optional
            Single measurement index to down-weight.

        bad_data_indices : iterable, optional
            Measurement indices to down-weight.

        bad_data_weight : float
            Relative weight factor. A value of 0.0 completely
            excludes the selected measurements from the WLS
            calculation.

        Returns
        -------
        x : ndarray
            Estimated state.

        r : ndarray
            Final residual vector.

        W : ndarray
            Final weight matrix.

        active_indices : ndarray
            Indices having non-zero WLS weight.
        """

        z = np.asarray(
            z,
            dtype=float,
        )

        x = np.asarray(
            x0,
            dtype=float,
        ).copy()

        bad_indices = self._normalize_bad_data_indices(
            bad_data_index,
            bad_data_indices,
        )

        ###################################################################
        # Covariance and weights
        ###################################################################

        R = self._build_covariance_matrix()

        W = self._build_weight_matrix(
            R,
            bad_indices,
            bad_data_weight,
        )

        active_indices = np.flatnonzero(
            np.diag(W) > 0.0
        )

        ###################################################################
        # Header
        ###################################################################

        print("\n==============================================")
        print(" Weighted Least Squares")
        print("==============================================")

        if bad_indices:

            print(
                "Isolating measurement indices "
                f"{bad_indices} with weight "
                f"{bad_data_weight:.3f}"
            )

            print(
                "Active measurement indices : "
                f"{active_indices}"
            )

        ###################################################################
        # Gauss-Newton iterations
        ###################################################################

        for iteration in range(
            self.max_iterations
        ):

            print(
                f"\nIteration {iteration + 1}"
            )

            ###############################################################
            # Measurement prediction
            ###############################################################

            h = measurement_model(x)

            ###############################################################
            # Residual
            ###############################################################

            r = z - h

            ###############################################################
            # Analytical Jacobian
            ###############################################################

            H = compute_jacobian(x)

            ###############################################################
            # Gain matrix
            ###############################################################

            G = H.T @ W @ H

            ###############################################################
            # Gradient
            ###############################################################

            g = H.T @ W @ r

            ###############################################################
            # State correction
            ###############################################################

            try:

                dx = np.linalg.solve(
                    G,
                    g,
                )

            except np.linalg.LinAlgError:

                print(
                    "Gain matrix is singular."
                )

                return (
                    x,
                    r,
                    W,
                    active_indices,
                )

            ###############################################################
            # Update state
            ###############################################################

            x = x + dx

            ###############################################################
            # Display convergence information
            ###############################################################

            print("\nCurrent Magnitudes")

            V = state_to_voltage(x)
            I = compute_currents(V)

            print(
                np.abs(I)
            )

            print("\nResidual Norm")
            print(
                np.linalg.norm(r)
            )

            print("\nCorrection Norm")
            print(
                np.linalg.norm(dx)
            )

            ###############################################################
            # Convergence
            ###############################################################

            if np.linalg.norm(dx) < self.tolerance:

                print("\nConverged.")
                break

        ###################################################################
        # Final consistency check
        ###################################################################

        V = state_to_voltage(x)
        I = compute_currents(V)

        print("\n==============================================")
        print(" Current Measurement Consistency")
        print("==============================================")

        for bus in range(NUM_BUSES):

            idx = 4 * bus

            measured_mag = z[
                idx + 2
            ]

            measured_ang = np.degrees(
                z[idx + 3]
            )

            predicted_mag = np.abs(
                I[bus]
            )

            predicted_ang = np.degrees(
                np.angle(I[bus])
            )

            if idx in bad_indices:

                status = " [ISOLATED]"

            else:

                status = ""

            print(
                f"\nBus {bus + 1}{status}"
            )

            print(
                "Measured Current Magnitude  : "
                f"{measured_mag:.6f}"
            )

            print(
                "Predicted Current Magnitude : "
                f"{predicted_mag:.6f}"
            )

            print(
                "Measured Current Angle      : "
                f"{measured_ang:.6f} deg"
            )

            print(
                "Predicted Current Angle     : "
                f"{predicted_ang:.6f} deg"
            )

        ###################################################################
        # Final state
        ###################################################################

        print("\n==============================================")
        print(" Final Estimated State")
        print("==============================================")

        print(x)

        return (
            x,
            r,
            W,
            active_indices,
        )