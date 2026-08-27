"""
===========================================================
jacobian.py

Analytical Jacobian Matrix

Stage 3.3

Computes

        H = ∂h/∂x

State Vector

x =

[V1 θ1
 V2 θ2
 V3 θ3]

Measurement Vector

h(x)

=

[V1 θ1 I1 φ1
 V2 θ2 I2 φ2
 V3 θ3 I3 φ3]

===========================================================
"""

import numpy as np

from network_model import NUM_BUSES

from measurement_model import (
    current_components,
    current_derivatives
)

def compute_jacobian(x):

    """
    Returns the analytical Jacobian matrix for the full
    measurement vector, including voltage magnitudes,
    voltage angles, current magnitudes, and current angles.
    """

    H = np.zeros((4 * NUM_BUSES, 2 * NUM_BUSES))

    ########################################################
    # Voltage Magnitude Measurements
    ########################################################

    for bus in range(NUM_BUSES):

        row = 4 * bus

        col = 2 * bus

        H[row, col] = 1.0

    ########################################################
    # Voltage Angle Measurements
    ########################################################

    for bus in range(NUM_BUSES):

        row = 4 * bus + 1

        col = 2 * bus + 1

        H[row, col] = 1.0

    ########################################################
    # Current Measurement Rows
    ########################################################

    I, IR, II = current_components(x)

    (
        dIR_dV,
        dIR_dTheta,
        dII_dV,
        dII_dTheta
    ) = current_derivatives(x)


    for bus in range(NUM_BUSES):

        row_mag = 4 * bus + 2
        row_ang = 4 * bus + 3

        I_mag = np.abs(I[bus])

        # Avoid division by zero
        if I_mag < 1e-12:
            continue

        for state_bus in range(NUM_BUSES):

            col_V = 2 * state_bus
            col_theta = 2 * state_bus + 1

            ####################################################
            # Current Magnitude Derivatives
            ####################################################

            H[row_mag, col_V] = (
                IR[bus] * dIR_dV[bus, state_bus]
                +
                II[bus] * dII_dV[bus, state_bus]
            ) / I_mag

            H[row_mag, col_theta] = (
                IR[bus] * dIR_dTheta[bus, state_bus]
                +
                II[bus] * dII_dTheta[bus, state_bus]
            ) / I_mag

            ####################################################
            # Current Angle Derivatives
            ####################################################

            denom = I_mag ** 2

            H[row_ang, col_V] = (
                IR[bus] * dII_dV[bus, state_bus]
                -
                II[bus] * dIR_dV[bus, state_bus]
            ) / denom

            H[row_ang, col_theta] = (
                IR[bus] * dII_dTheta[bus, state_bus]
                -
                II[bus] * dIR_dTheta[bus, state_bus]
            ) / denom

    return H