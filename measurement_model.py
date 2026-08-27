"""
===========================================================
measurement_model.py

Distribution System State Estimation
Stage 3.2

Measurement Model

Computes

        h(x)

from the state vector

        x = [V1 θ1 V2 θ2 ... VN θN]

using

        I = Ybus · V

NOTE
----
All internal angle calculations use RADIANS.

The predicted measurement vector is

h(x) =

[
Vmag1
Vphase1 (rad)
Imag1
Iphase1 (rad)

Vmag2
Vphase2 (rad)
Imag2
Iphase2 (rad)

...

VmagN
VphaseN
ImagN
IphaseN
]

===========================================================
"""

import numpy as np

from network_model import NUM_BUSES, YBUS


########################################################
# Convert State Vector to Complex Bus Voltages
########################################################

def state_to_voltage(x):
    """
    Convert

        x = [V1 θ1 V2 θ2 ...]

    into complex voltage vector.

    Angles are assumed to be in radians.
    """

    V = np.zeros(NUM_BUSES, dtype=complex)

    for bus in range(NUM_BUSES):

        magnitude = x[2 * bus]

        angle = x[2 * bus + 1]          # Already in radians

        V[bus] = magnitude * np.exp(1j * angle)

    return V


########################################################
# Compute Bus Currents
########################################################

def compute_currents(V):
    """
    Compute

        I = Ybus · V
    """

    return YBUS @ V


########################################################
# Current Components
########################################################

def current_components(x):
    """
    Returns

        I
        IR
        II
    """

    V = state_to_voltage(x)

    I = compute_currents(V)

    IR = np.real(I)

    II = np.imag(I)

    return I, IR, II


########################################################
# Analytical Current Derivatives
########################################################

def current_derivatives(x):
    """
    Computes

        ∂IR/∂V
        ∂IR/∂θ

        ∂II/∂V
        ∂II/∂θ

    Angles are assumed to be in radians.
    """

    dIR_dV = np.zeros((NUM_BUSES, NUM_BUSES))
    dIR_dTheta = np.zeros((NUM_BUSES, NUM_BUSES))

    dII_dV = np.zeros((NUM_BUSES, NUM_BUSES))
    dII_dTheta = np.zeros((NUM_BUSES, NUM_BUSES))

    G = np.real(YBUS)
    B = np.imag(YBUS)

    for i in range(NUM_BUSES):

        for j in range(NUM_BUSES):

            Vm = x[2 * j]

            theta = x[2 * j + 1]      # Already radians

            c = np.cos(theta)
            s = np.sin(theta)

            ################################################
            # Real Current
            ################################################

            dIR_dV[i, j] = (
                G[i, j] * c
                -
                B[i, j] * s
            )

            dIR_dTheta[i, j] = (
                Vm
                *
                (
                    -G[i, j] * s
                    -
                    B[i, j] * c
                )
            )

            ################################################
            # Imaginary Current
            ################################################

            dII_dV[i, j] = (
                G[i, j] * s
                +
                B[i, j] * c
            )

            dII_dTheta[i, j] = (
                Vm
                *
                (
                    G[i, j] * c
                    -
                    B[i, j] * s
                )
            )

    return (
        dIR_dV,
        dIR_dTheta,
        dII_dV,
        dII_dTheta
    )


########################################################
# Measurement Model
########################################################

def measurement_model(x):
    """
    Computes

        h(x)

    using

        I = Ybus · V

    All returned angles are in radians.
    """

    V = state_to_voltage(x)

    I = compute_currents(V)

    h = []

    for bus in range(NUM_BUSES):

        ####################################################
        # Voltage Measurements
        ####################################################

        h.append(np.abs(V[bus]))

        h.append(np.angle(V[bus]))      # radians

        ####################################################
        # Current Measurements
        ####################################################

        h.append(np.abs(I[bus]))

        h.append(np.angle(I[bus]))      # radians

    return np.array(h)