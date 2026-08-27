"""
===========================================================
network_model.py

Generic Distribution Network Model

This module automatically constructs the Bus Admittance
Matrix (Ybus) from line data.

Only LINE_DATA needs to be modified to scale the network.

Examples:

3 Bus:
    (1,2,Z12)
    (2,3,Z23)

6 Bus:
    (1,2,Z12)
    (2,3,Z23)
    (3,4,Z34)
    (4,5,Z45)
    (5,6,Z56)

IEEE 14:
    simply add all branch data.

===========================================================
"""

import numpy as np

###########################################################################
# NETWORK DEFINITION
###########################################################################

# (From Bus, To Bus, Impedance)

LINE_DATA = [

    (1, 2, 0.02 + 0.04j),

    (2, 3, 0.015 + 0.03j)

]

###########################################################################
# NUMBER OF BUSES
###########################################################################

NUM_BUSES = max(

    max(frm, to)

    for frm, to, _ in LINE_DATA

)

###########################################################################
# BUILD YBUS
###########################################################################

YBUS = np.zeros(

    (NUM_BUSES, NUM_BUSES),

    dtype=complex

)

for frm, to, impedance in LINE_DATA:

    y = 1 / impedance

    i = frm - 1
    j = to - 1

    YBUS[i, i] += y
    YBUS[j, j] += y

    YBUS[i, j] -= y
    YBUS[j, i] -= y

###########################################################################
# DISPLAY
###########################################################################

def print_network():

    print("\n====================================")
    print(" Distribution Network")
    print("====================================")

    print(f"\nNumber of Buses : {NUM_BUSES}")

    print("\nLine Data\n")

    for frm, to, z in LINE_DATA:

        print(

            f"Bus {frm}"

            f"  <---->  "

            f"Bus {to}"

            f"     Z = {z:.4f}"

        )

    print("\nYbus Matrix\n")

    print(YBUS)

    print("====================================")