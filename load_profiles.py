import numpy as np

def residential_load(t):
    return 0.70 + 0.20 * np.sin(2 * np.pi * 0.02 * t)

def commercial_load(t):
    return 0.90 + 0.10 * np.sin(2 * np.pi * 0.015 * t + np.pi / 4)

def industrial_load(t):
    return 1.10 + 0.05 * np.sin(2 * np.pi * 0.01 * t)

def apply_events(t, loads):
    L1, L2, L3 = loads
    if 2.0 <= t < 2.5:
        L3 *= 1.8
    if 4.0 <= t < 5.0:
        L2 *= 0.90
    if 6.0 <= t < 8.0:
        L1 *= 1.35
    if 8.0 <= t < 9.0:
        L3 *= 0.50
    return L1, L2, L3

def get_loads(t):
    return apply_events(t, (residential_load(t), commercial_load(t), industrial_load(t)))
