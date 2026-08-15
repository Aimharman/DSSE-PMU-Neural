"""
===========================================================
Distribution System State Estimator
===========================================================

Reads one PMU snapshot from PMU_Output.csv and constructs:

1. Measurement Vector z
2. Initial State Vector x0
3. Predicted Measurement Vector h(x)
4. Residual Vector r = z - h(x)

The estimator can now select a CSV row by sample index.  This is
used by main.py for automatic PDC-style scanning of the complete
simulation; no laptop timestamp or manually entered simulation time
is required.
===========================================================
"""

import numpy as np
import pandas as pd

ADC_RATE = 1000

from network_model import NUM_BUSES
from measurement_model import measurement_model


class StateEstimator:

    def __init__(self, csv_file, sample_index=None):
        self.csv_file = csv_file
        self.sample_index = sample_index

        self.df = pd.read_csv(csv_file)

        self.selected_row = None
        self.selected_index = None
        self.selected_timestamp = None

    def select_snapshot(self, sample_index=None):
        """Select a PMU snapshot by CSV sample index."""
        if sample_index is not None:
            self.sample_index = int(sample_index)

        if self.sample_index is None:
            self.selected_index = len(self.df) - 1
        else:
            self.selected_index = max(
                0,
                min(int(self.sample_index), len(self.df) - 1),
            )

        self.selected_row = self.df.iloc[self.selected_index]

        # Prefer the simulator's explicit time column when present.
        if "Time (s)" in self.df.columns:
            value = self.selected_row["Time (s)"]
            self.selected_timestamp = (
                float(value) if pd.notna(value)
                else self.selected_index / ADC_RATE
            )
        else:
            self.selected_timestamp = self.selected_index / ADC_RATE

        return self.selected_row

    def build_measurement_vector(self, apply_sync_correction=False):
        latest = self.select_snapshot()

        measurements = []
        self.measurement_names = []
        self.sync_offsets_used = {}

        for bus in range(1, NUM_BUSES + 1):

            voltage_magnitude = latest[f"PMU{bus} Voltage Magnitude"]
            voltage_phase = latest[f"PMU{bus} Voltage Phase"]
            current_magnitude = latest[f"PMU{bus} Current Magnitude"]
            current_phase = latest[f"PMU{bus} Current Phase"]

            offset = 0.0

            if apply_sync_correction:
                offset = latest.get(f"PMU{bus} Sync Offset", 0.0)
                if pd.isna(offset):
                    offset = 0.0

                voltage_phase -= offset
                current_phase -= offset

            self.sync_offsets_used[f"PMU{bus}"] = float(offset)

            measurements.extend([
                voltage_magnitude,
                voltage_phase,
                current_magnitude,
                current_phase,
            ])

            self.measurement_names.extend([
                f"PMU{bus} Voltage Magnitude",
                f"PMU{bus} Voltage Phase",
                f"PMU{bus} Current Magnitude",
                f"PMU{bus} Current Phase",
            ])

        self.z = np.asarray(measurements, dtype=float)

        self.z[1::4] = np.deg2rad(self.z[1::4])
        self.z[3::4] = np.deg2rad(self.z[3::4])

        return self.z

    def initialize_state(self, apply_sync_correction=False):
        latest = self.select_snapshot()

        state = []

        for bus in range(1, NUM_BUSES + 1):

            voltage_phase = latest[f"PMU{bus} Voltage Phase"]

            if apply_sync_correction:
                offset = latest.get(f"PMU{bus} Sync Offset", 0.0)
                if pd.isna(offset):
                    offset = 0.0
                voltage_phase -= offset

            state.extend([
                latest[f"PMU{bus} Voltage Magnitude"],
                np.deg2rad(voltage_phase),
            ])

        self.x = np.asarray(state, dtype=float)

        return self.x

    def predict_measurements(self):
        self.h = measurement_model(self.x)
        return self.h

    def compute_residual(self):
        self.residual = self.z - self.h
        return self.residual

    def summary(self):
        z_print = self.z.copy()
        h_print = self.h.copy()
        r_print = self.residual.copy()
        x_print = self.x.copy()

        z_print[1::4] = np.rad2deg(z_print[1::4])
        z_print[3::4] = np.rad2deg(z_print[3::4])

        h_print[1::4] = np.rad2deg(h_print[1::4])
        h_print[3::4] = np.rad2deg(h_print[3::4])

        r_print[1::4] = np.rad2deg(r_print[1::4])
        r_print[3::4] = np.rad2deg(r_print[3::4])

        x_print[1::2] = np.rad2deg(x_print[1::2])

        print("\n================================================")
        print(" Distribution System State Estimator")
        print("================================================")
        print(f"CSV Sample Index : {self.selected_index}")
        print(f"Simulation Time  : {self.selected_timestamp:.6f} s")

        print("\nMeasurement Vector (z)\n")
        print(z_print)

        print("\nInitial State Vector (x0)\n")
        print(x_print)

        print("\nPredicted Measurement h(x)\n")
        print(h_print)

        print("\nResidual Vector r = z - h(x)\n")
        print(r_print)

        print("\nMeasurement Dimension :", len(self.z))
        print("State Dimension       :", len(self.x))
        print("================================================")

    def run(self, apply_sync_correction=False, verbose=True):
        self.build_measurement_vector(
            apply_sync_correction=apply_sync_correction
        )
        self.initialize_state(
            apply_sync_correction=apply_sync_correction
        )
        self.predict_measurements()
        self.compute_residual()

        if verbose:
            self.summary()

        return self