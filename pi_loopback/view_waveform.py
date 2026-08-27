"""
view_waveform.py

Plot a captured loopback CSV (Time (s), Duty Cycle, Reconstructed Value)
with a horizontal scrollbar for time position and sliders for horizontal
(time-window) and vertical (amplitude) zoom.

Usage:
    python view_waveform.py pi_loopback/capture.csv
    python view_waveform.py pi_loopback/capture.csv --column "Reconstructed Value"
"""

import argparse
import csv
import math
from collections import deque
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


PMU_FIELDS = (
    "Voltage DFT Real", "Voltage DFT Imag", "Voltage Magnitude", "Voltage Phase",
    "Voltage DFT RMS", "Current DFT Real", "Current DFT Imag", "Current Magnitude",
    "Current Phase", "Current DFT RMS", "Sync Offset", "Sync Fault",
    "Sync Fault Active", "Mag Noise", "Phase Noise", "Clock Drift",
    "Clock Drift Fault", "Packet Loss", "Bad Data",
)


def canonical_columns():
    columns = [
        "Time (s)", "Voltage 1 (V)", "Current 1 (A)", "Voltage 2 (V)",
        "Current 2 (A)", "Voltage 3 (V)", "Current 3 (A)", "Peak (A)",
        "Signal Angle (deg)", "Delta t (s)",
    ]
    for pmu in range(1, 4):
        columns.extend(f"PMU{pmu} {field}" for field in PMU_FIELDS)
    return columns


def phasor(samples):
    count = len(samples)
    real = sum(value * math.cos(2.0 * math.pi * index / count)
               for index, value in enumerate(samples))
    imag = -sum(value * math.sin(2.0 * math.pi * index / count)
                for index, value in enumerate(samples))
    magnitude = 2.0 * math.hypot(real, imag) / count
    return real, imag, magnitude, math.degrees(math.atan2(imag, real)), magnitude / math.sqrt(2.0)


def export_pmu_csv(input_path, output_path, voltage_rms=220.0):
    """Export a 50 Hz/1 kHz loopback capture in the canonical scenario schema."""
    peak_voltage = voltage_rms * math.sqrt(2.0)
    voltage_samples = deque(maxlen=20)
    written = 0
    with open(input_path, newline="") as capture:
        reader = csv.DictReader(capture)
        required = {"Time (s)", "Reconstructed Value", "Poll Samples"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("input must be a loopback_combined capture CSV")
        capture_rows = list(reader)

    for index, raw_row in enumerate(capture_rows):
        if int(raw_row["Poll Samples"]) != 0:
            continue
        if index == 0 or index == len(capture_rows) - 1:
            raise ValueError("capture starts or ends with an empty sampling bin")
        previous_row = capture_rows[index - 1]
        next_row = capture_rows[index + 1]
        if int(previous_row["Poll Samples"]) == 0 or int(next_row["Poll Samples"]) == 0:
            raise ValueError("capture has consecutive empty sampling bins")
        raw_row["Reconstructed Value"] = str(
            (float(previous_row["Reconstructed Value"]) + float(next_row["Reconstructed Value"])) / 2.0
        )

    with open(output_path, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(canonical_columns())
        for index, raw_row in enumerate(capture_rows):
            time_s = float(raw_row["Time (s)"])
            if not math.isclose(time_s, index * 0.001, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("capture must be sampled at 1000 Hz")
            voltage = float(raw_row["Reconstructed Value"]) * peak_voltage
            voltage_samples.append(voltage)
            voltage_dft = phasor(voltage_samples) if len(voltage_samples) == 20 else (math.nan,) * 5
            current_dft = (0.0,) * 5 if len(voltage_samples) == 20 else (math.nan,) * 5
            row = [time_s, voltage, 0.0, voltage, 0.0, voltage, 0.0, 0.0,
                   (time_s * 50.0 * 360.0) % 360.0, 0.001]
            for _ in range(3):
                row.extend(voltage_dft)
                row.extend(current_dft)
                row.extend([0.0, math.nan, False, math.nan, math.nan, math.nan,
                            False, False, False])
            writer.writerow(row)
            written += 1
    if written < 20:
        raise ValueError("capture needs at least 20 valid samples")
    return written


def main():
    parser = argparse.ArgumentParser(description="Interactive viewer for loopback capture CSVs")
    parser.add_argument("csv_path", help="Path to a captured CSV (e.g. pi_loopback/capture.csv)")
    parser.add_argument("--time-column", default="Time (s)", help="Name of the time column")
    parser.add_argument("--column", default="Reconstructed Value", help="Name of the value column to plot")
    parser.add_argument("--export-pmu", type=Path, help="Write a 220 Vrms canonical PMU scenario CSV before plotting")
    parser.add_argument("--voltage-rms", type=float, default=220.0, help="Voltage calibration used by --export-pmu")
    args = parser.parse_args()

    if args.export_pmu:
        count = export_pmu_csv(args.csv_path, args.export_pmu, args.voltage_rms)
        print(f"Wrote {count} PMU samples to {args.export_pmu}")

    df = pd.read_csv(args.csv_path)
    t = df[args.time_column].to_numpy()
    y = df[args.column].to_numpy()

    t_min, t_max = float(t.min()), float(t.max())
    total_span = max(t_max - t_min, 1e-9)
    # pad the amplitude range so a flat/near-flat signal doesn't sit on the axis border
    y_abs_max = max(abs(float(y.min())), abs(float(y.max())), 1e-9) * 1.2

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.28)
    ax.axhline(0, color="0.7", linewidth=0.8, zorder=0)
    (line,) = ax.plot(t, y, linewidth=1, marker=".", markersize=2)
    ax.set_xlabel(args.time_column)
    ax.set_ylabel(args.column)
    ax.set_title(args.csv_path)
    ax.grid(True, alpha=0.3)

    # initial view: full time range, full amplitude range
    init_window = total_span
    init_offset = t_min

    ax_offset = plt.axes([0.15, 0.05, 0.7, 0.03])
    ax_window = plt.axes([0.15, 0.11, 0.7, 0.03])
    ax_yzoom = plt.axes([0.15, 0.17, 0.7, 0.03])

    s_offset = Slider(ax_offset, "Scroll (s)", t_min, t_max, valinit=init_offset)
    s_window = Slider(ax_window, "Time zoom (s)", total_span * 0.001, total_span, valinit=init_window)
    s_yzoom = Slider(ax_yzoom, "Amplitude zoom (x)", 1.0, 50.0, valinit=1.0)

    def update(_event=None):
        window = s_window.val
        offset = min(s_offset.val, t_max - window) if window < total_span else t_min
        offset = max(offset, t_min)

        ax.set_xlim(offset, offset + window)
        ax.set_ylim(-y_abs_max / s_yzoom.val, y_abs_max / s_yzoom.val)
        fig.canvas.draw_idle()

    s_offset.on_changed(update)
    s_window.on_changed(update)
    s_yzoom.on_changed(update)

    update()
    plt.show()


if __name__ == "__main__":
    main()
