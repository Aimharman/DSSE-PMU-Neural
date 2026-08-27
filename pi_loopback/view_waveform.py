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

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def main():
    parser = argparse.ArgumentParser(description="Interactive viewer for loopback capture CSVs")
    parser.add_argument("csv_path", help="Path to a captured CSV (e.g. pi_loopback/capture.csv)")
    parser.add_argument("--time-column", default="Time (s)", help="Name of the time column")
    parser.add_argument("--column", default="Reconstructed Value", help="Name of the value column to plot")
    args = parser.parse_args()

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
