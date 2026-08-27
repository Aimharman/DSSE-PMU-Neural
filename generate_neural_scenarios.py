"""Headless supervised-scenario generator built on the existing PMU simulator.

The real GUI simulator remains unchanged. This wrapper imports its numerical
simulation engine, disables plotting, and creates randomized single-fault CSVs
for supervised neural-network training.
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import types
from pathlib import Path

BASE = Path(__file__).resolve().parent
SIM = BASE / "pmu_simulator_fault_refactored_timing_separated(3).py"
OUT = BASE / "scenario_data"

# Lightweight GUI stubs for headless dataset generation. The user's GUI
# simulator still uses the real PyQt6/pyqtgraph modules when run normally.
if "pyqtgraph" not in sys.modules:
    pg = types.ModuleType("pyqtgraph")
    pg.GraphicsLayoutWidget = object
    pg.mkPen = lambda *a, **k: None
    pg.setConfigOptions = lambda *a, **k: None
    sys.modules["pyqtgraph"] = pg

if "PyQt6" not in sys.modules:
    qt6 = types.ModuleType("PyQt6")
    qtw = types.ModuleType("PyQt6.QtWidgets")
    qtc = types.ModuleType("PyQt6.QtCore")

    class _Dummy:
        def __init__(self, *a, **k): pass
        def addWidget(self, *a, **k): pass
        def setMinimum(self, *a, **k): pass
        def setMaximum(self, *a, **k): pass
        def setSingleStep(self, *a, **k): pass
        def setPageStep(self, *a, **k): pass
        def setWindowTitle(self, *a, **k): pass
        def resize(self, *a, **k): pass
        def show(self): pass
        def raise_(self): pass
        def activateWindow(self): pass

    qtw.QApplication = _Dummy
    qtw.QWidget = _Dummy
    qtw.QVBoxLayout = _Dummy
    qtw.QScrollBar = _Dummy

    class _Qt:
        class Orientation:
            Horizontal = 1

    qtc.Qt = _Qt
    qtc.QTimer = _Dummy
    sys.modules["PyQt6"] = qt6
    sys.modules["PyQt6.QtWidgets"] = qtw
    sys.modules["PyQt6.QtCore"] = qtc


class DummyTimer:
    def stop(self):
        pass


def fresh_module():
    spec = importlib.util.spec_from_file_location("pmu_sim_engine", SIM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(m, kind, pmu, duration, start, end, rng):
    m.SIMULATION_TIME = float(duration)
    m.TOTAL_SAMPLES = int(round(duration * m.ODR))
    m.CSV_PATH = str(OUT / "CURRENT_SCENARIO.csv")

    # Clean single-fault training conditions.
    m.ENABLE_SYNC_ERROR = False
    m.ENABLE_MEASUREMENT_NOISE = True
    m.ENABLE_PACKET_LOSS = False
    m.ENABLE_BAD_DATA = False
    m.ENABLE_SYNC_FAULT = False
    m.ENABLE_CLOCK_DRIFT = False

    if kind == "bad_data":
        m.ENABLE_BAD_DATA = True
        m.BAD_DATA_MODE = "faulty_pmu"
        m.FAULTY_PMU = pmu
        m.FAULT_START_TIME = start
        m.FAULT_END_TIME = end
        m.FAULT_PHASE_ERROR = rng.uniform(5.0, 20.0)
        m.FAULT_MAG_SCALE = rng.uniform(1.05, 1.25)

    elif kind == "sync":
        m.ENABLE_SYNC_FAULT = True
        m.SYNC_FAULT_PMU = pmu
        m.SYNC_FAULT_START_TIME = start
        m.SYNC_FAULT_END_TIME = end
        m.SYNC_FAULT_PHASE_ERROR = rng.uniform(4.0, 20.0)

    elif kind == "clock_drift":
        m.ENABLE_CLOCK_DRIFT = True
        m.CLOCK_DRIFT_PMU = pmu
        m.CLOCK_DRIFT_START_TIME = start
        m.CLOCK_DRIFT_END_TIME = end
        m.CLOCK_DRIFT_PPM = rng.uniform(100.0, 1200.0)


def run_one(kind, pmu, duration, start, end, rng, output_path):
    m = fresh_module()
    configure(m, kind, pmu, duration, start, end, rng)
    m.CSV_PATH = str(output_path)
    OUT.mkdir(parents=True, exist_ok=True)

    m.sample_index = 0
    m.csv_file = None
    m.csv_writer = None
    m.time_data = []
    m.current_data = [[], [], []]
    m.magnitude_data = [[], [], []]
    m.phase_data = [[], [], []]
    m.current_buffers = [m.deque(maxlen=m.N) for _ in range(3)]
    m.voltage_buffers = [m.deque(maxlen=m.N) for _ in range(3)]
    m.fault_sample_count = [0, 0, 0]
    m.random_spike_state = {1: None, 2: None, 3: None}
    m.random_event_remaining = 0
    m.random_event_pmu = None
    m.random_event_fault_type = None
    m.random_event_count = 0
    m.random_event_start_sample = None
    m.BAD_PMU = None

    m.update_plot = lambda: None
    m.timer = DummyTimer()
    m.setup_csv()
    while m.sample_index < m.TOTAL_SAMPLES:
        m.generate_one_sample()

    print(f"Generated {output_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--repeats", type=int, default=3,
                    help="randomized repetitions per fault class/PMU")
    ap.add_argument("--seed", type=int, default=20260815)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    # One healthy reference plus randomized single-fault scenarios.
    scenarios = [("normal", 0, rep + 1) for rep in range(max(2, args.repeats))]
    for kind in ("bad_data", "sync", "clock_drift"):
        for pmu in (1, 2, 3):
            for rep in range(args.repeats):
                scenarios.append((kind, pmu, rep + 1))

    for kind, pmu, rep in scenarios:
        if kind == "normal":
            start, end = 0.0, 0.0
            out = OUT / f"normal_r{rep:02d}.csv"
            run_one(kind, 1, args.duration, start, end, rng, out)
            continue

        # Randomize event timing while keeping a healthy lead-in and tail.
        start = rng.uniform(0.7, 1.4)
        end = rng.uniform(2.5, min(args.duration - 0.3, 3.4))
        if end <= start + 0.8:
            end = start + 1.0
        out = OUT / f"PMU{pmu}_{kind}_r{rep:02d}.csv"
        run_one(kind, pmu, args.duration, start, end, rng, out)

    print(f"\nGenerated {len(scenarios)} CSV scenarios in {OUT}")


if __name__ == "__main__":
    main()
