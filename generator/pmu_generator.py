#!/usr/bin/env python3
"""Canonical PMU scenario generator for the final DSSE + neural project.

This script wires the canonical simulator in the repository root to a CLI-based
scenario selection API:

    python generator/pmu_generator.py --scenario normal --replicate 1
    python generator/pmu_generator.py --scenario sync --pmu 2 --replicate 1
    python generator/pmu_generator.py --scenario clock_drift --pmu 3 --replicate 1
    python generator/pmu_generator.py --scenario bad_data --pmu 1 --replicate 1

The implementation intentionally uses the project’s canonical simulator as the
source of waveform generation, one-cycle DFT, and metadata fields while keeping
headless batch execution the default mode.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIMULATOR_PATH = ROOT / "generator" / "pmu_simulator_fault_refactored_timing_separated_voltage_window.py"


def load_simulator():
    spec = importlib.util.spec_from_file_location("pmu_canonical_simulator", SIMULATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def configure_scenario(module, scenario: str, pmu: int = 1, seed: int | None = None):
    np.random.seed(seed if seed is not None else 0)
    module.np.random.seed(seed if seed is not None else 0)

    # Default healthy configuration.
    module.ENABLE_SYNC_ERROR = True
    module.ENABLE_MEASUREMENT_NOISE = True
    module.ENABLE_PACKET_LOSS = False
    module.ENABLE_BAD_DATA = False
    module.ENABLE_SYNC_FAULT = False
    module.ENABLE_CLOCK_DRIFT = False
    module.SHOW_PLOT = False

    module.SYNC_FAULT_PMU = 2
    module.SYNC_FAULT_START_TIME = 2.0
    module.SYNC_FAULT_END_TIME = 6.0
    module.SYNC_FAULT_PHASE_ERROR = 10.0

    module.CLOCK_DRIFT_PMU = 2
    module.CLOCK_DRIFT_START_TIME = 2.0
    module.CLOCK_DRIFT_END_TIME = 8.0
    module.CLOCK_DRIFT_PPM = 1000.0

    module.BAD_DATA_MODE = "faulty_pmu"
    module.BAD_PMU = 1
    module.FAULTY_PMU = pmu
    module.FAULT_START_TIME = 2.0
    module.FAULT_END_TIME = module.SIMULATION_TIME
    module.FAULT_PHASE_ERROR = 20.0
    module.FAULT_MAG_SCALE = 1.20

    module.PMU1_SYNC_OFFSET = float(np.random.normal(0.0, module.SYNC_STD_DEG))
    module.PMU2_SYNC_OFFSET = float(np.random.normal(0.0, module.SYNC_STD_DEG))
    module.PMU3_SYNC_OFFSET = float(np.random.normal(0.0, module.SYNC_STD_DEG))

    if scenario.lower() == "normal":
        return
    if scenario.lower() == "sync":
        module.ENABLE_SYNC_FAULT = True
        module.SYNC_FAULT_PMU = int(pmu)
        return
    if scenario.lower() == "clock_drift":
        module.ENABLE_CLOCK_DRIFT = True
        module.CLOCK_DRIFT_PMU = int(pmu)
        return
    if scenario.lower() == "bad_data":
        module.ENABLE_BAD_DATA = True
        module.BAD_DATA_MODE = "faulty_pmu"
        module.FAULTY_PMU = int(pmu)
        return
    raise ValueError(f"Unsupported scenario: {scenario}")


def generate_scenario(scenario: str, output_path: str | Path, pmu: int = 1, seed: int | None = None):
    module = load_simulator()
    configure_scenario(module, scenario=scenario, pmu=pmu, seed=seed)
    module.CSV_PATH = str(output_path)
    module.sample_index = 0
    module.csv_file = None
    module.csv_writer = None
    module.setup_csv()
    while module.sample_index < module.TOTAL_SAMPLES:
        module.generate_one_sample()
    module.finish_simulation()
    return Path(output_path)


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate canonical PMU scenarios for DSSE + neural research.")
    parser.add_argument("--scenario", choices=["normal", "sync", "clock_drift", "bad_data"], default="normal")
    parser.add_argument("--pmu", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    scenario_name = args.scenario
    replicate = args.replicate
    output = Path(args.output) if args.output else ROOT / "data" / "scenarios" / f"{scenario_name}_pmu{args.pmu}_r{replicate:02d}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.headless:
        output = output
    generate_scenario(scenario_name, output, pmu=args.pmu, seed=args.seed + replicate)
    print(f"Generated scenario: {scenario_name} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
