#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python generator/pmu_generator.py --scenario normal --replicate 1 --seed 42
python generator/pmu_generator.py --scenario bad_data --pmu 1 --replicate 1 --seed 42
python generator/pmu_generator.py --scenario sync --pmu 2 --replicate 1 --seed 42
python generator/pmu_generator.py --scenario clock_drift --pmu 3 --replicate 1 --seed 42
