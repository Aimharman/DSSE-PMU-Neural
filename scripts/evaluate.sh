#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m evaluation.evaluate_windows data/scenarios/normal_pmu1_r01.csv
