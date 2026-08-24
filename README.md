# DSSE-PMU-Neural

A hybrid PMU fault-detection and state-estimation workflow built around a neural controller and a classical weighted least-squares (WLS) pipeline.

## What this project includes

- Neural fault type classification: NORMAL, BAD_DATA, SYNC, CLOCK_DRIFT
- Faulty-PMU localization: PMU1, PMU2, PMU3
- WLS + chi-squared residual validation for bad-data detection
- Decision fusion combining neural and statistical evidence
- Scenario-driven training and evaluation against real PMU CSV files

## Active scenario corpus

The working dataset is stored in `data/scenarios/` and contains the canonical 30 real scenarios:

- 3 normal runs
- 9 BAD_DATA cases (PMU1/2/3 x 3 repeats)
- 9 SYNC cases (PMU1/2/3 x 3 repeats)
- 9 CLOCK_DRIFT cases (PMU1/2/3 x 3 repeats)

The active training and evaluation pipeline uses this canonical dataset only.

## Training

```bash
python3 neural/train.py
```

This regenerates:

- `neural/model/controller.joblib`
- `neural/model/metadata.json`

The saved model metadata explicitly contains:

- fault classes: ["NORMAL", "BAD_DATA", "SYNC", "CLOCK_DRIFT"]
- PMU classes: [1, 2, 3]

## Evaluation

```bash
pytest -q
python3 -m evaluation.evaluate_windows data/scenarios/PMU2_clock_drift_r01.csv
```

## Key files

- `neural/train.py` — grouped scenario-level train/validation split
- `neural/controller.py` — controller load and prediction logic
- `evaluation/evaluate_windows.py` — blind sliding-window evaluation
- `generator/pmu_generator.py` — PMU data generator
- `dsse/wls.py` and `chi_squared_test.py` — state-estimation and anomaly checks

## Notes

- The model is trained with grouped splits by scenario type and PMU to avoid leakage across repeated runs of the same underlying fault mode.
- The validation set is constructed from real PMU scenarios covering all required fault classes and PMU IDs.
