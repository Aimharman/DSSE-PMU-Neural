# Neural Active Fault Management Controller — Conversion Plan

This folder converts the existing PMU/PDC project from a primarily chi-square
fault-identification project into a **Neural Active Fault Management Controller**.

## What remains

- PMU waveform/DFT simulator
- PDC 50-Hz scan concept
- WLS state estimator
- chi-square detector as the conventional baseline
- bad-data, synchronization-error and clock-drift scenarios
- existing live plots and report figures

The existing detector explicitly keeps simulator ground truth out of its
runtime decision path; the ground truth flags are suitable for supervised
training labels only. See the existing simulator's separate `Sync Fault`,
`Clock Drift Fault`, and `Bad Data` fields.

## New neural path

```text
PMU data -> 20-sample PDC window -> measurement-derived features
         -> MLP classifier -> fault class + confidence
         -> active management action -> measurement weights -> WLS
```

The neural network never receives the simulator's fault flags as inputs.
Those flags are used only to label training windows.

## Fault classes

- NORMAL
- PMU1/2/3 BAD_DATA
- PMU1/2/3 SYNC
- PMU1/2/3 CLOCK_DRIFT

Mixed simultaneous faults are excluded from the first single-label model.
They can be introduced after the single-fault classifier is validated.

## Training workflow

1. Generate one normal CSV and nine single-fault CSVs.
2. Put them in `scenario_data/`.
3. Run:

```bash
cd neural_controller
python train_neural_controller.py ../scenario_data/*.csv
```

4. The model is saved as `neural_fault_controller.joblib`.
5. Test the controller on a new CSV:

```bash
python - <<'PY'
from active_controller import scan_csv
out = scan_csv('../scenario_data/PMU2_clock_drift_test.csv',
               'neural_fault_controller.joblib')
print(out.to_string(index=False))
PY
```

## Important implementation rule

Do not train using columns such as:

- `PMU1 Bad Data`
- `PMU2 Sync Fault Active`
- `PMU3 Clock Drift Fault`

Those are labels, not neural inputs. The feature extractor uses measured
voltage/current magnitude, phase, phase differences, short-window statistics,
and slopes instead.

## Active management

The first controller policy is intentionally conservative:

- NORMAL -> accept all PMUs
- BAD_DATA -> down-weight the affected PMU
- SYNC -> down-weight phase data
- CLOCK_DRIFT -> down-weight PMU and request timing check
- low confidence -> hold/request more data

After this works, connect the returned weights to the WLS measurement-weight
matrix and compare neural control against the existing chi-square isolation
path.
