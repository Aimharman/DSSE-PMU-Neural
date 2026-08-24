# PMU generator

The canonical PMU scenario generator is implemented as a headless CLI around the project’s simulator at `pmu_simulator_fault_refactored_timing_separated_voltage_window.py`.

## Examples

```bash
python generator/pmu_generator.py --scenario normal --replicate 1
python generator/pmu_generator.py --scenario sync --pmu 2 --replicate 1
python generator/pmu_generator.py --scenario clock_drift --pmu 3 --replicate 1
python generator/pmu_generator.py --scenario bad_data --pmu 1 --replicate 1
```

## Output
Outputs are written under `data/scenarios/` by default.
