import pandas as pd
from multitask_active_controller_v2 import scan_csv

MODEL = "neural_active_controller_v2.joblib"

TESTS = {
    "NORMAL": "../scenario_data/normal_r03.csv",
    "SYNC": "../scenario_data/PMU2_sync_r03.csv",
    "CLOCK_DRIFT": "../scenario_data/PMU2_clock_drift_r03.csv",
    "BAD_DATA": "../scenario_data/PMU2_bad_data_r03.csv",
}

for name, csv_file in TESTS.items():

    output = f"test_{name.lower()}_v2.csv"

    print("\n" + "=" * 70)
    print(f" TESTING: {name}")
    print("=" * 70)

    df = scan_csv(csv_file, MODEL, output)
    df.to_csv(output, index=False)

    print("\nRaw fault:")
    print(df["raw_fault_type"].value_counts().to_string())

    print("\nNeural active fault:")
    print(df["active_fault_type"].value_counts().to_string())

    print("\nFaulty PMU:")
    print(df["active_faulty_pmu"].value_counts().to_string())

    print("\nManagement action:")
    print(df["management_action"].value_counts().to_string())

    print("\nMean neural confidence:")
    print(f"{df['neural_confidence'].mean():.4f}")

    print(f"\nSaved: {output}")


print("\n" + "=" * 70)
print(" ALL V2 TESTS COMPLETE")
print("=" * 70)
