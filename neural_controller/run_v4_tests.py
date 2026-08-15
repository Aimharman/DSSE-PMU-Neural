from multitask_active_controller_v4 import scan_csv
import pandas as pd

MODEL="neural_active_controller_v4.joblib"
TESTS={
 "NORMAL":"../scenario_data/normal_r03.csv",
 "SYNC":"../scenario_data/PMU2_sync_r03.csv",
 "CLOCK_DRIFT":"../scenario_data/PMU2_clock_drift_r03.csv",
 "BAD_DATA":"../scenario_data/PMU2_bad_data_r03.csv",
}

for name,path in TESTS.items():
    print("\n"+"="*72+"\nTESTING V4: "+name+"\n"+"="*72)
    outname=f"test_{name.lower()}_v4.csv"
    df=scan_csv(path,MODEL,outname)
    for col in ["raw_fault_type","active_fault_type","active_faulty_pmu","management_action"]:
        print("\n"+col+":")
        print(df[col].value_counts().to_string())
    print(f"\nMean confidence: {df['neural_confidence'].mean():.4f}")
    print("Saved:",outname)

print("\n"+"="*72+"\nV4 TESTS COMPLETE\n"+"="*72)
