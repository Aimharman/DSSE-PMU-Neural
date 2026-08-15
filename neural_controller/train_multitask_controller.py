"""Train a two-stage neural controller: fault type + faulty PMU."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_extractor import build_dataset

RANDOM_STATE = 42

def mlp():
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(32,16), activation="relu",
                              solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                              max_iter=500, early_stopping=False,
                              random_state=RANDOM_STATE))
    ])

def fault_type(label):
    if label == "NORMAL": return "NORMAL"
    if "BAD_DATA" in label: return "BAD_DATA"
    if "CLOCK_DRIFT" in label: return "CLOCK_DRIFT"
    if "SYNC" in label: return "SYNC"
    return "MIXED"

def pmu_label(label):
    if label == "NORMAL": return "NONE"
    return label.split("_")[0]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("csv", nargs="+"); ap.add_argument("--model-out", default="neural_active_controller.joblib")
    args=ap.parse_args()
    X,y,meta=build_dataset(args.csv)
    y_type=y.map(fault_type); y_pmu=y.map(pmu_label)
    groups=meta.source.to_numpy(); classes=set(y_type.unique())
    splitter=GroupShuffleSplit(n_splits=200,test_size=0.40,random_state=RANDOM_STATE)
    chosen=None
    for tr,te in splitter.split(X,y_type,groups):
        if classes.issubset(set(y_type.iloc[tr])) and classes.issubset(set(y_type.iloc[te])):
            chosen=(tr,te); break
    if chosen is None: raise SystemExit("Could not make group-held-out split with every fault type in both sets.")
    tr,te=chosen
    print("Held-out files:")
    for f in sorted(set(meta.iloc[te].source)): print(" ",f)

    type_model=mlp(); type_model.fit(X.iloc[tr], y_type.iloc[tr]); p_type=type_model.predict(X.iloc[te])
    print("\nFAULT TYPE MODEL")
    print(classification_report(y_type.iloc[te],p_type,digits=4,zero_division=0))
    print(pd.DataFrame(confusion_matrix(y_type.iloc[te],p_type,labels=type_model.named_steps['mlp'].classes_),index=type_model.named_steps['mlp'].classes_,columns=type_model.named_steps['mlp'].classes_).to_string())

    # PMU classifier is trained only on fault windows.
    fault_mask=(y_type!="NORMAL")
    tr_fault=[i for i in tr if fault_mask.iloc[i]]
    te_fault=[i for i in te if fault_mask.iloc[i]]
    pmu_model=mlp(); pmu_model.fit(X.iloc[tr_fault], y_pmu.iloc[tr_fault]); p_pmu=pmu_model.predict(X.iloc[te_fault])
    print("\nFAULTY PMU MODEL")
    print(classification_report(y_pmu.iloc[te_fault],p_pmu,digits=4,zero_division=0))

    bundle={"type_model":type_model,"pmu_model":pmu_model,"feature_names":list(X.columns),"classes_type":list(type_model.named_steps['mlp'].classes_),"classes_pmu":list(pmu_model.named_steps['mlp'].classes_),"window_samples":20,"pdc_rate_hz":50.0}
    joblib.dump(bundle,args.model_out)
    Path(args.model_out).with_suffix('.json').write_text(json.dumps({"windows":len(X),"type_classes":list(type_model.named_steps['mlp'].classes_),"pmu_classes":list(pmu_model.named_steps['mlp'].classes_)},indent=2))
    print("\nSaved",args.model_out)

if __name__=="__main__": main()
