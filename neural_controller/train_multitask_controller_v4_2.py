# Neural Active Fault Management Controller - V4.2
"""V4.1 training: gated neural controller with explicit timing physics features."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_extractor_v4_1 import build_dataset_v41

RANDOM_STATE = 42
TIMING = {"SYNC", "CLOCK_DRIFT"}

def mlp(hidden=(48,24), max_iter=1000):
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
             solver="adam", alpha=1e-4, learning_rate_init=1e-3,
             max_iter=max_iter, early_stopping=True, validation_fraction=0.15,
             n_iter_no_change=35, random_state=RANDOM_STATE))
    ])

def fault_type(label):
    if label == "NORMAL": return "NORMAL"
    if "BAD_DATA" in label: return "BAD_DATA"
    if "SYNC" in label: return "SYNC"
    if "CLOCK_DRIFT" in label: return "CLOCK_DRIFT"
    return "MIXED"

def pmu_label(label):
    if label == "NORMAL": return "NONE"
    return label.split("_")[0]

def group_split(X, y, meta):
    groups = meta["source"].to_numpy()
    classes = set(y.unique())
    sp = GroupShuffleSplit(n_splits=500, test_size=.40, random_state=RANDOM_STATE)
    for tr, te in sp.split(X, y, groups):
        if classes.issubset(set(y.iloc[tr])) and classes.issubset(set(y.iloc[te])):
            return tr, te
    raise SystemExit("Could not create complete group-held-out split.")

def report(title, y, pred, labels):
    print("\n" + title)
    print(classification_report(y, pred, digits=4, zero_division=0))
    cm = confusion_matrix(y, pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    return cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--model-out", default="neural_active_controller_v42.joblib")
    args = ap.parse_args()

    X, yraw, meta = build_dataset_v41(args.csv)
    y = yraw.map(fault_type)
    yp = yraw.map(pmu_label)

    print("="*64)
    print(" Neural Active Fault Management - Refactor v4.2")
    print(" Targeted SYNC vs CLOCK_DRIFT refinement")
    print("="*64)
    print(f"Total PDC windows : {len(X)}")
    print(f"Feature count     : {X.shape[1]}")
    print("Class distribution:")
    print(y.value_counts().sort_index().to_string())

    tr, te = group_split(X, y, meta)
    print("\nHeld-out scenario files:")
    for f in sorted(set(meta.iloc[te].source)): print(" ", f)

    state_y = y.map(lambda z: "NORMAL" if z == "NORMAL" else "FAULT")
    state_model = mlp((32,16), 800)
    state_model.fit(X.iloc[tr], state_y.iloc[tr])
    cm_state = report("NORMAL / FAULT GATE",
                      state_y.iloc[te], state_model.predict(X.iloc[te]),
                      ["FAULT","NORMAL"])

    fault_mask = y != "NORMAL"
    trf = [i for i in tr if fault_mask.iloc[i]]
    tef = [i for i in te if fault_mask.iloc[i]]
    bt_y = y.map(lambda z: "BAD_DATA" if z == "BAD_DATA" else
                 ("TIMING" if z in TIMING else "OTHER"))
    bt_model = mlp((40,20), 900)
    bt_model.fit(X.iloc[trf], bt_y.iloc[trf])
    cm_bt = report("BAD_DATA / TIMING GATE",
                   bt_y.iloc[tef], bt_model.predict(X.iloc[tef]),
                   ["BAD_DATA","TIMING"])

    timing_cols = [c for c in X.columns if c.startswith("timing_")
                   or c.startswith("v4_") or c.startswith("v41_")]
    timing_mask = y.isin(TIMING)
    trt = [i for i in tr if timing_mask.iloc[i]]
    tet = [i for i in te if timing_mask.iloc[i]]
    tm = mlp((56,28), 1200)

    yy = y.iloc[trt].reset_index(drop=True)
    Xt = X.iloc[trt][timing_cols].reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_STATE)
    parts = []
    target = yy.value_counts().max()
    for cls, n in yy.value_counts().items():
        idx = np.flatnonzero(yy.to_numpy() == cls)
        if n < target:
            idx = rng.choice(idx, size=target, replace=True)
        parts.extend(idx.tolist())
    rng.shuffle(parts)
    tm.fit(Xt.iloc[parts], yy.iloc[parts])
    cm_t = report("SYNC / CLOCK_DRIFT SPECIALIST",
                  y.iloc[tet], tm.predict(X.iloc[tet][timing_cols]),
                  ["CLOCK_DRIFT","SYNC"])

    pm = mlp((48,24), 900)
    pm.fit(X.iloc[trf], yp.iloc[trf])
    cm_p = report("FAULTY PMU MODEL", yp.iloc[tef], pm.predict(X.iloc[tef]),
                  ["PMU1","PMU2","PMU3"])

    bundle = {
      "state_model": state_model, "bad_timing_model": bt_model,
      "timing_model": tm, "pmu_model": pm,
      "feature_names": list(X.columns),
      "timing_feature_names": timing_cols,
      "classes_state": list(state_model.named_steps["mlp"].classes_),
      "classes_bad_timing": list(bt_model.named_steps["mlp"].classes_),
      "classes_timing": list(tm.named_steps["mlp"].classes_),
      "classes_pmu": list(pm.named_steps["mlp"].classes_),
      "controller_version": "v4.1_gated_physics_timing",
      "feature_extractor_version": "v4.1",
      "window_samples": 20, "pdc_rate_hz": 50.0,
    }
    joblib.dump(bundle, args.model_out)
    meta.to_csv(Path(args.model_out).with_suffix(".dataset.csv"), index=False)
    Path(args.model_out).with_suffix(".json").write_text(json.dumps({
      "controller_version": "v4.1_gated_physics_timing",
      "windows": len(X), "features": X.shape[1],
      "state_cm": cm_state.tolist(), "bad_timing_cm": cm_bt.tolist(),
      "timing_cm": cm_t.tolist(), "pmu_cm": cm_p.tolist()
    }, indent=2))
    print("\nSaved:", args.model_out)

if __name__ == "__main__":
    main()
