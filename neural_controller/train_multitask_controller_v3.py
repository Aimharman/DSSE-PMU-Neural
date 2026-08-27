"""Refactor v3: hierarchical Neural Active Fault Management Controller.

Primary neural model:
    NORMAL / BAD_DATA / CLOCK_DRIFT / SYNC

Specialist neural model:
    SYNC vs CLOCK_DRIFT only

The specialist is trained only on timing-fault windows and uses timing-derived
features. It is then used to arbitrate the primary model's timing-fault class.
The PMU classifier remains a separate neural model trained only on fault windows.

Ground-truth flags are used only for training labels/evaluation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_extractor_v2 import build_dataset

RANDOM_STATE = 42
TIMING_CLASSES = {"SYNC", "CLOCK_DRIFT"}

def mlp(hidden=(32,16), max_iter=700):
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )),
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

def balance_indices(y, seed=RANDOM_STATE):
    """Simple deterministic oversampling for the timing specialist."""
    rng = np.random.default_rng(seed)
    parts = []
    counts = y.value_counts()
    target = int(counts.max())
    for cls, count in counts.items():
        idx = np.flatnonzero(y.to_numpy() == cls)
        if count < target:
            idx = rng.choice(idx, size=target, replace=True)
        parts.extend(idx.tolist())
    rng.shuffle(parts)
    return np.asarray(parts, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--model-out", default="neural_active_controller_v3.joblib")
    args = ap.parse_args()

    X, y_raw, meta = build_dataset(args.csv)
    y_type = y_raw.map(fault_type)
    y_pmu = y_raw.map(pmu_label)

    print("=" * 58)
    print(" Neural Active Fault Management - Refactor v3")
    print(" Hierarchical timing-fault specialist")
    print("=" * 58)
    print(f"Total PDC windows : {len(X)}")
    print(f"Feature count     : {X.shape[1]}")
    print("Class distribution:")
    print(y_type.value_counts().sort_index().to_string())

    groups = meta["source"].to_numpy()
    classes = set(y_type.unique())
    splitter = GroupShuffleSplit(n_splits=300, test_size=0.40,
                                 random_state=RANDOM_STATE)
    chosen = None
    for tr, te in splitter.split(X, y_type, groups):
        if classes.issubset(set(y_type.iloc[tr])) and classes.issubset(set(y_type.iloc[te])):
            chosen = (tr, te)
            break
    if chosen is None:
        raise SystemExit("Could not construct a complete group-held-out split.")
    tr, te = chosen

    print("\nHeld-out scenario files:")
    for f in sorted(set(meta.iloc[te].source)):
        print(" ", f)

    # Primary four-class model.
    type_model = mlp()
    type_model.fit(X.iloc[tr], y_type.iloc[tr])
    pred_type = type_model.predict(X.iloc[te])
    type_labels = list(type_model.named_steps["mlp"].classes_)

    print("\nPRIMARY FAULT-TYPE MODEL")
    print(classification_report(y_type.iloc[te], pred_type, digits=4, zero_division=0))
    cm_type = confusion_matrix(y_type.iloc[te], pred_type, labels=type_labels)
    print(pd.DataFrame(cm_type, index=type_labels, columns=type_labels).to_string())

    # Timing specialist. Same held-out scenario groups as primary model.
    timing_mask = y_type.isin(TIMING_CLASSES)
    tr_t = [i for i in tr if timing_mask.iloc[i]]
    te_t = [i for i in te if timing_mask.iloc[i]]
    if not tr_t or not te_t:
        raise SystemExit("Timing specialist has no SYNC/CLOCK_DRIFT samples in train/test.")

    timing_features = [
        c for c in X.columns
        if c.startswith("timing_")
        and any(k in c for k in (
            "slope", "delta", "offset_stability",
            "late_change_abs", "persistent_slope", "mean", "std", "range"
        ))
    ]
    # Keep only finite, actually useful timing columns.
    timing_features = [c for c in timing_features if X[c].notna().all()]
    if not timing_features:
        raise SystemExit("No timing features found for specialist.")

    Xt_tr = X.iloc[tr_t][timing_features].reset_index(drop=True)
    yt_tr = y_type.iloc[tr_t].reset_index(drop=True)
    Xt_te = X.iloc[te_t][timing_features].reset_index(drop=True)
    yt_te = y_type.iloc[te_t].reset_index(drop=True)

    bal = balance_indices(yt_tr)
    timing_model = mlp(hidden=(48,24), max_iter=900)
    timing_model.fit(Xt_tr.iloc[bal], yt_tr.iloc[bal])
    pred_timing = timing_model.predict(Xt_te)

    timing_labels = list(timing_model.named_steps["mlp"].classes_)
    print("\nSYNC vs CLOCK_DRIFT SPECIALIST")
    print(f"Timing feature count: {len(timing_features)}")
    print(classification_report(yt_te, pred_timing, digits=4, zero_division=0))
    cm_timing = confusion_matrix(yt_te, pred_timing, labels=timing_labels)
    print(pd.DataFrame(cm_timing, index=timing_labels, columns=timing_labels).to_string())

    # PMU model, fault windows only.
    fault_mask = y_type != "NORMAL"
    tr_fault = [i for i in tr if fault_mask.iloc[i]]
    te_fault = [i for i in te if fault_mask.iloc[i]]
    pmu_model = mlp()
    pmu_model.fit(X.iloc[tr_fault], y_pmu.iloc[tr_fault])
    pred_pmu = pmu_model.predict(X.iloc[te_fault])
    pmu_labels = list(pmu_model.named_steps["mlp"].classes_)

    print("\nFAULTY PMU MODEL")
    print(classification_report(y_pmu.iloc[te_fault], pred_pmu, digits=4, zero_division=0))
    cm_pmu = confusion_matrix(y_pmu.iloc[te_fault], pred_pmu, labels=pmu_labels)
    print(pd.DataFrame(cm_pmu, index=pmu_labels, columns=pmu_labels).to_string())

    bundle = {
        "type_model": type_model,
        "timing_model": timing_model,
        "pmu_model": pmu_model,
        "feature_names": list(X.columns),
        "timing_feature_names": timing_features,
        "classes_type": type_labels,
        "classes_timing": timing_labels,
        "classes_pmu": pmu_labels,
        "random_state": RANDOM_STATE,
        "window_samples": 20,
        "pdc_rate_hz": 50.0,
        "feature_extractor_version": "v2",
        "controller_version": "v3_hierarchical",
        "timing_recent_samples": 100,
        "timing_long_samples": 200,
    }
    joblib.dump(bundle, args.model_out)

    metrics = {
        "controller_version": "v3_hierarchical",
        "windows": int(len(X)),
        "train_windows": int(len(tr)),
        "test_windows": int(len(te)),
        "feature_count": int(X.shape[1]),
        "timing_feature_count": int(len(timing_features)),
        "primary_type_confusion_matrix": cm_type.tolist(),
        "timing_confusion_matrix": cm_timing.tolist(),
        "pmu_confusion_matrix": cm_pmu.tolist(),
        "timing_classes": timing_labels,
        "type_classes": type_labels,
        "pmu_classes": pmu_labels,
    }
    Path(args.model_out).with_suffix(".json").write_text(
        json.dumps(metrics, indent=2)
    )
    meta.to_csv(Path(args.model_out).with_suffix(".dataset.csv"), index=False)
    print(f"\nSaved: {args.model_out}")

if __name__ == "__main__":
    main()
