"""Train the two-stage Neural Active Fault Management Controller using
feature_extractor_v2.

Usage:
    python3 train_multitask_controller_v2.py ../scenario_data/*.csv

The model is intentionally saved under a new name so the v1 model remains
available for comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extractor_v2 import build_dataset


RANDOM_STATE = 42


def mlp():
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )),
    ])


def fault_type(label):
    if label == "NORMAL":
        return "NORMAL"
    if "BAD_DATA" in label:
        return "BAD_DATA"
    if "CLOCK_DRIFT" in label:
        return "CLOCK_DRIFT"
    if "SYNC" in label:
        return "SYNC"
    return "MIXED"


def pmu_label(label):
    if label == "NORMAL":
        return "NONE"
    return label.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument(
        "--model-out",
        default="neural_active_controller_v2.joblib",
    )
    args = ap.parse_args()

    X, y, meta = build_dataset(args.csv)

    if len(X) == 0:
        raise SystemExit("No valid PDC windows were extracted.")

    y_type = y.map(fault_type)
    y_pmu = y.map(pmu_label)

    print("=" * 46)
    print(" Neural Active Fault Management - Refactor v2")
    print("=" * 46)
    print(f"Total PDC windows : {len(X)}")
    print(f"Feature count     : {X.shape[1]}")
    print("Class distribution:")
    print(y_type.value_counts().sort_index().to_string())

    groups = meta["source"].to_numpy()
    classes = set(y_type.unique())

    splitter = GroupShuffleSplit(
        n_splits=200,
        test_size=0.40,
        random_state=RANDOM_STATE,
    )

    chosen = None

    for tr, te in splitter.split(X, y_type, groups):
        train_classes = set(y_type.iloc[tr])
        test_classes = set(y_type.iloc[te])

        if classes.issubset(train_classes) and classes.issubset(test_classes):
            chosen = (tr, te)
            break

    if chosen is None:
        raise SystemExit(
            "Could not make a group-held-out split with every fault "
            "type in both sets."
        )

    tr, te = chosen

    print("\nHeld-out scenario files:")
    for f in sorted(set(meta.iloc[te].source)):
        print(" ", f)

    type_model = mlp()
    type_model.fit(X.iloc[tr], y_type.iloc[tr])
    p_type = type_model.predict(X.iloc[te])

    type_labels = list(type_model.named_steps["mlp"].classes_)

    print("\nFAULT TYPE MODEL")
    print(classification_report(
        y_type.iloc[te],
        p_type,
        digits=4,
        zero_division=0,
    ))

    cm_type = confusion_matrix(
        y_type.iloc[te],
        p_type,
        labels=type_labels,
    )

    print("Confusion matrix (rows=true, columns=predicted)")
    print(pd.DataFrame(
        cm_type,
        index=type_labels,
        columns=type_labels,
    ).to_string())

    # PMU classifier is trained only on fault windows.
    fault_mask = y_type != "NORMAL"
    tr_fault = [i for i in tr if fault_mask.iloc[i]]
    te_fault = [i for i in te if fault_mask.iloc[i]]

    pmu_model = mlp()
    pmu_model.fit(X.iloc[tr_fault], y_pmu.iloc[tr_fault])
    p_pmu = pmu_model.predict(X.iloc[te_fault])

    pmu_labels = list(pmu_model.named_steps["mlp"].classes_)

    print("\nFAULTY PMU MODEL")
    print(classification_report(
        y_pmu.iloc[te_fault],
        p_pmu,
        digits=4,
        zero_division=0,
    ))

    cm_pmu = confusion_matrix(
        y_pmu.iloc[te_fault],
        p_pmu,
        labels=pmu_labels,
    )

    print("Confusion matrix (rows=true, columns=predicted)")
    print(pd.DataFrame(
        cm_pmu,
        index=pmu_labels,
        columns=pmu_labels,
    ).to_string())

    bundle = {
        "type_model": type_model,
        "pmu_model": pmu_model,
        "feature_names": list(X.columns),
        "classes_type": type_labels,
        "classes_pmu": pmu_labels,
        "random_state": RANDOM_STATE,
        "window_samples": 20,
        "pdc_rate_hz": 50.0,
        "feature_extractor_version": "v2",
        "timing_recent_samples": 100,
        "timing_long_samples": 200,
    }

    joblib.dump(bundle, args.model_out)

    metrics = {
        "windows": int(len(X)),
        "train_windows": int(len(tr)),
        "test_windows": int(len(te)),
        "feature_count": int(X.shape[1]),
        "feature_extractor_version": "v2",
        "type_classes": type_labels,
        "pmu_classes": pmu_labels,
        "type_confusion_matrix": cm_type.tolist(),
        "pmu_confusion_matrix": cm_pmu.tolist(),
    }

    Path(args.model_out).with_suffix(".json").write_text(
        json.dumps(metrics, indent=2)
    )

    meta.to_csv(
        Path(args.model_out).with_suffix(".dataset.csv"),
        index=False,
    )

    print(f"\nSaved: {args.model_out}")


if __name__ == "__main__":
    main()
