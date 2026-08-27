"""Train the primary multitask Neural Active Fault Management Controller.

The controller learns two related tasks from measurement-derived features:
    1. fault type: NORMAL / BAD_DATA / SYNC / CLOCK_DRIFT
    2. faulty PMU: PMU1 / PMU2 / PMU3, trained only on fault windows

Ground-truth simulator columns are used only to construct labels.
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

from feature_extractor import build_dataset

RANDOM_STATE = 42


def make_mlp():
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


def choose_group_split(X, y_type, y_pmu, groups):
    """Find a scenario-held-out split containing every relevant class."""
    type_classes = set(y_type.unique())
    pmu_classes = set(y_pmu[y_type != "NORMAL"].unique())

    splitter = GroupShuffleSplit(
        n_splits=500, test_size=0.40, random_state=RANDOM_STATE
    )

    for train_idx, test_idx in splitter.split(X, y_type, groups=groups):
        train_type = set(y_type.iloc[train_idx].unique())
        test_type = set(y_type.iloc[test_idx].unique())

        train_fault_pmu = set(
            y_pmu.iloc[train_idx][y_type.iloc[train_idx] != "NORMAL"].unique()
        )
        test_fault_pmu = set(
            y_pmu.iloc[test_idx][y_type.iloc[test_idx] != "NORMAL"].unique()
        )

        if (
            type_classes.issubset(train_type)
            and type_classes.issubset(test_type)
            and pmu_classes.issubset(train_fault_pmu)
            and pmu_classes.issubset(test_fault_pmu)
        ):
            return train_idx, test_idx

    raise SystemExit(
        "Could not construct a scenario-held-out split containing all "
        "fault types and PMU classes in both train and test sets."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument(
        "--model-out",
        default="neural_active_controller.joblib",
    )
    args = ap.parse_args()

    X, y, meta = build_dataset(args.csv)

    if len(X) == 0:
        raise SystemExit("No valid PDC windows were extracted.")

    y_type = y.map(fault_type)
    y_pmu = y.map(pmu_label)
    groups = meta["source"].to_numpy()

    train_idx, test_idx = choose_group_split(
        X, y_type, y_pmu, groups
    )

    print("==============================================")
    print(" Neural Active Fault Management Controller")
    print("==============================================")
    print(f"Total PDC windows : {len(X)}")
    print(f"Train windows     : {len(train_idx)}")
    print(f"Test windows      : {len(test_idx)}")
    print()
    print("Held-out scenario files:")
    for src in sorted(set(meta.iloc[test_idx]["source"])):
        print(f"  {src}")

    type_model = make_mlp()
    type_model.fit(X.iloc[train_idx], y_type.iloc[train_idx])
    type_pred = type_model.predict(X.iloc[test_idx])

    print("\nFAULT TYPE MODEL")
    print(
        classification_report(
            y_type.iloc[test_idx],
            type_pred,
            digits=4,
            zero_division=0,
        )
    )

    type_labels = list(type_model.named_steps["mlp"].classes_)
    type_cm = confusion_matrix(
        y_type.iloc[test_idx],
        type_pred,
        labels=type_labels,
    )
    print("Confusion matrix (rows=true, columns=predicted)")
    print(
        pd.DataFrame(
            type_cm,
            index=type_labels,
            columns=type_labels,
        ).to_string()
    )

    train_fault = [
        i for i in train_idx if y_type.iloc[i] != "NORMAL"
    ]
    test_fault = [
        i for i in test_idx if y_type.iloc[i] != "NORMAL"
    ]

    pmu_model = make_mlp()
    pmu_model.fit(X.iloc[train_fault], y_pmu.iloc[train_fault])
    pmu_pred = pmu_model.predict(X.iloc[test_fault])

    print("\nFAULTY PMU MODEL")
    print(
        classification_report(
            y_pmu.iloc[test_fault],
            pmu_pred,
            digits=4,
            zero_division=0,
        )
    )

    pmu_labels = list(pmu_model.named_steps["mlp"].classes_)
    pmu_cm = confusion_matrix(
        y_pmu.iloc[test_fault],
        pmu_pred,
        labels=pmu_labels,
    )
    print("Confusion matrix (rows=true, columns=predicted)")
    print(
        pd.DataFrame(
            pmu_cm,
            index=pmu_labels,
            columns=pmu_labels,
        ).to_string()
    )

    bundle = {
        "type_model": type_model,
        "pmu_model": pmu_model,
        "feature_names": list(X.columns),
        "classes_type": type_labels,
        "classes_pmu": pmu_labels,
        "window_samples": 20,
        "pdc_rate_hz": 50.0,
        "activation_windows": 2,
        "recovery_windows": 3,
        "confidence_threshold": 0.70,
        "model_description": "Two-stage MLP: fault type + affected PMU",
    }

    joblib.dump(bundle, args.model_out)

    metrics = {
        "windows": int(len(X)),
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "type_classes": type_labels,
        "pmu_classes": pmu_labels,
        "model": "two-stage MLP(32,16)",
        "feature_count": int(X.shape[1]),
        "type_confusion_matrix": type_cm.tolist(),
        "pmu_confusion_matrix": pmu_cm.tolist(),
        "activation_windows": 2,
        "recovery_windows": 3,
        "confidence_threshold": 0.70,
    }

    Path(args.model_out).with_suffix(".json").write_text(
        json.dumps(metrics, indent=2)
    )

    print(f"\nModel saved: {args.model_out}")


if __name__ == "__main__":
    main()
