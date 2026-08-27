"""Train the Neural Active Fault Management Controller.

Usage:
    python train_neural_controller.py scenario_data/*.csv

The trained model is saved as neural_fault_controller.joblib.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extractor import build_dataset

RANDOM_STATE = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="Simulator CSV files")
    ap.add_argument("--include-mixed", action="store_true")
    ap.add_argument("--model-out", default="neural_fault_controller.joblib")
    args = ap.parse_args()

    X, y, meta = build_dataset(args.csv, include_mixed=args.include_mixed)
    if len(X) == 0:
        raise SystemExit("No valid PDC windows were extracted.")

    print("==============================================")
    print(" Neural Active Fault Management - Training")
    print("==============================================")
    print(f"PDC windows : {len(X)}")
    print("Class distribution:")
    print(y.value_counts().sort_index().to_string())

    # Stratification requires at least two examples per class.
    counts = y.value_counts()
    if (counts < 2).any():
        raise SystemExit("Every class needs at least 2 PDC windows. Generate more scenarios.")

    # Split by complete scenario file, not by individual windows. This prevents
    # windows from the same simulated event appearing in both train and test.
    groups = meta["source"].to_numpy()
    unique_classes = set(y.unique())
    splitter = GroupShuffleSplit(n_splits=200, test_size=0.40, random_state=RANDOM_STATE)
    chosen = None
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        train_classes = set(y.iloc[train_idx].unique())
        test_classes = set(y.iloc[test_idx].unique())
        if unique_classes.issubset(train_classes) and unique_classes.issubset(test_classes):
            chosen = (train_idx, test_idx)
            break
    if chosen is None:
        raise SystemExit("Could not construct a group-held-out split containing every class in both sets.")

    train_idx, test_idx = chosen
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print("Held-out scenario files:")
    for src in sorted(set(meta.iloc[test_idx]["source"])):
        print(f"  {src}")

    model = Pipeline([
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

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\nClassification report")
    print(classification_report(y_test, pred, digits=4, zero_division=0))

    labels = list(model.named_steps["mlp"].classes_)
    cm = confusion_matrix(y_test, pred, labels=labels)
    print("Confusion matrix (rows=true, columns=predicted)")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    bundle = {
        "model": model,
        "feature_names": list(X.columns),
        "classes": labels,
        "random_state": RANDOM_STATE,
        "window_samples": 20,
        "pdc_rate_hz": 50.0,
    }
    joblib.dump(bundle, args.model_out)

    metrics = {
        "windows": int(len(X)),
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "classes": labels,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "model": "MLP(32,16)",
        "feature_count": int(X.shape[1]),
        "confusion_matrix": cm.tolist(),
    }
    Path(args.model_out).with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    meta.to_csv(Path(args.model_out).with_suffix(".dataset.csv"), index=False)

    print(f"\nModel saved: {args.model_out}")


if __name__ == "__main__":
    main()
