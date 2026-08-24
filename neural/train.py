from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    from .feature_extractor import extract_window_features
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from neural.feature_extractor import extract_window_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "scenarios"
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "controller.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

FAULT_CLASSES = ["NORMAL", "BAD_DATA", "SYNC", "CLOCK_DRIFT"]
PMU_CLASSES = [1, 2, 3]


def _scenario_files() -> list[Path]:
    primary = sorted(DATA_DIR.glob("*.csv"))
    if primary:
        return primary
    raise FileNotFoundError(f"No scenario CSV files found under {DATA_DIR}")


def _scenario_kind(path: Path) -> str:
    name = path.name.lower()
    if "normal" in name:
        return "NORMAL"
    if "bad_data" in name:
        return "BAD_DATA"
    if "sync" in name:
        return "SYNC"
    if "clock_drift" in name:
        return "CLOCK_DRIFT"
    raise ValueError(f"Unsupported scenario filename: {path.name}")


def _scenario_pmu(path: Path) -> int:
    match = re.search(r"pmu(\d)", path.name.lower())
    return int(match.group(1)) if match else 0


def _window_label(window_df):
    for pmu in (1, 2, 3):
        sync_active = window_df.get(f"PMU{pmu} Sync Fault Active", pd.Series([False] * len(window_df)))
        drift_active = window_df.get(f"PMU{pmu} Clock Drift Fault", pd.Series([False] * len(window_df)))
        bad_data = window_df.get(f"PMU{pmu} Bad Data", pd.Series([False] * len(window_df)))
        if bool(sync_active.fillna(False).astype(bool).any()):
            return "SYNC", pmu
        if bool(drift_active.fillna(False).astype(bool).any()):
            return "CLOCK_DRIFT", pmu
        if bool(bad_data.fillna(False).astype(bool).any()):
            return "BAD_DATA", pmu
    return "NORMAL", 0


def _iter_windows(df, window_size=128, stride=20):
    if df.empty:
        return []
    if len(df) < window_size:
        return [df.copy()]
    return [df.iloc[start:start + window_size].copy() for start in range(0, len(df) - window_size + 1, stride)]


def _collect_scenario_windows(csv_path, window_size=128, stride=20, max_windows_per_file=400):
    df = pd.read_csv(csv_path)
    samples = []
    windows = _iter_windows(df, window_size=window_size, stride=stride)
    if len(windows) > max_windows_per_file:
        idx = np.linspace(0, len(windows) - 1, max_windows_per_file, dtype=int)
        windows = [windows[i] for i in idx]
    for window in windows:
        label, pmu = _window_label(window)
        features = extract_window_features(window)
        samples.append({
            "features": np.asarray(features, dtype=float),
            "fault_type": label,
            "faulty_pmu": int(pmu),
        })
    return samples


def train_controller(window_size=128, stride=20, random_seed=42, max_windows_per_file=400):
    scenario_files = _scenario_files()

    grouped: dict[str, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in scenario_files:
        label = _scenario_kind(path)
        pmu = _scenario_pmu(path)
        grouped[label][pmu].append(path)

    train_files, val_files = [], []
    for label in FAULT_CLASSES:
        files_by_pmu = grouped.get(label, {})
        if label == "NORMAL":
            normal_files = sorted(files_by_pmu.get(0, []))
            if len(normal_files) >= 2:
                train_group, val_group = train_test_split(normal_files, test_size=1 / 3, random_state=random_seed, shuffle=True)
            else:
                train_group, val_group = normal_files, []
            train_files.extend(train_group)
            val_files.extend(val_group)
            continue

        for pmu in sorted(PMU_CLASSES):
            pmu_files = sorted(files_by_pmu.get(pmu, []))
            if not pmu_files:
                continue
            if len(pmu_files) >= 2:
                pmu_train, pmu_val = train_test_split(
                    pmu_files,
                    test_size=1 / 3,
                    random_state=random_seed + pmu,
                    shuffle=True,
                )
            else:
                pmu_train, pmu_val = pmu_files, []
            train_files.extend(pmu_train)
            val_files.extend(pmu_val)

    X_train, y_train, pmu_train = [], [], []
    X_val, y_val, pmu_val = [], [], []

    for csv_path in train_files:
        for sample in _collect_scenario_windows(csv_path, window_size=window_size, stride=stride, max_windows_per_file=max_windows_per_file):
            X_train.append(sample["features"])
            y_train.append(sample["fault_type"])
            pmu_train.append(sample["faulty_pmu"])

    for csv_path in val_files:
        for sample in _collect_scenario_windows(csv_path, window_size=window_size, stride=stride, max_windows_per_file=max_windows_per_file):
            X_val.append(sample["features"])
            y_val.append(sample["fault_type"])
            pmu_val.append(sample["faulty_pmu"])

    if not X_train or not X_val:
        raise RuntimeError("Training and validation windows are empty. Check the scenario corpus under data/scenarios/")

    X_train = np.vstack(X_train)
    X_val = np.vstack(X_val)
    y_train = np.asarray(y_train, dtype=object)
    y_val = np.asarray(y_val, dtype=object)
    pmu_train = np.asarray(pmu_train, dtype=int)
    pmu_val = np.asarray(pmu_val, dtype=int)

    train_counts = {label: int(np.sum(y_train == label)) for label in FAULT_CLASSES}
    print("Window-level class distribution (train):", train_counts)
    print("Validation class distribution:", {label: int(np.sum(y_val == label)) for label in FAULT_CLASSES})

    type_model = RandomForestClassifier(
        n_estimators=120,
        random_state=random_seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    type_model.fit(X_train, y_train)

    fault_mask = y_train != "NORMAL"
    pmu_model = RandomForestClassifier(
        n_estimators=120,
        random_state=random_seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    pmu_model.fit(X_train[fault_mask], pmu_train[fault_mask])

    val_pred = type_model.predict(X_val)
    validation_accuracy = float(np.mean(val_pred == y_val))

    payload = {
        "fault_type_model": type_model,
        "pmu_model": pmu_model,
        "feature_count": int(X_train.shape[1]),
        "classes": FAULT_CLASSES,
        "faulty_pmu_classes": PMU_CLASSES,
        "training_scenarios": [p.name for p in train_files],
        "validation_scenarios": [p.name for p in val_files],
        "sklearn_version": sklearn.__version__,
        "random_seed": int(random_seed),
        "validation_accuracy": validation_accuracy,
    }

    joblib.dump(payload, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as fp:
        json.dump({
            "feature_count": int(X_train.shape[1]),
            "classes": FAULT_CLASSES,
            "faulty_pmu_classes": PMU_CLASSES,
            "training_scenarios": [p.name for p in train_files],
            "validation_scenarios": [p.name for p in val_files],
            "sklearn_version": sklearn.__version__,
            "random_seed": int(random_seed),
            "validation_accuracy": validation_accuracy,
        }, fp, indent=2)

    print(f"Saved canonical model to {MODEL_PATH}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Classes: {FAULT_CLASSES}")
    print(f"PMU classes: {PMU_CLASSES}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    return payload


if __name__ == "__main__":
    train_controller()
