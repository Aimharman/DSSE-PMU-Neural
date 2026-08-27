from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from neural.controller import NeuralController
from neural.feature_extractor import extract_window_features

LABELS = ["NORMAL", "SYNC", "CLOCK_DRIFT", "BAD_DATA"]


def infer_window_truth(window_df):
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


def iter_windows(df, window_size=128, stride=20):
    if len(df) < window_size:
        return [df.copy()]
    windows = []
    for start in range(0, len(df) - window_size + 1, stride):
        windows.append(df.iloc[start:start + window_size].copy())
    if not windows:
        windows = [df.copy()]
    return windows


def evaluate_windows(csv_path, model=None, window_size=128, stride=20):
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "confusion_matrix": np.zeros((1, 1), dtype=int), "fault_type_accuracy": 0.0, "faulty_pmu_accuracy": 0.0, "false_positive_rate": 0.0, "false_negative_rate": 0.0, "detection_latency": 0.0}

    controller = model or NeuralController()
    y_true, y_pred, actual_pmu, predicted_pmu = [], [], [], []
    fault_start = None
    for pmu in (1, 2, 3):
        for col in (f"PMU{pmu} Sync Fault Active", f"PMU{pmu} Clock Drift Fault", f"PMU{pmu} Bad Data"):
            if col in df.columns:
                active = df[col].fillna(False).astype(bool)
                if active.any():
                    candidate = float(df.loc[active, "Time (s)"].min()) if "Time (s)" in df.columns else 0.0
                    if fault_start is None or candidate < fault_start:
                        fault_start = candidate

    latencies = []
    for idx, window in enumerate(iter_windows(df, window_size=window_size, stride=stride)):
        true_label, true_pmu = infer_window_truth(window)
        features = np.nan_to_num(extract_window_features(window), nan=0.0, posinf=0.0, neginf=0.0)
        pred = controller.predict(features)
        if isinstance(pred, dict):
            pred_type = str(pred.get("fault_type", "NORMAL")).upper()
            pred_pmu = int(pred.get("faulty_pmu", 0) or 0)
        else:
            pred_type = str(pred[0]).upper() if len(pred) else "NORMAL"
            pred_pmu = 0
        y_true.append(true_label)
        y_pred.append(pred_type)
        actual_pmu.append(true_pmu)
        predicted_pmu.append(pred_pmu)
        if true_label != "NORMAL" and pred_type == true_label:
            if "Time (s)" in window.columns:
                window_start = float(window["Time (s)"].iloc[0])
            else:
                window_start = float(idx * stride)
            if fault_start is not None:
                latencies.append(max(0.0, window_start - fault_start))
            else:
                latencies.append(max(0.0, window_start))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    precision = precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

    normal_true = (y_true == "NORMAL")
    normal_pred = (y_pred == "NORMAL")
    false_positive_rate = float(np.sum((~normal_pred) & normal_true)) / max(1, np.sum(normal_true))

    anomaly_true = (y_true != "NORMAL")
    anomaly_pred = (y_pred == "NORMAL")
    false_negative_rate = float(np.sum(anomaly_true & anomaly_pred)) / max(1, np.sum(anomaly_true))

    fault_type_hits = float(np.sum((y_true != "NORMAL") & (y_pred == y_true))) / max(1, np.sum(y_true != "NORMAL"))
    actual_pmu = np.asarray(actual_pmu)
    predicted_pmu = np.asarray(predicted_pmu)
    pmu_hits = float(np.sum((actual_pmu != 0) & (predicted_pmu == actual_pmu))) / max(1, np.sum(actual_pmu != 0))

    detection_latency = float(np.min(latencies)) if latencies else 0.0
    result = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm,
        "fault_type_accuracy": fault_type_hits,
        "faulty_pmu_accuracy": pmu_hits,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "detection_latency": detection_latency,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Blind temporal-window evaluation for the final PMU pipeline.")
    parser.add_argument("csv", type=str, nargs="?", default="data/scenarios/normal_pmu1_r01.csv")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=20)
    args = parser.parse_args()
    print(evaluate_windows(args.csv, window_size=args.window_size, stride=args.stride))


if __name__ == "__main__":
    main()
