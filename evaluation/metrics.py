from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred, labels=None):
    labels = labels or sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "labels": labels,
        "confusion_matrix": cm,
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
