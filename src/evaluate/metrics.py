"""
metrics.py — Değerlendirme Metrikleri
Accuracy, Precision, Recall, F1 + seed agregasyonu.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def aggregate_seed_results(metrics_list: list) -> dict:
    """
    5 seed sonucunu ortalama ± std olarak özetler.

    Args:
        metrics_list: [{"accuracy": .., "f1": ..}, ...]  (her seed için bir dict)
    Returns:
        {"accuracy_mean": .., "accuracy_std": .., "f1_mean": .., "f1_std": .., ...}
    """
    keys = metrics_list[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"]  = float(np.std(vals))
    return result
