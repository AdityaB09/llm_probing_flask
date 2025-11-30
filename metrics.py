from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_basic_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1}


def expected_calibration_error(y_true, probas, n_bins: int = 10) -> float:
    """
    Simple ECE for multiclass classification.
    y_true: (N,) int labels
    probas: (N, C) predicted probabilities
    """
    probas = np.asarray(probas)
    confidences = probas.max(axis=1)
    predictions = probas.argmax(axis=1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > start) & (confidences <= end)
        if not np.any(mask):
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = (predictions[mask] == np.asarray(y_true)[mask]).mean()
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)
