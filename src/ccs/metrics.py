"""Evaluation metrics: MAE, R² and Pearson's r."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "mean_error": float(err.mean()),
        "std_error": float(err.std()),
    }
