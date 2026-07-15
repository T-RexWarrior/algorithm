"""Regression metrics with safe handling of degenerate station series."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size < 2 or np.nanstd(y_true) == 0:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if not y_true.size:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "count": 0}
    errors = y_pred - y_true
    mse = float(np.mean(errors**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(errors))),
        "r2": safe_r2(y_true, y_pred),
        "count": int(y_true.size),
    }
