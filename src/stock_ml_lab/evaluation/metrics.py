from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    directional_accuracy: float
    directional_coverage: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _validate_1d(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {true.shape} and {pred.shape}."
        )
    if true.size == 0:
        raise ValueError("Metrics require at least one observation.")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require finite y_true and y_pred values.")

    return true, pred


def directional_metrics(
    y_true,
    y_pred,
    *,
    zero_tolerance: float = 1e-12,
) -> tuple[float, float]:
    """Return directional accuracy and coverage.

    Predictions with magnitude <= zero_tolerance are treated as abstentions.
    This prevents the zero-return baseline from receiving an artificial
    directional score.
    """
    true, pred = _validate_1d(y_true, y_pred)

    active = np.abs(pred) > zero_tolerance
    coverage = float(active.mean())

    if not active.any():
        return float("nan"), coverage

    accuracy = float((np.sign(pred[active]) == np.sign(true[active])).mean())
    return accuracy, coverage


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    true, pred = _validate_1d(y_true, y_pred)
    directional_accuracy, directional_coverage = directional_metrics(true, pred)

    return RegressionMetrics(
        mae=float(mean_absolute_error(true, pred)),
        rmse=float(root_mean_squared_error(true, pred)),
        directional_accuracy=directional_accuracy,
        directional_coverage=directional_coverage,
    )
