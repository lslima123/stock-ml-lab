from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier: float
    log_loss: float
    actual_positive_rate: float
    predicted_positive_rate: float
    mean_probability: float
    ece: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _validate(y_true, probability) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("y_true and probability must be non-empty and aligned.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Classification targets must contain only 0 and 1.")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Probabilities must be finite and lie in [0, 1].")
    return y, p


def calibration_table(
    y_true,
    probability,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Quantile reliability table with weighted absolute calibration error.

    Quantile bins are used so sparse tails do not create many empty reliability
    bins on modest OOS samples.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    y, p = _validate(y_true, probability)

    # Rank first so equal probabilities still yield deterministic bins.
    ranks = pd.Series(p).rank(method="first", pct=True).to_numpy()
    bin_id = np.minimum((ranks * n_bins).astype(int), n_bins - 1)

    rows = []
    for bucket in range(n_bins):
        mask = bin_id == bucket
        if not np.any(mask):
            continue
        mean_p = float(p[mask].mean())
        observed = float(y[mask].mean())
        rows.append(
            {
                "bin": bucket + 1,
                "count": int(mask.sum()),
                "mean_probability": mean_p,
                "observed_positive_rate": observed,
                "calibration_gap": observed - mean_p,
                "abs_calibration_gap": abs(observed - mean_p),
            }
        )
    return pd.DataFrame(rows).set_index("bin")


def expected_calibration_error(y_true, probability, *, n_bins: int = 10) -> float:
    table = calibration_table(y_true, probability, n_bins=n_bins)
    weights = table["count"].to_numpy(dtype=float)
    gaps = table["abs_calibration_gap"].to_numpy(dtype=float)
    return float(np.average(gaps, weights=weights))


def classification_metrics(
    y_true,
    probability,
    *,
    classification_threshold: float = 0.5,
    calibration_bins: int = 10,
) -> ClassificationMetrics:
    y, p = _validate(y_true, probability)
    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must lie strictly between 0 and 1.")

    pred = (p >= classification_threshold).astype(int)
    auc = float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=auc,
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        actual_positive_rate=float(y.mean()),
        predicted_positive_rate=float(pred.mean()),
        mean_probability=float(p.mean()),
        ece=expected_calibration_error(y, p, n_bins=calibration_bins),
    )
